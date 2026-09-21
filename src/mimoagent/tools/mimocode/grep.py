"""MiMo-Code content search tool.

The public contract intentionally matches MiMo-Code's native ``grep`` tool:
the model supplies only a regex, an optional path, and an optional include
glob. Results are matching lines (not a selectable files/count mode), ordered
by file modification time and capped at 100 matches.
"""

import json
import posixpath
import shlex
import threading
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.mimocode.paths import resolve_path, session_cwd
from mimoagent.tools.mimocode.prompts import load_tool_prompt
from mimoagent.tools.ripgrep import RipgrepUnavailable, resolve_host_rg

_RG_DEST = "/usr/local/bin/rg"


_MAX_LINE_LENGTH = 2000
_MAX_MATCHES = 100
# Paths per mtime probe. Each contributes a quoted `set -- "$@" <path>` clause to
# the command string, so the batch has to stay well under MAX_ARG_STRLEN even for
# long nested paths.
_MTIME_BATCH = 500


class GrepTool(BaseTool):
    """Search file contents with ripgrep using the MiMo-Code contract."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Guards the stage-on-first-use so parallel Grep calls in one turn
        # don't race a half-copied binary. Per tool instance == per agent.
        self._rg_lock = threading.Lock()
        self._rg_ready_envs: set[int] = set()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return load_tool_prompt("grep")

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in. Defaults to the current working directory.",
                },
                "include": {
                    "type": "string",
                    "description": 'File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")',
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for grep execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        pattern = params.get("pattern")
        if not pattern:
            raise ToolException("Missing required parameter: pattern")

        self._ensure_rg(env)
        # Match upstream path semantics: relative paths are rooted at the
        # agent session cwd, and a file path searches only that file.
        requested_path = resolve_path(str(params["path"]), context) if params.get("path") else session_cwd(context)
        requested_path = requested_path or "."
        path_type = self._path_type(env, requested_path, context)
        if path_type == "NOTFOUND":
            return ToolOutput(output=f"Error: Path does not exist: {requested_path}", success=False)
        if path_type not in ("FILE", "DIR"):
            return ToolOutput(output=f"Error: Could not inspect path: {requested_path}", success=False)

        if path_type == "DIR":
            search_cwd = requested_path
            search_target = "."
        else:
            search_cwd = posixpath.dirname(requested_path) or "/"
            search_target = posixpath.basename(requested_path)

        argv = [
            "rg",
            "--no-config",
            "--json",
            "--hidden",
            "--glob=!.git/*",
            "--no-messages",
        ]
        include = params.get("include")
        if include:
            argv.append(f"--glob={include}")
        argv.extend(["--", str(pattern), search_target])
        command = " ".join(shlex.quote(arg) for arg in argv)
        result = env.execute(command, cwd=search_cwd)
        returncode = result.get("returncode", 1)
        if returncode == 2:
            return ToolOutput(
                output=f"Error: ripgrep failed to search with this pattern: {result.get('output', '').strip()}",
                success=False,
            )
        if returncode not in (0, 1):
            return ToolOutput(
                output=f"Error: ripgrep failed with exit code {returncode}: {result.get('output', '').strip()}",
                success=False,
            )

        rows = self._parse_matches(result.get("output", ""), search_cwd)
        if not rows:
            return ToolOutput(output="No files found", success=True, metadata={"matches": 0, "truncated": False})

        mtimes = self._file_mtimes(env, rows, context)
        # ``path``/``line`` break mtime ties so the shown slice is reproducible.
        # Upstream sorts on mtime alone; a freshly checked-out
        # testbed gives nearly every file one identical mtime, which would leave
        # the 100-match cut entirely up to ripgrep's readdir order.
        rows.sort(key=lambda row: (-mtimes.get(row["path"], 0), row["path"], row["line"]))

        total = len(rows)
        truncated = total > _MAX_MATCHES
        shown = rows[:_MAX_MATCHES]
        output = [f"Found {total} matches" + (f" (showing first {_MAX_MATCHES})" if truncated else "")]

        current_path = ""
        for row in shown:
            path = row["path"]
            if current_path != path:
                if current_path:
                    output.append("")
                current_path = path
                output.append(f"{path}:")
            line = row["text"]
            if len(line) > _MAX_LINE_LENGTH:
                line = line[:_MAX_LINE_LENGTH] + "..."
            output.append(f"  Line {row['line']}: {line}")

        if truncated:
            output.extend(
                [
                    "",
                    (
                        f"(Results truncated: showing {_MAX_MATCHES} of {total} matches "
                        f"({total - _MAX_MATCHES} hidden). Consider using a more specific path or pattern.)"
                    ),
                ]
            )
        return ToolOutput(
            output="\n".join(output),
            success=True,
            metadata={"matches": total, "truncated": truncated},
        )

    @staticmethod
    def _path_type(env: Any, path: str, context: dict[str, Any] | None) -> str:
        quoted = shlex.quote(path)
        probe = f"if [ -f {quoted} ]; then echo FILE; elif [ -d {quoted} ]; then echo DIR; else echo NOTFOUND; fi"
        return env.execute(probe, cwd=session_cwd(context) or None).get("output", "").strip()

    @staticmethod
    def _parse_matches(output: str, search_cwd: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_line in output.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            raw_path = ((data.get("path") or {}).get("text") or "").replace("\\", "/")
            if not raw_path:
                continue
            path = raw_path if raw_path.startswith("/") else posixpath.normpath(posixpath.join(search_cwd, raw_path))
            text = str((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
            rows.append({"path": path, "line": int(data.get("line_number") or 0), "text": text})
        return rows

    @staticmethod
    def _file_mtimes(env: Any, rows: list[dict[str, Any]], context: dict[str, Any] | None) -> dict[str, int]:
        paths = list(dict.fromkeys(row["path"] for row in rows))
        if not paths:
            return {}
        # Batched because the whole path list travels inside the command string:
        # one `set -- "$@" <path>` per file crosses Linux MAX_ARG_STRLEN (128 KiB)
        # at roughly 2-3k files and the exec fails outright with E2BIG. A wide
        # pattern on a real testbed hits that easily, and the mtime sort needs
        # every path, so split the probe instead of shrinking the input.
        mtimes: dict[str, int] = {}
        cwd = session_cwd(context) or None
        for start in range(0, len(paths), _MTIME_BATCH):
            batch = paths[start : start + _MTIME_BATCH]
            commands = ["set --"]
            for path in batch:
                commands.append(f'set -- "$@" {shlex.quote(path)}')
            commands.append(
                'i=0; for f in "$@"; do i=$((i+1)); '
                'printf \'%s\\t%s\\n\' "$i" "$(stat -c %Y -- "$f" 2>/dev/null || echo 0)"; done'
            )
            result = env.execute("; ".join(commands), cwd=cwd)
            for line in result.get("output", "").splitlines():
                try:
                    ordinal, value = line.split("\t", 1)
                    mtimes[batch[int(ordinal) - 1]] = int(value)
                except (ValueError, IndexError):
                    continue
        return mtimes

    def _ensure_rg(self, env: Any) -> None:
        """Make sure ``rg`` is runnable in ``env``, staging a static binary on
        first use if the image doesn't ship one.

        Lazy by design (per user decision): environments whose rollouts never
        call Grep pay no copy cost. The per-env memo makes the probe a no-op
        after the first successful call.
        """
        env_key = id(env)
        if env_key in self._rg_ready_envs:
            return
        with self._rg_lock:
            if env_key in self._rg_ready_envs:
                return
            which = env.execute("command -v rg || true").get("output", "").strip()
            if which:
                self._rg_ready_envs.add(env_key)
                return
            try:
                host_rg = resolve_host_rg()
            except RipgrepUnavailable as e:
                raise ToolException(
                    f"ripgrep (rg) is not installed in this environment and no static binary could be "
                    f"resolved on the host: {e}. The Grep tool requires rg."
                ) from e
            try:
                env.copy_to(str(host_rg), _RG_DEST)
            except Exception as e:
                raise ToolException(f"Failed to stage rg into the environment: {e}")
            res = env.execute(f"chmod +x {shlex.quote(_RG_DEST)} && command -v rg")
            if res.get("returncode", 1) != 0 or not res.get("output", "").strip():
                raise ToolException(f"Staged rg at {_RG_DEST} but it is not runnable: {res.get('output', '')[:300]}")
            self._rg_ready_envs.add(env_key)
