"""MiMo-Code file-pattern matching tool."""

import posixpath
import shlex
import threading
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.mimocode.paths import resolve_path, session_cwd
from mimoagent.tools.mimocode.prompts import load_tool_prompt
from mimoagent.tools.ripgrep import RipgrepUnavailable, resolve_host_rg


class GlobTool(BaseTool):
    """Resolve a glob pattern, sorted by mtime (newest first)."""

    _MAX_RESULTS = 100

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._rg_lock = threading.Lock()
        self._rg_ready_envs: set[int] = set()

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return load_tool_prompt("glob")

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. If not specified, the current working directory will be used. "
                        'IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" '
                        "- simply omit it for the default behavior. Must be a valid directory path if provided."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for glob execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        pattern = params.get("pattern")
        if not pattern:
            raise ToolException("Missing required parameter: pattern")

        self._ensure_rg(env)

        search = resolve_path(str(params["path"]), context) if params.get("path") else session_cwd(context) or "."
        path_type = self._path_type(env, search, context)
        if path_type == "FILE":
            return ToolOutput(output=f"Error: glob path must be a directory: {search}", success=False)
        if path_type != "DIR":
            return ToolOutput(output=f"Error: Path does not exist: {search}", success=False)

        argv = ["rg", "--no-config", "--files", "--hidden", "--glob=!.git/*", f"--glob={pattern}", "."]
        command = " ".join(shlex.quote(arg) for arg in argv)
        result = env.execute(command, cwd=search)
        returncode = result.get("returncode", 1)
        if returncode not in (0, 1):
            return ToolOutput(
                output=f"Error: ripgrep failed with exit code {returncode}: {result.get('output', '').strip()}",
                success=False,
            )

        paths = []
        for raw in result.get("output", "").splitlines():
            if not raw.strip():
                continue
            path = raw.replace("\\", "/")
            paths.append(path if path.startswith("/") else posixpath.normpath(posixpath.join(search, path)))

        if not paths:
            return ToolOutput(output="No files found", success=True, metadata={"count": 0, "truncated": False})

        # Match Mimocode's 100-result presentation cap and report whether more
        # paths were available in the source stream.
        #
        # Upstream keeps the first 100 paths off a lazy ripgrep stream and only
        # then sorts by mtime, so its candidate set is whatever readdir order
        # surfaced first — it never promised the globally newest files either.
        # It can afford that because Stream.take terminates the walk early; here
        # env.execute already returned the whole listing, so ordering by path
        # first costs nothing and makes one tree state always yield the same 100
        # candidates. ``path`` is also the mtime tie-break, because a freshly
        # checked-out testbed gives nearly every file the same mtime.
        paths.sort()
        truncated = len(paths) > self._MAX_RESULTS
        paths = paths[: self._MAX_RESULTS]
        mtimes = self._file_mtimes(env, paths, context)
        paths.sort(key=lambda path: (-mtimes.get(path, 0), path))

        output = [*paths]
        if truncated:
            output.extend(
                [
                    "",
                    "(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)",
                ]
            )
        return ToolOutput(
            output="\n".join(output),
            success=True,
            metadata={"count": len(paths), "truncated": truncated},
        )

    @staticmethod
    def _path_type(env: Any, path: str, context: dict[str, Any] | None) -> str:
        quoted = shlex.quote(path)
        probe = f"if [ -f {quoted} ]; then echo FILE; elif [ -d {quoted} ]; then echo DIR; else echo NOTFOUND; fi"
        return env.execute(probe, cwd=session_cwd(context) or None).get("output", "").strip()

    @staticmethod
    def _file_mtimes(env: Any, paths: list[str], context: dict[str, Any] | None) -> dict[str, int]:
        if not paths:
            return {}
        commands = ["set --"]
        for path in paths:
            commands.append(f'set -- "$@" {shlex.quote(path)}')
        commands.append(
            'i=0; for f in "$@"; do i=$((i+1)); '
            'printf \'%s\\t%s\\n\' "$i" "$(stat -c %Y -- "$f" 2>/dev/null || echo 0)"; done'
        )
        result = env.execute("; ".join(commands), cwd=session_cwd(context) or None)
        mtimes: dict[str, int] = {}
        for line in result.get("output", "").splitlines():
            try:
                ordinal, value = line.split("\t", 1)
                mtimes[paths[int(ordinal) - 1]] = int(value)
            except (ValueError, IndexError):
                continue
        return mtimes

    def _ensure_rg(self, env: Any) -> None:
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
                    f"ripgrep is not installed in the environment and none could be resolved on the host: {e}"
                ) from e
            try:
                env.copy_to(str(host_rg), "/usr/local/bin/rg")
            except Exception as e:
                raise ToolException(f"Failed to stage rg into the environment: {e}")
            res = env.execute("chmod +x /usr/local/bin/rg && command -v rg")
            if res.get("returncode", 1) != 0 or not res.get("output", "").strip():
                raise ToolException(f"Staged rg but it is not runnable: {res.get('output', '')[:300]}")
            self._rg_ready_envs.add(env_key)
