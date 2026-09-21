"""MiMo non-Codex Bash tool.

The execution path is intentionally close to the copied CC tool. MiMo's
session cwd is supplied by the agent context; Bash itself starts a fresh shell,
so a command-local ``cd`` does not persist.

Output is bounded here with MiMo-Code's budgets (``bash.ts``: 2000 lines /
50 KB) but in the head+tail shape of the agent-level ``truncate_middle``: the
beginning and the end survive, with a marker stating what was cut between
them. bash.ts itself keeps only the tail and points the model at a spool file
holding the full output; this harness has no such file, so dropping the head
would lose a test run's first traceback for good. ``MimocodeAgentConfig`` sets
``max_observation_length`` to 0 so this is the only cut.
"""

from dataclasses import dataclass
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput
from mimoagent.tools.mimocode.paths import resolve_path, session_cwd
from mimoagent.tools.mimocode.prompts import load_bash_prompt

TRUNCATION_PREFIX = "...output truncated...\n"


@dataclass
class BashToolConfig(ToolConfig):
    timeout: int = 30
    max_timeout: int = 300
    cwd: str = ""
    # MiMo-Code's bash budget (Hh / th in bash.ts), shared between head and tail.
    max_output_lines: int = 2000
    max_output_bytes: int = 51200

    def __post_init__(self) -> None:
        if self.max_timeout < self.timeout:
            raise ValueError(f"Bash max_timeout ({self.max_timeout}s) must be >= default timeout ({self.timeout}s)")
        if self.max_output_lines < 1 or self.max_output_bytes < 1:
            raise ValueError("Bash max_output_lines and max_output_bytes must be positive")


def keep_tail(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    """Return ``(text, False)`` when it fits, else the last lines within both budgets and ``True``.

    Mirrors MiMo-Code's tail truncation: walk lines from the end, stop at the
    first line that would overflow either budget. A single line that is itself
    over the byte budget keeps its last ``max_bytes`` bytes on a UTF-8 boundary,
    so an oversized one-line dump still shows its ending.
    """
    lines = text.split("\n")
    if len(lines) <= max_lines and len(text.encode("utf-8")) <= max_bytes:
        return text, False
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if len(kept) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if kept else 0)
        if size + line_bytes > max_bytes:
            if size == 0:
                # Nothing but the empty field after a trailing newline is kept
                # yet, so this oversized line is the whole observation. (bash.ts
                # only does this when the trailing field is absent and returns
                # an empty body otherwise.)
                available = max_bytes - (1 if kept else 0)
                if available > 0:  # bytes[-0:] would be the whole line
                    kept.append(line.encode("utf-8")[-available:].decode("utf-8", errors="ignore"))
            break
        kept.append(line)
        size += line_bytes
    kept.reverse()
    return "\n".join(kept), True


def keep_head(text: str, max_lines: int, max_bytes: int) -> str:
    """The first lines of ``text`` within both budgets."""
    kept: list[str] = []
    size = 0
    for line in text.split("\n"):
        if len(kept) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if kept else 0)
        if size + line_bytes > max_bytes:
            break
        kept.append(line)
        size += line_bytes
    return "\n".join(kept)


def bound_output(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    """Return ``(observation, truncated)`` for a command's output.

    Within budget the text passes through. Otherwise the head gets up to half
    of each budget, the tail whatever is left, and a marker between them says
    how much was cut. The head is a prefix and the tail a suffix of the
    original with no overlap, so the marker always names a real gap.
    """
    if len(text.split("\n")) <= max_lines and len(text.encode("utf-8")) <= max_bytes:
        return text, False
    head = keep_head(text, max_lines // 2, max_bytes // 2)
    rest = text[len(head) :]
    if head and rest.startswith("\n"):
        rest = rest[1:]
    head_lines = head.count("\n") + 1 if head else 0
    tail, _ = keep_tail(rest, max_lines - head_lines, max_bytes - len(head.encode("utf-8")))
    middle = rest[: len(rest) - len(tail)]
    if tail and middle.endswith("\n"):
        middle = middle[:-1]
    omitted_bytes = len(middle.encode("utf-8"))
    omitted_lines = middle.count("\n") + 1 if middle else 0
    marker = f"...{omitted_lines} lines ({omitted_bytes} bytes) omitted here...\n"
    return f"{TRUNCATION_PREFIX}{head}\n{marker}{tail}" if head else f"{TRUNCATION_PREFIX}{marker}{tail}", True


class BashTool(BaseTool):
    """Execute one shell command in the current session directory."""

    def _create_config(self, config_dict: dict[str, Any]) -> BashToolConfig:
        return BashToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return load_bash_prompt(
            self.config.timeout,
            self.config.max_timeout,
            self.config.max_output_lines,
            self.config.max_output_bytes,
        )

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {
                    "type": "number",
                    "description": f"Optional timeout in milliseconds (capped at {self.config.max_timeout * 1000})",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command.",
                },
                "description": {
                    "type": "string",
                    "description": "A concise 5-10 word description of the command",
                },
            },
            "required": ["command", "description"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for bash execution")
        command = self._extract_command(params)
        timeout_s = self._resolve_timeout(
            params.get("timeout") if isinstance(params, dict) else None,
            self.config.timeout,
            self.config.max_timeout,
        )
        workdir = params.get("workdir") if isinstance(params, dict) else None
        cwd = resolve_path(str(workdir), context) if workdir else str(session_cwd(context) or self.config.cwd or "")
        try:
            result = env.execute(command, cwd=cwd, timeout=timeout_s)
        except TransportError:
            raise
        except Exception as e:
            raise ToolException(f"Failed to execute command: {e}")

        output = result.get("output", "")
        returncode = result.get("returncode")
        reason = result.get("reason", "ok")
        metadata = {"returncode": returncode, "reason": reason, "cwd": cwd}
        output, _truncated = bound_output(output, self.config.max_output_lines, self.config.max_output_bytes)
        if reason in ("pod_timeout", "client_timeout"):
            return ToolOutput(
                output=f"Error: Command timed out after {timeout_s}s. Partial output below.\n{output}",
                success=False,
                metadata=metadata,
            )
        if reason == "transport_error":
            return ToolOutput(
                output=f"Error: Transport error while executing command:\n{output}",
                success=False,
                metadata=metadata,
            )
        rc = returncode if returncode is not None else -1
        output = output or "(no output)"
        if rc != 0:
            # Metadata is not rendered, and for `false` / `test -e` / `grep -q`
            # the exit code is the entire result. Neutral wording: non-zero is
            # not always failure.
            output += f"\n(exit code {rc})"
        return ToolOutput(output=output, success=rc == 0, metadata={**metadata, "returncode": rc})

    @staticmethod
    def _extract_command(params: Any) -> str:
        if not isinstance(params, dict):
            raise ToolException(f"Bash tool params must be a dictionary, got {type(params).__name__}")
        command = (params.get("command") or "").strip()
        if not command:
            raise ToolException("Bash tool requires a non-empty 'command' parameter")
        return command

    @staticmethod
    def _resolve_timeout(raw_ms: Any, default_s: int, max_s: int) -> int:
        if raw_ms is None:
            return default_s
        try:
            requested_s = float(raw_ms) / 1000.0
        except (TypeError, ValueError):
            return default_s
        return max(1, min(int(requested_s), max_s))
