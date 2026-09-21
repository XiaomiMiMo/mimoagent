"""Codex ``exec_command``: run a shell command.

Keeps the nested ``tools.exec_command`` argument names from the Codex catalogue
(``cmd`` / ``workdir`` / ``max_output_tokens``), but not upstream's unified-exec
semantics: PTY sessions, ``write_stdin`` and yield-and-resume are deliberately
absent because SWE-RL needs the command to actually finish. Each call runs to
completion in a fresh shell, capped by ``max_timeout``.

The deadline keeps upstream's parameter name, ``yield_time_ms``, because that
is the name the checkpoint was trained on (56–61% of exec_command calls on the
0907 galaxy run carried it, and the ``timeout_ms`` that ``cfef3f1d`` advertised
instead seeded a rising ``timeout_time_ms`` blend). Its meaning here is the
one-shot one and the description says so: the command is terminated when the
value expires and there is no session to resume, unlike the cell-level
``// @exec:`` pragma of the same name. The value is honoured as given (clamped
to ``[1s, max_timeout]``): a small value is a short deadline, not a preview.
``timeout_ms`` and every other unknown field are rejected so the checkpoint
gets a signal instead of a silently ignored parameter.

Hitting the deadline is not an exception: it returns an unsuccessful result
carrying the partial output, so a nested call in a ``Promise.all`` cannot
discard its siblings' results. The description and the parameter docs must
state that, not upstream's interactive contract. Timeout text is rendered from
config for the same reason ``tools/cc/bash.py`` does it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput
from mimoagent.tools.codex.output_budget import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    UNIFIED_EXEC_OUTPUT_MAX_BYTES,
    collect_head_tail,
    resolve_max_tokens,
)


@dataclass
class ExecCommandConfig(ToolConfig):
    """Exec command deadline limits.

    Output is not budgeted here. The tool returns what the process wrote,
    bounded only by the 1 MiB collection layer; the ``max_output_tokens`` the
    model passed travels in the result metadata and is applied by the caller
    (the code-mode runtime for a nested call, the agent for a direct one), the
    way codex-rs applies it in ``code_mode_result`` / ``to_response_item``
    rather than in the process runner.
    """

    timeout: int = 30
    max_timeout: int = 300
    cwd: str = ""

    def __post_init__(self) -> None:
        if self.max_timeout < self.timeout:
            raise ValueError(
                f"exec_command max_timeout ({self.max_timeout}s) must be >= default timeout ({self.timeout}s)"
            )


class ExecCommandTool(BaseTool):
    """Run a command; Codex name/schema (``cmd``, not ``command``)."""

    def _create_config(self, config_dict: dict[str, Any]) -> ExecCommandConfig:
        return ExecCommandConfig(**config_dict)

    @property
    def name(self) -> str:
        return "exec_command"

    @property
    def description(self) -> str:
        default_ms = self.config.timeout * 1000
        max_ms = self.config.max_timeout * 1000
        return f"""Runs a shell command to completion and returns its output.

Each invocation runs in a fresh shell — directory and environment changes inside `cmd` do NOT persist. Pass `workdir` (or prefix with `cd /path && ...`) instead of relying on a persisted working directory. `workdir` defaults to the working directory stated in your instructions.

The command runs until it exits or hits the `yield_time_ms` deadline ({default_ms}ms by default, up to {max_ms}ms), at which point it is terminated; its partial output is returned as an unsuccessful result rather than an error. Unlike the cell-level `// @exec:` pragma of the same name, there is no session to resume: nothing keeps running after the call returns, so set `yield_time_ms` to the time the command genuinely needs (a full test run may need the maximum) and do not start long-lived processes expecting to attach to them later.

When you search for text or files, reach first for `rg` or `rg --files`; they are much faster than alternatives like `grep`. Do not create or edit files with `cat` or other shell write tricks — use `apply_patch`.

Do not chain shell commands with separators like `echo "====";` or `printf '---'`. Exercise caution when escaping text — backticks and `$()` passed to `cmd` will still execute. Use non-interactive flags; avoid vi/nano or anything requiring stdin."""

    def get_function_parameters(self) -> dict[str, Any]:
        default_ms = self.config.timeout * 1000
        max_ms = self.config.max_timeout * 1000
        return {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "max_output_tokens": {
                    "type": "number",
                    # Upstream's sentence verbatim (shell_spec.rs). It describes the
                    # direct-call policy; a nested call gets the collected output as
                    # is when the field is omitted, also as upstream.
                    "description": (
                        f"Output token budget. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens; "
                        "larger requests may be capped by policy."
                    ),
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the working directory stated in your instructions.",
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": (
                        f"Maximum command runtime, in milliseconds; the command is terminated when it "
                        f"expires and cannot be resumed. Defaults to {default_ms}ms and is capped at {max_ms}ms."
                    ),
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for exec_command")
        if not isinstance(params, dict):
            raise ToolException(f"exec_command params must be a dictionary, got {type(params).__name__}")

        # ``timeout_ms`` is deliberately not in this set: see the module docstring.
        unsupported = set(params) - {"cmd", "max_output_tokens", "workdir", "yield_time_ms"}
        if unsupported:
            raise ToolException(f"exec_command has unsupported field(s): {', '.join(sorted(unsupported))}")

        cmd = (params.get("cmd") or "").strip()
        if not cmd:
            raise ToolException("exec_command requires a non-empty 'cmd' parameter")

        workdir = (params.get("workdir") or "").strip() or self.config.cwd
        if workdir and not Path(workdir).is_absolute():
            env_cwd = getattr(getattr(env, "config", None), "cwd", "") or ""
            if env_cwd:
                workdir = str(Path(env_cwd) / workdir)
        timeout_s = self._resolve_timeout(params.get("yield_time_ms"))

        t0 = time.monotonic()
        try:
            result = env.execute(cmd, cwd=workdir, timeout=timeout_s)
        except TransportError:
            raise
        except Exception as e:
            raise ToolException(f"Failed to execute command: {e}")
        wall = round(time.monotonic() - t0, 3)

        output = result.get("output", "") or ""
        returncode = result.get("returncode")
        reason = result.get("reason", "ok")

        if reason == "transport_error":
            return ToolOutput(
                output=f"Error: Transport error while executing command:\n{output}",
                success=False,
                metadata={
                    "exit_code": returncode,
                    "wall_time_seconds": wall,
                    "reason": reason,
                },
            )

        # Layer 1 of the upstream stack: the process reader keeps 1 MiB, head
        # and tail. The token budget the model asked for is NOT applied here;
        # it rides in the metadata for the layer that owns it.
        requested = params.get("max_output_tokens")
        max_output_tokens = None if requested is None else resolve_max_tokens(requested)
        if reason in ("pod_timeout", "client_timeout"):
            # Not an exception: under ``ptc: true`` a raised timeout rejects the
            # nested promise, and one slow command in a ``Promise.all`` then
            # discards every sibling's output. The retry hint mirrors
            # ``tools/bash.py`` so both surfaces read the same way.
            max_ms = self.config.max_timeout * 1000
            output = (
                f"Error: Command timed out after {timeout_s}s. Partial output below. "
                f"Retry with a larger `yield_time_ms` (max {max_ms}ms), or use a narrower command.\n{output}"
            )
        collected, omitted_bytes = collect_head_tail(output, UNIFIED_EXEC_OUTPUT_MAX_BYTES)
        metadata: dict[str, Any] = {
            "wall_time_seconds": wall,
            "max_output_tokens": max_output_tokens,
            "output_omitted_bytes": omitted_bytes,
        }
        if reason in ("pod_timeout", "client_timeout"):
            metadata.update({"exit_code": returncode, "reason": reason, "timed_out": True})
            return ToolOutput(output=collected, success=False, metadata=metadata)
        rc = returncode if returncode is not None else -1
        metadata["exit_code"] = rc
        return ToolOutput(output=collected, success=rc == 0, metadata=metadata)

    def _resolve_timeout(self, yield_time_ms: Any) -> int:
        """Seconds to wait before the command is terminated.

        ``ms`` → ``s``, clamped to ``[1, max_timeout]``. There is deliberately no
        floor at ``config.timeout``: the previous ``max(requested, default)``
        mapped every request under the default onto it, so a model asking for a
        5s probe waited the full 60s and had no way to tell.
        """

        default_s = self.config.timeout
        max_s = self.config.max_timeout
        if yield_time_ms is None or isinstance(yield_time_ms, bool):
            return default_s
        try:
            requested_s = float(yield_time_ms) / 1000.0
        except (TypeError, ValueError):
            return default_s
        if requested_s != requested_s:  # NaN survives float(); int(NaN) raises
            return default_s
        return max(1, min(int(requested_s), max_s))
