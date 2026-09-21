"""Bash command execution tool."""

from dataclasses import dataclass
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput


@dataclass
class BashToolConfig(ToolConfig):
    """Configuration for the Bash tool.

    ``timeout`` is the default per-command timeout (seconds) used when the
    model doesn't pass one; ``max_timeout`` caps whatever the model asks for
    (and the config default itself) so a single call can never hang a rollout
    for longer than this.
    """

    timeout: int = 30
    max_timeout: int = 600
    cwd: str = ""


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def _create_config(self, config_dict: dict[str, Any]) -> BashToolConfig:
        return BashToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return """Executes a given bash command and returns its output.

- Directory or environment variable changes are not persistent. Every action is executed in a new subshell. However, you can prefix any command with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Always use non-interactive flags (-y for apt) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- Commands are killed after the `timeout` parameter's seconds; pass a larger value for long-running commands (builds, test suites)
- Every tool call in a message runs concurrently, including edit and write: a command that reads, tests, or modifies a file can race with an edit or write of that file, or with another bash call writing it, issued in the same message."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command string to execute; can span multiple lines and is used exactly as provided.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Optional timeout in seconds for this command "
                        f"(default {self.config.timeout}, max {self.config.max_timeout}). "
                        f"Set a higher value for long-running commands (builds, test suites)."
                    ),
                },
            },
            "required": ["command"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for bash execution")

        command = self._extract_command(params)
        timeout = self._resolve_timeout(params)

        try:
            result = env.execute(command, cwd=self.config.cwd or "", timeout=timeout)
        except TransportError:
            raise  # Let infra errors propagate — execute_action handles them
        except Exception as e:
            raise ToolException(f"Failed to execute command: {e}")

        output = result.get("output", "")
        returncode = result.get("returncode")
        reason = result.get("reason", "ok")

        # Operation failures are error outputs, not exceptions (see
        # ToolException docstring): the model gets the same Error: shape as
        # from read/edit, plus the metadata line.
        if reason in ("pod_timeout", "client_timeout"):
            return ToolOutput(
                output=(
                    f"Error: Command timed out after {timeout}s. Partial output below. "
                    f"Retry with a larger `timeout` parameter (max {self.config.max_timeout}s), "
                    f"or use a narrower command.\n{output}"
                ),
                success=False,
                metadata={"returncode": returncode, "reason": reason, "timeout": timeout},
            )
        if reason == "transport_error":
            return ToolOutput(
                output=f"Error: Transport error while executing command:\n{output}",
                success=False,
                metadata={"returncode": returncode, "reason": reason},
            )

        rc = returncode if returncode is not None else -1
        return ToolOutput(
            output=output,
            success=rc == 0,
            metadata={"returncode": rc},
        )

    @staticmethod
    def _extract_command(params: Any) -> str:
        if not isinstance(params, dict):
            raise ToolException(f"Bash tool params must be a dictionary, got {type(params).__name__}")
        command = (params.get("command") or "").strip()
        if not command:
            raise ToolException("Bash tool requires a non-empty 'command' parameter")
        return command

    def _resolve_timeout(self, params: dict) -> int:
        """Effective timeout: model param if given, else config default; both capped.

        An unusable value (non-numeric, <= 0) falls back to the default rather
        than erroring: a bad timeout should not cost the model a whole turn.
        """
        raw = params.get("timeout")
        if raw is None:
            return min(self.config.timeout, self.config.max_timeout)
        try:
            timeout = int(raw)
        except (TypeError, ValueError):
            return min(self.config.timeout, self.config.max_timeout)
        if timeout <= 0:
            return min(self.config.timeout, self.config.max_timeout)
        return min(timeout, self.config.max_timeout)
