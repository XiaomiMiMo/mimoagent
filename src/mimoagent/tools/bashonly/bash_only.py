"""Bash-only tool for agents that do not expose dedicated file tools.

This keeps the execution behavior of :class:`BashTool` while giving the model
an accurate description: bash is the complete tool surface, so commands such
as ``cat``, ``sed`` and shell redirections are explicitly allowed.
"""

from dataclasses import dataclass
from typing import Any

from mimoagent.tools.bash import BashTool, BashToolConfig


@dataclass
class BashOnlyToolConfig(BashToolConfig):
    """Configuration defaults for the bash-only profile."""

    timeout: int = 60
    max_timeout: int = 300


class BashOnlyTool(BashTool):
    """Execute repository operations through the ``bash`` tool surface."""

    def _create_config(self, config_dict: dict[str, Any]) -> BashOnlyToolConfig:
        return BashOnlyToolConfig(**config_dict)

    @property
    def name(self) -> str:
        # Keep the wire-level function name aligned with the default profile.
        # ``bash-only`` is only the registry selector for this isolated class.
        return "bash"

    @property
    def description(self) -> str:
        # Structure follows Claude Code's Bash description, inverted where the
        # facts differ: bash is the only tool (so cat/sed are the intended file
        # path), cwd does not persist (fresh subshell per call), and timeout is
        # seconds rather than milliseconds.
        default_s = self.config.timeout
        max_s = self.config.max_timeout
        return f"""Executes a bash command and returns its output.

This is the only available tool. Use standard shell commands and redirections to inspect, search, create, and edit files, run tests, and perform other tasks.

- Commands start in the configured current working directory.
- Each call runs in a new subshell, so directory and environment changes do not persist. Include any required `cd` or environment assignments in each command.
- Shell utilities such as `cat`, `sed`, `awk`, `grep`, and `find`, as well as shell redirections, may be used for file operations.
- Avoid interactive commands. Use non-interactive options where supported.
- Multiple calls in one message run concurrently and can race when they touch the same file (for example one call editing a file while another runs its tests, or two calls writing the same file); concurrent writes overwrite each other without any error.
- `timeout` is in seconds: default {default_s}, max {max_s}. Use a larger value for long-running builds or test suites."""
