"""MiMo tool prompts copied from the captured non-Codex request resources."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_ROOT = Path(__file__).parent


@cache
def load_tool_prompt(name: str) -> str:
    """Load a verbatim MiMo tool description bundled beside the tools."""
    return (_ROOT / f"{name}.txt").read_text(encoding="utf-8")


@cache
def load_bash_prompt(default_timeout_s: int, max_timeout_s: int, max_output_lines: int, max_output_bytes: int) -> str:
    """Render the trimmed Bash prompt's dynamic fields.

    The timeout and output budgets come from this tool's config: the upstream
    resource states a fixed 120000ms default, and a model told it has two
    minutes will not pass an explicit timeout, then misreads the kill as a
    failing test. The output budget is rendered for the same reason — the
    prompt must describe the cut the tool actually makes.
    """
    # The sandbox's facts, not the host's: env.execute runs `/bin/bash -lc` in
    # a Linux pod, and a prompt that read the host's platform / $SHELL varied
    # across workers.
    return (
        load_tool_prompt("bash")
        .replace("${os}", "linux")
        .replace("${shell}", "bash")
        .replace("${default_timeout_ms}", str(default_timeout_s * 1000))
        .replace("${max_timeout_ms}", str(max_timeout_s * 1000))
        .replace("${max_output_lines}", str(max_output_lines))
        .replace("${max_output_bytes}", str(max_output_bytes))
    )
