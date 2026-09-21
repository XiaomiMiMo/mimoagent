"""Subagent presets exposed by the native Mimocode actor tool.

The repository also has a shared CC/default-agent roster in ``subagents.py``.
Mimocode deliberately exposes only the two upstream actor profiles needed by
the RL surface: a read-only explorer and a full-capability general worker.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8").rstrip()


# ``description`` is the capability blurb the actor tool advertises; the tool list
# is rendered from ``tools`` there rather than repeated here, so the two cannot
# drift. Both presets carry ``task`` because the ledger is shared by reference
# with the parent (see tools/mimocode/state.py) and a child without the tool could
# not read or update the plan it is part of.
MIMOCODE_SUBAGENT_PRESETS: dict[str, dict[str, Any]] = {
    "explore": {
        "description": "investigate a focused question read-only and return a direct answer.",
        "system_template": _load_prompt("mimocode_explore.txt"),
        "instance_template": "{{task}}",
        "tools": [
            {"tool": "bash"},
            {"tool": "read"},
            {"tool": "grep"},
            {"tool": "glob"},
            {"tool": "task"},
        ],
        "step_limit": 500,
    },
    "general": {
        "description": "complete a bounded implementation or investigation and return a report.",
        "system_template": _load_prompt("mimocode_general.txt"),
        "instance_template": "{{task}}",
        "tools": [
            {"tool": "bash"},
            {"tool": "read"},
            {"tool": "write"},
            {"tool": "edit"},
            {"tool": "task"},
        ],
        "step_limit": 500,
    },
}


def available_mimocode_subagent_types() -> list[str]:
    return list(MIMOCODE_SUBAGENT_PRESETS)


def mimocode_subagent_kwargs(subagent_type: str) -> dict[str, Any]:
    """Return a fresh config dict for a Mimocode child agent."""
    if subagent_type not in MIMOCODE_SUBAGENT_PRESETS:
        raise KeyError(
            f"Unknown Mimocode subagent type: {subagent_type!r}. Available: {available_mimocode_subagent_types()}"
        )
    kwargs = deepcopy(MIMOCODE_SUBAGENT_PRESETS[subagent_type])
    kwargs.pop("description", None)
    return kwargs


__all__ = [
    "MIMOCODE_SUBAGENT_PRESETS",
    "available_mimocode_subagent_types",
    "mimocode_subagent_kwargs",
]
