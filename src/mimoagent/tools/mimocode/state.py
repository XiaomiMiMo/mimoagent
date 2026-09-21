"""Shape of the agent-local tool state carried in the Mimocode tool context.

The state is threaded through ``get_tool_context()`` and consumed by the tools:
``paths`` resolves relative paths against ``cwd``, ``task`` keeps its ledger in
``tasks``, and ``read``/``write``/``edit`` use ``read_files`` as the read gate.
Building it here rather than inline keeps what an ``actor.run`` child inherits an
explicit decision instead of a side effect of copying the parent wholesale.
"""

from __future__ import annotations

from typing import Any


def new_tool_state(cwd: str = "") -> dict[str, Any]:
    """State for a primary agent starting a rollout."""
    return {"cwd": cwd, "tasks": {}, "read_files": set()}


def derive_child_state(parent: dict[str, Any]) -> dict[str, Any]:
    """State for an ``actor.run`` child.

    Shares the task ledger by reference, so the plan stays one tree across the
    rollout. Inherits ``cwd`` so paths resolve the same way on both sides. Starts
    a fresh read gate: the child never sees the parent's conversation, so a file
    the parent read carries no information for it — handing over that set would
    let the child overwrite a file it has not looked at.
    """
    return {
        "cwd": parent.get("cwd", ""),
        "tasks": parent.setdefault("tasks", {}),
        "read_files": set(),
    }


__all__ = ["derive_child_state", "new_tool_state"]
