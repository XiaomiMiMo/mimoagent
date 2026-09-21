"""Path helpers for the agent-local MiMo session cwd."""

import posixpath
from typing import Any


def session_cwd(context: dict[str, Any] | None) -> str:
    context = context or {}
    state = context.get("state")
    if isinstance(state, dict) and state.get("cwd"):
        return str(state["cwd"])
    env_config = getattr(context.get("env"), "config", None)
    return str(getattr(env_config, "cwd", "") or "")


def resolve_path(path: str, context: dict[str, Any] | None) -> str:
    """Resolve a model path against the agent-local cwd, using POSIX paths."""
    if path.startswith("/"):
        return posixpath.normpath(path)
    base = session_cwd(context)
    return posixpath.normpath(posixpath.join(base or ".", path))
