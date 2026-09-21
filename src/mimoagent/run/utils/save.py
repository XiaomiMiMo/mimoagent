import json
from collections.abc import Callable
from pathlib import Path

from mimoagent import Agent, __version__
from mimoagent.utils.log import AgentLogContext


def _count_tool_calls(messages: list[dict]) -> dict[str, int]:
    """Per-tool invocation counts for one agent's messages."""
    counts: dict[str, int] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            tool = fn.get("name") or "?"
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def _agent_traj(agent) -> dict:
    """Serialize one agent as ``{messages, tools, ...}`` — mirrors the
    top-level ``traj`` shape so parent and children have the same structure."""
    entry = {"messages": list(agent.messages)}
    entry |= agent.get_model_query_kwargs()  # tools, tool_choice, etc.
    return entry


def save_traj(
    agent: Agent | None,
    path: Path,
    *,
    print_path: bool = True,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    log_context: AgentLogContext | None = None,
    print_fct: Callable = print,
    **kwargs,
):
    """Save the trajectory of the agent to a file.

    Args:
        agent: The primary agent (typically the parent/``main`` agent).
        path: Destination for the JSON trajectory.
        print_path: Print confirmation of the output path.
        exit_status: Agent run's final status.
        result: Final submission/result string.
        extra_info: Additional fields to merge into the ``info`` dict.
        log_context: Optional :class:`AgentLogContext` — when present its
            registered trajectories (``{name: messages}``) are serialized as
            ``traj.trajs`` and ``info.tool_calls`` is filled with per-agent
            per-tool invocation counts.
        **kwargs: Additional top-level fields to merge into the output dict.
    """
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "total_tokens": 0,
            },
            "model_name": agent.model.config.model_name if agent is not None else None,
            "mimo_version": __version__,
        },
        "trajs": {},
        "trajectory_format": "mimoagent",
    } | kwargs

    if agent is not None:
        data["info"]["model_stats"]["api_calls"] = agent.model.n_calls
        if hasattr(agent.model, "token_stats"):
            tokens = agent.model.token_stats
            data["info"]["model_stats"]["input_tokens"] = tokens.input_tokens
            data["info"]["model_stats"]["output_tokens"] = tokens.output_tokens
            data["info"]["model_stats"]["cache_read_tokens"] = tokens.cache_read_tokens
            data["info"]["model_stats"]["cache_creation_tokens"] = tokens.cache_creation_tokens
            data["info"]["model_stats"]["total_tokens"] = tokens.total_tokens

    if log_context is not None and log_context.trajectories:
        # Per-agent trajectories: each entry = {messages, tools, tool_choice}.
        data["trajs"] = {name: _agent_traj(ag) for name, ag in log_context.trajectories.items()}
        data["info"]["tool_calls"] = {
            name: counts for name, ag in log_context.trajectories.items() if (counts := _count_tool_calls(ag.messages))
        }
    elif agent is not None:
        # No log context — fall back to a single-agent ``main`` entry so the
        # shape stays consistent.
        data["trajs"] = {"main": _agent_traj(agent)}
        counts = _count_tool_calls(agent.messages)
        if counts:
            data["info"]["tool_calls"] = {"main": counts}

    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    if print_path:
        print_fct(f"Saved trajectory to '{path}'")
