"""Helper for collecting per-agent tool-call-error logs.

Lives in ``utils/`` (not ``run/utils/``) so the environments layer can use it
without inverting the dependency direction.
"""

from __future__ import annotations

from collections.abc import Iterable

import xxhash


def _agent_context_hash(messages: list[dict]) -> str | None:
    """xxh64 of ``messages[0].content + messages[1].content`` for one agent.

    The first two messages are the agent's system prompt and the rendered
    instance/task — together they uniquely identify which logical context
    this agent is, even across reruns of the same instance.
    """
    if len(messages) < 2:
        return None
    a = messages[0].get("content") or ""
    b = messages[1].get("content") or ""
    return xxhash.xxh64((a + b).encode("utf-8")).hexdigest()


def collect_tool_call_errors(agents: Iterable) -> dict[str, list[bool]]:
    """Map each agent's context-hash to its per-turn ``tool_call_errors`` list.

    Skips agents whose first two messages aren't yet primed (e.g. crashed
    before ``run()`` could seed system+user). On hash collision (extremely
    unlikely; would mean two agents share both messages verbatim) the later
    entry wins.
    """
    out: dict[str, list[bool]] = {}
    for ag in agents:
        if ag is None:
            continue
        h = _agent_context_hash(getattr(ag, "messages", []))
        if h is None:
            continue
        out[h] = list(getattr(ag, "tool_call_errors", []))
    return out
