"""Native MiMo-Code (non-Codex) agent.

Selected with ``agent.type: mimocode-agent``. The loop itself lives in
:mod:`~mimoagent.agents.mimocode.mimocode_agent`; its lowercase tool catalogue is
:mod:`mimoagent.tools.mimocode`. ``prompts/`` holds the static system prompts for
the primary agent and for the two ``actor.run`` child presets.
"""

from mimoagent.agents.mimocode.mimocode_agent import MimocodeAgent, MimocodeAgentConfig
from mimoagent.agents.mimocode.mimocode_subagents import (
    MIMOCODE_SUBAGENT_PRESETS,
    available_mimocode_subagent_types,
    mimocode_subagent_kwargs,
)

__all__ = [
    "MIMOCODE_SUBAGENT_PRESETS",
    "MimocodeAgent",
    "MimocodeAgentConfig",
    "available_mimocode_subagent_types",
    "mimocode_subagent_kwargs",
]
