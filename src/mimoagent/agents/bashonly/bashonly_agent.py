"""Native bash-only mimoagent harness.

``BashOnlyAgent`` keeps :class:`DefaultAgent`'s tool-calling loop and
runtime behavior, but its default tool surface contains only the dedicated
``bash-only`` implementation. The implementation advertises the wire-level
function name ``bash`` so model/tool-call distributions stay compatible with
the default profile. The anti-reward-hacking guard is registered as an action
interceptor and stays off unless the yaml enables it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mimoagent import Environment, Model
from mimoagent.agents.antihack import AntiHackConfig, AntiHackGuard
from mimoagent.agents.default import DefaultAgent, DefaultAgentConfig


@dataclass
class BashOnlyAgentConfig(DefaultAgentConfig):
    """Prompts and limits of the default agent profile, for the bash-only harness."""

    system_template: str = """You are an agent, your current working directory is {{cwd}}.

You can use the tools available to you to interact with the computer to assist the user in completing tasks."""
    instance_template: str = """Fix the following issue:

{{task}}"""
    tools: list[dict[str, Any]] = field(
        default_factory=lambda: [{"tool": "bash-only", "config": {"timeout": 60, "max_timeout": 300}}]
    )
    step_limit: int = 500

    # Runtime anti-reward-hacking guard (see agents/antihack.py). Default-off: an
    # absent ``antihack:`` yaml block (or ``enabled: false``) is a strict no-op.
    antihack: Any = field(default_factory=AntiHackConfig)


class BashOnlyAgent(DefaultAgent):
    """DefaultAgent with a single bash tool and bash-only guidance."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = BashOnlyAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        # Anti-hack guard as an action interceptor: a flagged call returns the
        # dummy from execute_action and never reaches _execute_tool.
        self.antihack = AntiHackGuard(self.config.antihack)
        self.add_action_interceptor(self.antihack)


__all__ = ["BashOnlyAgent", "BashOnlyAgentConfig"]
