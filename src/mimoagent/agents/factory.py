"""Agent selection by name.

A tiny registry that maps the ``agent.type`` config field to an agent class, so
a run can pick its scaffold from YAML without touching the rollout code:

    agent:
      type: claude-code   # or "default" / "bashonly-agent" (native tool-calling agents)
      ...                 # "cc-agent" / "codex-agent" are native catalogues

``type`` defaults to ``"default"`` — configs that omit it behave exactly as
before. Blackbox agents (Claude Code, Codex CLI, OpenCode, ...) live under
``mimoagent.agents.blackbox`` and are resolved lazily so importing this module
never forces their (optional) dependencies.
``bashonly-agent`` is a native ``DefaultAgent`` variant whose default
tool surface contains only bash.

All agents share the same constructor contract as :class:`DefaultAgent`
(``Agent(model, env, *, msg_path=..., **agent_config)``) and the same
:class:`mimoagent.Agent` runtime protocol, so the caller is agnostic to which
one it built.
"""

from __future__ import annotations

from collections.abc import Callable

from mimoagent import Environment, Model
from mimoagent.agents.base import BaseAgent

# Map ``type`` → a zero-arg importer returning the agent class. Lazy so optional
# third-party scaffolds don't import at package load.
_AGENT_LOADERS: dict[str, Callable[[], type[BaseAgent]]] = {}


def _load_default() -> type[BaseAgent]:
    from mimoagent.agents.default import DefaultAgent

    return DefaultAgent


def _load_bashonly_agent() -> type[BaseAgent]:
    from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent

    return BashOnlyAgent


def _load_cc_agent() -> type[BaseAgent]:
    from mimoagent.agents.cc.cc_agent import CCAgent

    return CCAgent


def _load_mimocode_agent() -> type[BaseAgent]:
    from mimoagent.agents.mimocode import MimocodeAgent

    return MimocodeAgent


def _load_codex_agent() -> type[BaseAgent]:
    from mimoagent.agents.codex.codex_agent import CodexAgent

    return CodexAgent


def _load_claude_code() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.claude_code import ClaudeCodeAgent

    return ClaudeCodeAgent


def _load_codex() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.codex import CodexAgent

    return CodexAgent


def _load_mimocode() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.mimocode import MimoCodeAgent

    return MimoCodeAgent


def _load_pi() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.pi import PiAgent

    return PiAgent


def _load_grok() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.grok import GrokAgent

    return GrokAgent


def _load_kimi_code() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.kimi_code import KimiCodeAgent

    return KimiCodeAgent


def _load_kimi_cli() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.kimi_cli import KimiCliAgent

    return KimiCliAgent


def _load_kilocode() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.kilocode import KilocodeAgent

    return KilocodeAgent


def _load_mini_swe_agent() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.mini_swe_agent import MiniSweAgent

    return MiniSweAgent


def _load_openclaw() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.openclaw import OpenclawAgent

    return OpenclawAgent


def _load_opencode() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.opencode import OpencodeAgent

    return OpencodeAgent


def _load_omp() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.omp import OmpAgent

    return OmpAgent


def _load_hermes() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.hermes import HermesAgent

    return HermesAgent


def _load_dsh() -> type[BaseAgent]:
    from mimoagent.agents.blackbox.dsh import DshAgent

    return DshAgent


_AGENT_LOADERS["default"] = _load_default
_AGENT_LOADERS["bashonly-agent"] = _load_bashonly_agent
_AGENT_LOADERS["cc-agent"] = _load_cc_agent
_AGENT_LOADERS["mimocode-agent"] = _load_mimocode_agent
_AGENT_LOADERS["codex-agent"] = _load_codex_agent
_AGENT_LOADERS["claude-code"] = _load_claude_code
_AGENT_LOADERS["codex"] = _load_codex
_AGENT_LOADERS["mimocode"] = _load_mimocode
_AGENT_LOADERS["pi"] = _load_pi
_AGENT_LOADERS["grok"] = _load_grok
_AGENT_LOADERS["kimi-code"] = _load_kimi_code
_AGENT_LOADERS["kimi-cli"] = _load_kimi_cli
_AGENT_LOADERS["kilocode"] = _load_kilocode
_AGENT_LOADERS["mini-swe-agent"] = _load_mini_swe_agent
_AGENT_LOADERS["openclaw"] = _load_openclaw
_AGENT_LOADERS["opencode"] = _load_opencode
_AGENT_LOADERS["omp"] = _load_omp
_AGENT_LOADERS["hermes"] = _load_hermes
_AGENT_LOADERS["dsh"] = _load_dsh


def list_agent_types() -> list[str]:
    return sorted(_AGENT_LOADERS)


def get_agent_class(agent_type: str) -> type[BaseAgent]:
    """Resolve an agent ``type`` string to its class."""
    loader = _AGENT_LOADERS.get(agent_type)
    if loader is None:
        raise ValueError(f"Unknown agent type: {agent_type!r}. Available: {list_agent_types()}")
    return loader()


def make_agent(
    agent_type: str,
    model: Model,
    env: Environment,
    **agent_config,
) -> BaseAgent:
    """Build an agent of ``agent_type`` with the same call shape as DefaultAgent."""
    return get_agent_class(agent_type)(model, env, **agent_config)
