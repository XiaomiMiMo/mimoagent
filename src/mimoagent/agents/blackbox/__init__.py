"""Blackbox (third-party) agents.

These agents do **not** drive the mimoagent ``step()`` loop. Instead they run a
self-contained third-party scaffold (Claude Code, Codex CLI, OpenCode, ...)
*inside the test pod*, then hand control back. They still satisfy the
:class:`mimoagent.Agent` protocol (``run(task) -> (status, message)``), so the
surrounding rollout machinery — environment setup, ``calculate_reward`` (which
grades ``git diff HEAD`` in the repo), and ``save_traj`` — works unchanged.

The trick that makes this clean: mimoagent grades the *workspace*, not the
conversation. A blackbox agent only has to (1) get the scaffold into the pod,
(2) point it at the task and the model gateway, (3) let it edit the repo. The
existing reward path does the rest.
"""

from mimoagent.agents.blackbox.claude_code import ClaudeCodeAgent, ClaudeCodeAgentConfig
from mimoagent.agents.blackbox.codex import CodexAgent, CodexAgentConfig
from mimoagent.agents.blackbox.dsh import DshAgent, DshAgentConfig
from mimoagent.agents.blackbox.grok import GrokAgent, GrokAgentConfig
from mimoagent.agents.blackbox.hermes import HermesAgent, HermesAgentConfig
from mimoagent.agents.blackbox.kilocode import KilocodeAgent, KilocodeAgentConfig
from mimoagent.agents.blackbox.kimi_cli import KimiCliAgent, KimiCliAgentConfig
from mimoagent.agents.blackbox.kimi_code import KimiCodeAgent, KimiCodeAgentConfig
from mimoagent.agents.blackbox.mimocode import MimoCodeAgent, MimoCodeAgentConfig
from mimoagent.agents.blackbox.mini_swe_agent import (
    MiniSweAgent,
    MiniSweAgentConfig,
)
from mimoagent.agents.blackbox.omp import OmpAgent, OmpAgentConfig
from mimoagent.agents.blackbox.openclaw import OpenclawAgent, OpenclawAgentConfig
from mimoagent.agents.blackbox.opencode import OpencodeAgent, OpencodeAgentConfig
from mimoagent.agents.blackbox.pi import PiAgent, PiAgentConfig

__all__ = [
    "DshAgent",
    "DshAgentConfig",
    "GrokAgent",
    "GrokAgentConfig",
    "HermesAgent",
    "HermesAgentConfig",
    "KilocodeAgent",
    "KilocodeAgentConfig",
    "KimiCliAgent",
    "KimiCliAgentConfig",
    "KimiCodeAgent",
    "KimiCodeAgentConfig",
    "MiniSweAgent",
    "MiniSweAgentConfig",
    "OmpAgent",
    "OmpAgentConfig",
    "OpenclawAgent",
    "OpenclawAgentConfig",
    "OpencodeAgent",
    "OpencodeAgentConfig",
    "ClaudeCodeAgent",
    "ClaudeCodeAgentConfig",
    "CodexAgent",
    "CodexAgentConfig",
    "MimoCodeAgent",
    "MimoCodeAgentConfig",
    "PiAgent",
    "PiAgentConfig",
]
