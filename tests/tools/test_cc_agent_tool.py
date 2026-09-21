"""The CC ``Agent`` tool hands the parent's anti-hack guard config to the subagent.

A subagent built from preset kwargs alone would carry a default (disabled)
guard, making ``Agent`` a one-step bypass of everything the parent is blocked
from reading. The mimocode ``actor`` tool already propagates the config; this
pins the same contract for the CC catalogue.
"""

from __future__ import annotations

from mimoagent.agents.cc.cc_agent import CCAgent
from mimoagent.environments.local import LocalEnvironment


class ScriptedModel:
    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._index = -1

    def query(self, messages: list[dict], **kwargs) -> dict:
        self._index += 1
        return self._responses[self._index]

    def get_template_vars(self) -> dict:
        return {}


def _spawn(parent: CCAgent):
    tool = parent.tool_registry.get("Agent")
    child_model = ScriptedModel([{"content": "explored"}])
    result = tool.execute(
        {"prompt": "look around", "subagent_type": "explore"},
        {"env": parent.env, "model": child_model, "agent": parent},
    )
    assert result.success
    assert len(parent.subagents) == 1
    return parent.subagents[0]


def test_subagent_inherits_the_parent_antihack_config(tmp_path):
    parent = CCAgent(
        model=ScriptedModel([]),
        env=LocalEnvironment(cwd=str(tmp_path)),
        tools=[{"tool": "Agent"}],
        antihack={"enabled": True, "dummy_output": "nope", "bash_patterns": [r"\bcurl\b"]},
    )

    child = _spawn(parent)

    assert child.antihack.enabled is True
    assert child.antihack.config.dummy_output == "nope"
    assert child.antihack.config.bash_patterns == [r"\bcurl\b"]
    # The child guard is live on its own interceptor chain, not just configured.
    blocked = child.execute_action({"tool": "bash", "params": {"command": "curl https://x/y"}})
    assert blocked["output"] == "nope"
    assert child.antihack.blocks and child.antihack.blocks[0]["tool"] == "bash"
    assert parent.antihack.blocks == []  # telemetry stays per agent


def test_subagent_guard_stays_off_when_the_parent_never_opted_in(tmp_path):
    parent = CCAgent(model=ScriptedModel([]), env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "Agent"}])

    child = _spawn(parent)

    assert child.antihack.enabled is False
