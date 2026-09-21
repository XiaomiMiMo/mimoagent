"""The bashonly-agent harness is a native DefaultAgent variant."""

from types import SimpleNamespace

from mimoagent.agents.bashonly.bashonly_agent import BashOnlyAgent, BashOnlyAgentConfig
from mimoagent.agents.default import DefaultAgent
from mimoagent.agents.factory import get_agent_class, make_agent


class _Model:
    config = SimpleNamespace(model_name="test")


class _Env:
    config = SimpleNamespace(cwd="/testbed")


def test_factory_registers_a_native_bashonly_agent_harness():
    assert get_agent_class("bashonly-agent") is BashOnlyAgent


def test_harness_inherits_default_agent_and_uses_bash_only_defaults():
    agent = make_agent("bashonly-agent", _Model(), _Env())

    assert isinstance(agent, DefaultAgent)
    assert isinstance(agent.config, BashOnlyAgentConfig)
    assert agent.config.tools == [{"tool": "bash-only", "config": {"timeout": 60, "max_timeout": 300}}]
    assert agent.config.step_limit == 500
    assert agent.config.tool_parallel_workers == 8
    assert [tool["function"]["name"] for tool in agent.get_model_query_kwargs()["tools"]] == ["bash"]
    assert "the only available tool" in agent.get_model_query_kwargs()["tools"][0]["function"]["description"]


def test_explicit_tool_config_can_override_the_harness_default():
    agent = BashOnlyAgent(_Model(), _Env(), tools=[{"tool": "bash-only", "config": {"timeout": 12}}])

    tool = agent.tool_registry.get("bash")
    assert tool.config.timeout == 12
    assert tool.config.max_timeout == 300


def test_antihack_block_is_accepted_and_default_off():
    """An ``antihack:`` yaml block must reach the registered guard instead of being
    dropped as an unknown config key, and must stay off when the block is absent."""
    agent = BashOnlyAgent(_Model(), _Env(), antihack={"enabled": True, "dummy_output": "nope", "dummy_success": True})

    assert agent.antihack.enabled is True
    assert agent.antihack.dummy_output().output == "nope"
    assert agent.antihack.dummy_output().success is True
    # Wire-level tool name is lowercase ``bash``; the guard must cover that spelling.
    verdict = agent.antihack.inspect(
        {"tool": "bash", "params": {"command": "curl https://raw.githubusercontent.com/a/b/c"}}
    )
    assert verdict is not None and verdict.tool == "bash"

    assert BashOnlyAgent(_Model(), _Env()).antihack.enabled is False
