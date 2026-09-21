"""The ``DefaultAgent`` action-interceptor seam.

``add_action_interceptor`` registers pre-dispatch policies; ``execute_action``
consults them in order and the first non-``None`` result stands in for the tool
call. The anti-hack guard is one such policy (see ``test_antihack.py``); these
tests pin the seam itself, independent of any particular policy.
"""

from __future__ import annotations

from types import SimpleNamespace

from mimoagent.agents.default import DefaultAgent
from mimoagent.tools.base import ToolOutput
from mimoagent.tools.registry import ToolRegistry


class _Model:
    config = SimpleNamespace(model_name="test")


class _Env:
    config = SimpleNamespace(cwd="/testbed")


class _RecordingTool:
    name = "bash"

    def __init__(self):
        self.calls: list[dict] = []

    def execute(self, params, context):
        self.calls.append(params)
        return ToolOutput(output="ran", success=True, metadata={"returncode": 0})


def _agent(tool):
    agent = DefaultAgent(_Model(), _Env(), tools=[{"tool": "bash"}])
    agent.tool_registry = ToolRegistry()
    agent.tool_registry.tools[tool.name] = tool
    return agent


def _result(text):
    return {"output": text, "success": False, "metadata": {}}


def test_no_interceptors_by_default_and_tool_runs():
    tool = _RecordingTool()
    agent = _agent(tool)

    assert agent._action_interceptors == []
    assert agent.execute_action({"tool": "bash", "params": {"command": "ls"}})["output"] == "ran"
    assert tool.calls == [{"command": "ls"}]


def test_interceptor_returning_none_falls_through_to_the_tool():
    tool = _RecordingTool()
    agent = _agent(tool)
    seen = []
    agent.add_action_interceptor(lambda action: seen.append(action) and None)

    result = agent.execute_action({"tool": "bash", "params": {"command": "ls"}})

    assert result["output"] == "ran"
    assert seen == [{"tool": "bash", "params": {"command": "ls"}}]
    assert tool.calls == [{"command": "ls"}]


def test_first_interceptor_result_wins_and_skips_the_tool():
    tool = _RecordingTool()
    agent = _agent(tool)
    order = []

    def passes(action):
        order.append("passes")
        return

    def blocks(action):
        order.append("blocks")
        return _result("blocked")

    def never(action):  # pragma: no cover - must not run
        order.append("never")
        return _result("unreachable")

    agent.add_action_interceptor(passes)
    agent.add_action_interceptor(blocks)
    agent.add_action_interceptor(never)

    result = agent.execute_action({"tool": "bash", "params": {"command": "ls"}})

    assert result == _result("blocked")
    assert order == ["passes", "blocks"]
    assert tool.calls == []


def test_interceptor_result_bypasses_execute_tool_decorators():
    """A substituted result must not pass through ``_execute_tool``: variants
    that decorate real results (Mimocode's cwd metadata) must never touch it."""

    class Decorating(DefaultAgent):
        def _execute_tool(self, action):
            result = super()._execute_tool(action)
            result["metadata"]["decorated"] = True
            return result

    tool = _RecordingTool()
    agent = Decorating(_Model(), _Env(), tools=[{"tool": "bash"}])
    agent.tool_registry = ToolRegistry()
    agent.tool_registry.tools[tool.name] = tool
    agent.add_action_interceptor(lambda action: _result("blocked") if action["params"].get("command") == "rm" else None)

    blocked = agent.execute_action({"tool": "bash", "params": {"command": "rm"}})
    real = agent.execute_action({"tool": "bash", "params": {"command": "ls"}})

    assert blocked["metadata"] == {}
    assert real["metadata"]["decorated"] is True
    assert tool.calls == [{"command": "ls"}]


def test_interceptors_run_inside_parallel_tool_calls():
    """``_run_tool_calls`` routes every parsed call through ``execute_action``,
    so interceptors also cover the parallel worker path."""
    tool = _RecordingTool()
    agent = _agent(tool)
    agent.add_action_interceptor(lambda action: _result("blocked") if "rm" in action["params"]["command"] else None)

    def call(call_id, command):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": f'{{"command": "{command}"}}'},
        }

    outcomes = agent._run_tool_calls([call("c1", "ls"), call("c2", "rm -rf x"), call("c3", "pwd")])

    assert [o["output"] for o in outcomes] == ["ran", "blocked", "ran"]
    assert sorted(c["command"] for c in tool.calls) == ["ls", "pwd"]
