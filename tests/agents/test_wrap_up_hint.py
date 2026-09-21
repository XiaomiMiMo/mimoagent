"""Opt-in wrap-up hint: a user turn when the last LLM call nears the context window."""

from __future__ import annotations

import json

from mimoagent.agents.cc.cc_agent import CCAgent
from mimoagent.environments.local import LocalEnvironment
from mimoagent.models import TokenStats


class ScriptedModel:
    def __init__(self, responses: list[dict], *, input_tokens: int = 100, output_tokens: int = 50):
        self._responses = responses
        self._index = -1
        self.n_calls = 0
        self.seen_messages: list[list[dict]] = []
        self.token_stats = TokenStats()
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.n_calls += 1
        self.token_stats.input_tokens += self._input_tokens
        self.token_stats.output_tokens += self._output_tokens
        self.seen_messages.append([dict(m) for m in messages])
        self._index += 1
        return self._responses[self._index]

    def get_template_vars(self) -> dict:
        return {}


def _bash(call_id: str, command: str = "echo hi") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "Bash", "arguments": json.dumps({"command": command})},
    }


def _agent(model, **overrides) -> CCAgent:
    cfg = dict(
        tools=[{"tool": "Bash"}],
        step_limit=20,
        show_context_usage=False,
    )
    cfg.update(overrides)
    return CCAgent(model=model, env=LocalEnvironment(), **cfg)


_WRAP_UP_MARKER = "[SYSTEM NOTICE: CONTEXT BUDGET]"


def _wrap_up_messages(agent: CCAgent) -> list[dict]:
    return [m for m in agent.messages if m.get("role") == "user" and _WRAP_UP_MARKER in str(m.get("content", ""))]


def test_wrap_up_hint_off_by_default():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ],
        input_tokens=10_000,
        output_tokens=10_000,
    )
    agent = _agent(model, wrap_up_hint_context_window=1, wrap_up_hint_fraction=0.0)
    status, _ = agent.run("task")
    assert status == "Idle"
    assert not _wrap_up_messages(agent)


def test_wrap_up_hint_fires_on_last_request_usage():
    # 180 + 20 = 200 tokens this call; window 200 at fraction 0.5 → fire, 0% left.
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ],
        input_tokens=180,
        output_tokens=20,
    )
    agent = _agent(
        model,
        wrap_up_hint=True,
        wrap_up_hint_context_window=200,
        wrap_up_hint_fraction=0.5,
        step_limit=20,
    )
    status, _ = agent.run("task")
    assert status == "Idle"
    assert agent._last_request_tokens == 200
    hints = _wrap_up_messages(agent)
    assert hints, "expected a wrap-up user turn after the tool result"
    content = hints[0]["content"]
    assert "Enter completion mode." in content
    assert "~0% of the context window remains" in content
    assert any(_WRAP_UP_MARKER in str(m.get("content", "")) for m in model.seen_messages[1])


def test_wrap_up_hint_stays_quiet_under_budget():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ],
        input_tokens=100,
        output_tokens=50,
    )
    agent = _agent(
        model,
        wrap_up_hint=True,
        wrap_up_hint_context_window=1_000_000,
        wrap_up_hint_fraction=0.9,
        step_limit=20,
    )
    status, _ = agent.run("task")
    assert status == "Idle"
    assert agent._last_request_tokens == 150
    assert not _wrap_up_messages(agent)


def test_wrap_up_hint_uses_last_call_not_cumulative_stats():
    # Two calls of 80+20 each. Cumulative token_stats = 200, last call = 100.
    # Window 150 at 0.9 → last call 100 < 135, so no hint. Cumulative 200 would fire.
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ],
        input_tokens=80,
        output_tokens=20,
    )
    agent = _agent(
        model,
        wrap_up_hint=True,
        wrap_up_hint_context_window=150,
        wrap_up_hint_fraction=0.9,
        step_limit=20,
    )
    status, _ = agent.run("task")
    assert status == "Idle"
    assert model.token_stats.input_tokens + model.token_stats.output_tokens == 200
    assert agent._last_request_tokens == 100
    assert not _wrap_up_messages(agent)


def test_wrap_up_hint_ignores_step_limit():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ]
    )
    agent = _agent(
        model,
        wrap_up_hint=True,
        wrap_up_hint_context_window=1_000_000,
        wrap_up_hint_fraction=0.5,
        step_limit=2,
    )
    status, _ = agent.run("task")
    assert status == "Idle"
    assert not _wrap_up_messages(agent)


def test_wrap_up_hint_skips_when_usage_missing():
    class NoStatsModel(ScriptedModel):
        def __init__(self, responses):
            super().__init__(responses)
            del self.token_stats

        def query(self, messages, **kwargs):
            self.n_calls += 1
            self.seen_messages.append([dict(m) for m in messages])
            self._index += 1
            return self._responses[self._index]

    model = NoStatsModel(
        [
            {"content": "", "tool_calls": [_bash("c1")]},
            {"content": "done"},
        ]
    )
    agent = _agent(
        model,
        wrap_up_hint=True,
        wrap_up_hint_context_window=8,
        wrap_up_hint_fraction=0.5,
    )
    status, _ = agent.run("task")
    assert status == "Idle"
    assert agent._last_request_tokens == 0
    assert not _wrap_up_messages(agent)
