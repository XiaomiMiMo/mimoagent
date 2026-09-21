"""Tests for model-decided conversation compaction (the Compact tool).

The model frees context by calling Compact; the actual summarize-and-rebuild
runs at the top of the next ``query()`` (see ``CCAgent._do_compact``). These
tests use a scripted model so the rollout is deterministic.
"""

from __future__ import annotations

import json

from mimoagent.agents.cc.cc_agent import CCAgent, CCAgentConfig
from mimoagent.compaction import (
    CODEX_SUMMARY_PREFIX,
    COMPACT_BOUNDARY_PREFIX,
    COMPACT_ORIGINAL_USER_ISSUE_PREFIX,
    COMPACT_SUMMARY_PREFIX,
    build_local_compacted_history,
)
from mimoagent.environments.local import LocalEnvironment


class ScriptedModel:
    """Returns a queued sequence of assistant responses.

    Each item is a dict already in the ``model.query`` return shape, e.g.
    ``{"content": ...}`` or ``{"content": "", "tool_calls": [...]}``. Recording
    every ``messages`` snapshot passed in lets tests inspect what the summary
    turn was given.
    """

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._index = -1
        self.n_calls = 0
        self.seen_messages: list[list[dict]] = []
        self.seen_kwargs: list[dict] = []

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.n_calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_kwargs.append(kwargs)
        self._index += 1
        return self._responses[self._index]

    def get_template_vars(self) -> dict:
        return {}


def _bash_call(call_id: str, command: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "Bash", "arguments": json.dumps({"command": command})},
    }


def _compact_call(call_id: str, instructions: str | None = None) -> dict:
    args = {} if instructions is None else {"instructions": instructions}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "Compact", "arguments": json.dumps(args)},
    }


def _make_agent(model, **config_overrides) -> CCAgent:
    cfg = dict(
        tools=[{"tool": "Bash"}, {"tool": "Compact"}],
        step_limit=20,
        show_context_usage=False,
    )
    cfg.update(config_overrides)
    return CCAgent(model=model, env=LocalEnvironment(), config_class=CCAgentConfig, **cfg)


def _assert_tool_pairing(messages: list[dict]) -> None:
    """Every tool message must follow an assistant message whose tool_calls
    contain its tool_call_id."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        assert i > 0, "tool message cannot be first"
        # The preceding message in our flat history is either the assistant
        # that issued the calls, or an earlier tool response from the same
        # assistant turn. Walk back to the nearest assistant.
        j = i - 1
        while j >= 0 and messages[j].get("role") != "assistant":
            j -= 1
        assert j >= 0, "tool message with no preceding assistant"
        call_ids = {tc.get("id") for tc in (messages[j].get("tool_calls") or [])}
        assert msg.get("tool_call_id") in call_ids


SUMMARY_RESPONSE = {
    "content": "<analysis>thinking</analysis>\n<summary>Did work on the bug. Next: run tests.</summary>"
}
SUMMARY_RESPONSE_2 = {
    "content": "<analysis>thinking again</analysis>\n<summary>Kept working on the bug. Next: finish.</summary>"
}


def test_compaction_rebuilds_history():
    # Two real tool turns, then a Compact call, then the summary turn, then idle.
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash_call("c1", "echo hello")]},
            {"content": "", "tool_calls": [_bash_call("c2", "echo world")]},
            {"content": "", "tool_calls": [_compact_call("c3")]},
            SUMMARY_RESPONSE,  # the separate summary turn inside _do_compact
            {"content": "All done."},  # post-compaction idle turn
        ]
    )
    agent = _make_agent(model)
    status, _ = agent.run("do the task")

    assert status == "Idle"
    assert agent._compact_count == 1

    roles = [m["role"] for m in agent.messages]
    # Expect: system, boundary(user), original issue anchor(user), summary(user), assistant("All done.")
    assert roles == ["system", "user", "user", "user", "assistant"]
    assert agent.messages[1]["content"].startswith(COMPACT_BOUNDARY_PREFIX)
    assert agent.messages[2]["content"].startswith(COMPACT_ORIGINAL_USER_ISSUE_PREFIX)
    assert "do the task" in agent.messages[2]["content"]
    assert agent.messages[3]["content"].startswith(COMPACT_SUMMARY_PREFIX)
    assert "Did work on the bug" in agent.messages[3]["content"]
    # No orphaned tool messages remain.
    assert not any(m["role"] == "tool" for m in agent.messages)
    _assert_tool_pairing(agent.messages)


def test_summary_turn_sees_compact_round_and_no_tools():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash_call("c1", "echo hi")]},
            {"content": "", "tool_calls": [_compact_call("c3")]},
            SUMMARY_RESPONSE,
            {"content": "done"},
        ]
    )
    agent = _make_agent(model)
    agent.run("task")

    # The 3rd model call is the summary turn (calls: step1, step2(compact), summary, post).
    summary_messages = model.seen_messages[2]
    summary_kwargs = model.seen_kwargs[2]
    # Summary turn must not be given tools/tool_choice.
    assert "tools" not in summary_kwargs and "tool_choice" not in summary_kwargs
    # It includes the just-completed Compact round, so its tool_call/response pair.
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in summary_messages)
    assert any(m.get("role") == "tool" for m in summary_messages)
    _assert_tool_pairing(summary_messages)


def test_compaction_preserves_original_issue_anchor_across_repeated_compactions():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_compact_call("c1")]},
            SUMMARY_RESPONSE,
            {"content": "", "tool_calls": [_compact_call("c2")]},
            SUMMARY_RESPONSE_2,
            {"content": "done"},
        ]
    )
    agent = _make_agent(model)
    agent.run("original issue: preserve this exact task")

    anchors = [
        message
        for message in agent.messages
        if str(message.get("content", "")).startswith(COMPACT_ORIGINAL_USER_ISSUE_PREFIX)
    ]
    assert len(anchors) == 1
    assert "original issue: preserve this exact task" in anchors[0]["content"]
    assert agent._compact_count == 2

    # The normal model call after the first compaction and the second summary
    # turn both retain the anchor, so repeated compaction cannot lose the task.
    post_first_compact_messages = model.seen_messages[2]
    second_summary_messages = model.seen_messages[3]
    assert any(
        str(message.get("content", "")).startswith(COMPACT_ORIGINAL_USER_ISSUE_PREFIX)
        for message in post_first_compact_messages
    )
    assert any(
        str(message.get("content", "")).startswith(COMPACT_ORIGINAL_USER_ISSUE_PREFIX)
        for message in second_summary_messages
    )


def test_empty_summary_keeps_history():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash_call("c1", "echo hi")]},
            {"content": "", "tool_calls": [_compact_call("c3")]},
            {"content": ""},  # summary turn returns nothing usable
            {"content": "done"},
        ]
    )
    agent = _make_agent(model)

    status, _ = agent.run("task")
    assert status == "Idle"
    # Compaction was best-effort and produced nothing usable -> not counted,
    # history NOT collapsed to [system, boundary, summary].
    assert agent._compact_count == 0
    roles = [m["role"] for m in agent.messages]
    assert "tool" in roles  # the original Bash tool result survives
    assert not any(str(m.get("content", "")).startswith(COMPACT_BOUNDARY_PREFIX) for m in agent.messages)
    _assert_tool_pairing(agent.messages)


def test_compact_instructions_forwarded_to_summary_prompt():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_compact_call("c1", instructions="KEEP the failing test output")]},
            SUMMARY_RESPONSE,
            {"content": "done"},
        ]
    )
    agent = _make_agent(model)
    agent.run("task")

    # Summary turn is the 2nd model call; its last message is the summary request.
    summary_request = model.seen_messages[1][-1]["content"]
    assert "KEEP the failing test output" in summary_request


def test_context_usage_footer_threshold():
    # Low budget so a single tool result crosses the warn fraction.
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash_call("c1", "echo " + "x" * 4000)]},
            {"content": "done"},
        ]
    )
    agent = _make_agent(
        model,
        show_context_usage=True,
        compaction_threshold_tokens=200,
        context_usage_warn_fraction=0.5,
    )
    agent.run("task")

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert tool_msgs, "expected a tool result"
    assert "<context_usage>" in tool_msgs[0]["content"]
    assert "Call Compact" in tool_msgs[0]["content"]


def test_context_usage_footer_absent_below_threshold():
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_bash_call("c1", "echo hi")]},
            {"content": "done"},
        ]
    )
    agent = _make_agent(
        model,
        show_context_usage=True,
        compaction_threshold_tokens=1_000_000,
        context_usage_warn_fraction=0.7,
    )
    agent.run("task")

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert tool_msgs
    assert "<context_usage>" not in tool_msgs[0]["content"]


def test_codex_local_compaction_keeps_recent_user_messages_only():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "tool work", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "large output " * 100},
        {"role": "user", "content": "new task"},
    ]
    compacted = build_local_compacted_history(messages, "handoff", max_user_tokens=3)
    assert [m["role"] for m in compacted] == ["system", "user", "user", "user"]
    assert compacted[-1]["content"].startswith(CODEX_SUMMARY_PREFIX)
    assert "new task" in compacted[-2]["content"]
    assert not any(m["role"] in {"assistant", "tool"} for m in compacted)


def test_codex_compaction_does_not_retain_prior_codex_summary_as_user_history():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {"role": "user", "content": f"{CODEX_SUMMARY_PREFIX}\nold handoff"},
        {"role": "user", "content": "latest request"},
    ]
    compacted = build_local_compacted_history(messages, "new handoff")
    contents = [m["content"] for m in compacted]
    assert "old handoff" not in "\n".join(contents)
    assert "original task" in contents
    assert "latest request" in contents


def test_codex_summary_prefix_matches_codex_rs_template():
    from mimoagent.compaction import CODEX_SUMMARY_PREFIX

    # Verbatim from openai/codex codex-rs/prompts/templates/compact/summary_prefix.md
    # (no trailing newline). Pinned here rather than read from an out-of-repo
    # checkout so the test is reproducible on any machine; update both together
    # if upstream changes the wording.
    expected = (
        "Another language model started to solve this problem and produced a summary of its "
        "thinking process. You also have access to the state of the tools that were used by "
        "that language model. Use this to build on the work that has already been done and "
        "avoid duplicating work. Here is the summary produced by the other language model, "
        "use the information in this summary to assist with your own analysis:"
    )
    assert CODEX_SUMMARY_PREFIX == expected
