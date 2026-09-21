"""Tests for the user-agent sidecar (UserAgentDriver).

Both the main agent and the user-agent are DefaultAgents driven by scripted
models — a response without tool calls ends a turn (Idle), so each ``run()``
consumes exactly one scripted response. A callable response simulates side
effects the agent performs during its turn (e.g. writing the query file).
"""

import json
import shlex
from types import SimpleNamespace

import pytest

from mimoagent.agents.default import DefaultAgent
from mimoagent.agents.user_agent import (
    UserAgentDriver,
    UserAgentDriverConfig,
    env_trajectory_files,
)

WORKSPACE = "/tmp/.mimo-user-agent"


class ScriptedModel:
    """Returns canned assistant messages in order.

    A response may be: a string (plain content), an Exception (raised), or a
    callable (invoked for side effects; its return value is the content).
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.n_calls = 0
        self.config = SimpleNamespace(model_name="scripted")
        self.token_stats = SimpleNamespace(
            input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0
        )

    def query(self, messages, **kwargs):
        resp = self._responses[self.n_calls]
        self.n_calls += 1
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            resp = resp()
        return {"content": resp}

    def get_template_vars(self):
        return {}


class DummyEnv:
    """Fake environment with a dict-backed filesystem for the workspace files."""

    def __init__(self):
        self.config = SimpleNamespace(cwd="/testbed")
        self.commands = []
        self.copies = []  # (remote_path, content) — content read at copy time
        self.files = {}

    def execute(self, command, cwd="", timeout=None):
        self.commands.append(command)
        parts = shlex.split(command)
        if parts[0] == "cat":
            content = self.files.get(parts[1])
            if content is None:
                return {"output": f"cat: {parts[1]}: No such file", "returncode": 1}
            return {"output": content, "returncode": 0}
        return {"output": "", "returncode": 0}

    def copy_to(self, src, dst, **kwargs):
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.files[dst] = content
        self.copies.append((dst, content))

    def get_template_vars(self):
        return {"cwd": "/testbed"}


def make_main_agent(responses, msg_path=None):
    return DefaultAgent(ScriptedModel(responses), DummyEnv(), tools=[], msg_path=msg_path)


def make_driver(main_agent, ua_responses, **config):
    config.setdefault("tools", [])
    return UserAgentDriver(main_agent, config, model=ScriptedModel(ua_responses))


def deliver(env, round_idx, text, chat_text="wrote the follow-up to the query file"):
    """A scripted UA response that writes ``text`` into query_<round_idx>.md."""

    def _respond():
        env.files[f"{WORKSPACE}/query_{round_idx}.md"] = text
        return chat_text

    return _respond


def user_messages(agent):
    return [m["content"] for m in agent.messages if m["role"] == "user"]


def test_drives_configured_rounds():
    main = make_main_agent(["main turn 1", "main turn 2", "main turn 3"])
    env = main.env
    driver = make_driver(main, [deliver(env, 1, "follow-up A"), deliver(env, 2, "follow-up B")], rounds=2)

    exit_status, result = driver.run("fix the bug")

    assert (exit_status, result) == ("Idle", "main turn 3")
    # Main agent saw: instance prompt, then the two delivered follow-ups verbatim.
    assert user_messages(main) == ["Your task: fix the bug", "follow-up A", "follow-up B"]
    assert main.model.n_calls == 3
    assert driver.agent.model.n_calls == 2


def test_query_history_accumulates_in_workspace():
    main = make_main_agent(["main turn 1", "main turn 2", "main turn 3"])
    env = main.env
    driver = make_driver(main, [deliver(env, 1, "follow-up A"), deliver(env, 2, "follow-up B")], rounds=2)
    driver.run("fix the bug")

    # query_0.md is the original task, seeded by the driver; later queries pile up.
    assert env.files[f"{WORKSPACE}/query_0.md"] == "fix the bug"
    assert env.files[f"{WORKSPACE}/query_1.md"] == "follow-up A"
    assert env.files[f"{WORKSPACE}/query_2.md"] == "follow-up B"


def test_query_delivered_via_file_not_chat_text():
    main = make_main_agent(["main turn 1", "main turn 2"])
    env = main.env
    # Chat text differs from the file content — only the file must be used.
    driver = make_driver(main, [deliver(env, 1, "the real query", chat_text="IGNORE ME chatty summary")], rounds=1)
    driver.run("fix the bug")

    assert user_messages(main)[1] == "the real query"


def test_no_query_file_stops_driving():
    main = make_main_agent(["main turn 1"])
    driver = make_driver(main, ["forgot to write the file"], rounds=3)

    exit_status, result = driver.run("fix the bug")

    assert (exit_status, result) == ("Idle", "main turn 1")
    assert main.model.n_calls == 1


def test_publishes_msg_file_verbatim_and_refreshed(tmp_path):
    msg_path = tmp_path / "agent_msgs" / "main.log"
    main = make_main_agent(["main turn 1", "main turn 2", "main turn 3"], msg_path=msg_path)
    env = main.env
    driver = make_driver(main, [deliver(env, 1, "follow-up A"), deliver(env, 2, "follow-up B")], rounds=2)
    driver.run("fix the bug")

    # The msg file is mirrored (same name) into the workspace after every turn.
    mirrors = [content for path, content in env.copies if path == f"{WORKSPACE}/main.log"]
    assert len(mirrors) == 2
    first, second = mirrors
    # Verbatim copy of the agent's own dump — and cumulative across turns.
    assert first == msg_path.read_text()[: len(first)]
    assert "main turn 1" in first
    assert "main turn 2" not in first
    assert "main turn 2" in second
    assert any(cmd.startswith("mkdir -p") for cmd in env.commands)


def test_prompts_point_to_workspace_not_inline(tmp_path):
    msg_path = tmp_path / "agent_msgs" / "main.log"
    main = make_main_agent(["main turn 1", "main turn 2", "main turn 3"], msg_path=msg_path)
    env = main.env
    driver = make_driver(main, [deliver(env, 1, "follow-up A"), deliver(env, 2, "follow-up B")], rounds=2)
    driver.run("fix the bug")

    first_prompt, followup_prompt = user_messages(driver.agent)
    assert "fix the bug" in first_prompt
    assert f"{WORKSPACE}/main.log" in first_prompt
    assert f"{WORKSPACE}/query_1.md" in first_prompt
    assert "main turn 1" not in first_prompt
    assert f"{WORKSPACE}/main.log" in followup_prompt
    assert f"{WORKSPACE}/query_2.md" in followup_prompt
    assert "main turn 2" not in followup_prompt
    # The user-agent keeps one continuous conversation across rounds.
    assert driver.agent.messages[0]["role"] == "system"
    assert WORKSPACE in driver.agent.messages[0]["content"]


def test_falls_back_to_raw_messages_json_without_msg_file():
    main = make_main_agent(["main turn 1", "main turn 2"])
    env = main.env
    driver = make_driver(main, [deliver(env, 1, "follow-up A")], rounds=1)
    driver.run("fix the bug")

    dumped = json.loads(env.files[f"{WORKSPACE}/messages.json"])
    # Raw message list, pasted as-is.
    assert dumped == main.messages[: len(dumped)]


def test_blackbox_trajectory_referenced_in_place_no_copy():
    class FakeBlackboxAgent:
        IDLE_STATUS = "Completed"
        ENV_TRAJECTORY_FILES = ["/tmp/fake-logs/session.txt"]

        def __init__(self, env, replies):
            self.env = env
            self.messages = []
            self._replies = list(replies)

        def run(self, task):
            self.messages.append({"role": "user", "content": task})
            reply = self._replies.pop(0)
            self.messages.append({"role": "assistant", "content": reply})
            return self.IDLE_STATUS, reply

    env = DummyEnv()
    main = FakeBlackboxAgent(env, ["done 1", "done 2"])
    driver = make_driver(main, [deliver(env, 1, "follow-up A")], rounds=1)

    exit_status, result = driver.run("fix the bug")

    assert (exit_status, result) == ("Completed", "done 2")
    # The in-pod session log is referenced directly: only query_0.md is copied.
    assert [path for path, _ in env.copies] == [f"{WORKSPACE}/query_0.md"]
    (first_prompt,) = user_messages(driver.agent)
    assert "/tmp/fake-logs/session.txt" in first_prompt
    assert main.messages[-2]["content"] == "follow-up A"


def test_agents_declare_env_trajectory_files():
    from mimoagent.agents.blackbox.claude_code import ClaudeCodeAgent
    from mimoagent.agents.blackbox.codex import CodexAgent

    assert ClaudeCodeAgent.ENV_TRAJECTORY_FILES == ["/tmp/mimo-claude-logs/claude-code.txt"]
    assert CodexAgent.ENV_TRAJECTORY_FILES == ["/tmp/mimo-codex-logs/codex.txt"]
    # Native agents write no in-env trajectory; the driver falls back to mirroring.
    assert DefaultAgent.ENV_TRAJECTORY_FILES == []

    class SubclassedClaude(ClaudeCodeAgent):  # inherits the declaration
        pass

    assert env_trajectory_files(SubclassedClaude.__new__(SubclassedClaude)) == ["/tmp/mimo-claude-logs/claude-code.txt"]
    assert env_trajectory_files(object()) == []


def test_main_agent_error_stops_driving():
    main = make_main_agent([RuntimeError("model down")])
    driver = make_driver(main, ["should never be used"], rounds=2)

    exit_status, _ = driver.run("fix the bug")

    assert exit_status == "ModelQueryError"
    assert driver.agent.model.n_calls == 0


def test_user_agent_failure_stops_driving():
    main = make_main_agent(["main turn 1"])
    driver = make_driver(main, [RuntimeError("ua model down")], rounds=2)

    exit_status, result = driver.run("fix the bug")

    assert (exit_status, result) == ("Idle", "main turn 1")
    assert main.model.n_calls == 1


def test_requires_separate_model_config():
    main = make_main_agent(["main turn 1"])
    with pytest.raises(ValueError, match="model"):
        UserAgentDriver(main, {"rounds": 1, "tools": []})


def test_rejects_bad_rounds():
    main = make_main_agent(["main turn 1"])
    with pytest.raises(ValueError, match="rounds"):
        make_driver(main, [], rounds=0)


def test_config_from_dict_ignores_unknown_keys():
    cfg = {"rounds": 2, "model": {"model_name": "x"}, "legacy_key": True}
    main = make_main_agent(["main turn 1"])
    driver = UserAgentDriver(main, cfg, model=ScriptedModel([]))
    assert isinstance(driver.config, UserAgentDriverConfig)
    assert driver.config.rounds == 2
