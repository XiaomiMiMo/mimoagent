"""Tests for the Claude Code blackbox agent + the agent factory.

These exercise the host-side orchestration (install → stage → run → collect)
against a fake environment; the SDK runner itself runs inside the pod and is not
imported here.
"""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimoagent.agents.blackbox.claude_code import ClaudeCodeAgent
from mimoagent.agents.factory import get_agent_class, list_agent_types, make_agent


class _Stats:
    def __init__(self):
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_creation_tokens = 0


class _ModelCfg:
    model_name = "anthropic/claude-opus-4-7-004"
    model_kwargs = {"base_url": "http://gw/anthropic", "api_key": "sk-test"}


class _Model:
    def __init__(self):
        self.config = _ModelCfg()
        self.n_calls = 0
        self.token_stats = _Stats()

    def query(self, *a, **k):
        return {}

    def get_template_vars(self):
        return {}


class FakeEnv:
    """Records copy_to/copy_out/execute; simulates the pod's runner + sdk_result.json."""

    class _C:
        cwd = "/testbed"

    def __init__(self, *, runner_rc=0, is_error=False):
        self.config = self._C()
        self.cmds = []
        self.copies = []
        self.copy_contents = {}
        self.copies_out = []
        self.detached = []
        self._runner_rc = runner_rc
        self._is_error = is_error

    def execute_detached(self, command, cwd="", timeout=None, *, idle_files=(), idle_timeout=0):
        self.detached.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
                "idle_files": list(idle_files),
                "idle_timeout": idle_timeout,
            }
        )
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src, dst, **k):
        self.copies.append((src, dst))
        try:
            self.copy_contents[dst] = Path(src).read_text()
        except (OSError, UnicodeDecodeError):
            pass

    def copy_out(self, src, dst, **k):
        self.copies_out.append((src, dst))

    def execute(self, command, cwd="", timeout=None):
        self.cmds.append(command)
        if "install-claude-code" in command:
            return {"output": "CLAUDE_CODE_INSTALL_OK\n", "returncode": 0}
        if "run-claude-sdk" in command or "run_claude_sdk" in command:
            return {"output": "runner done", "returncode": self._runner_rc}
        if "sdk_result.json" in command:
            return {
                "output": json.dumps(
                    {
                        "result": "Fixed the bug",
                        "is_error": self._is_error,
                        "num_turns": 5,
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "cache_read_input_tokens": 5,
                            "cache_creation_input_tokens": 1,
                        },
                    }
                ),
                "returncode": 0,
            }
        return {"output": "", "returncode": 0}


def test_factory_lists_and_resolves():
    assert "default" in list_agent_types()
    assert "claude-code" in list_agent_types()
    assert get_agent_class("default").__name__ == "DefaultAgent"
    assert get_agent_class("claude-code").__name__ == "ClaudeCodeAgent"


def test_factory_unknown_type_raises():
    with pytest.raises(ValueError):
        get_agent_class("does-not-exist")


def test_env_derivation_reuses_model_config_with_base_url():
    agent = ClaudeCodeAgent(_Model(), FakeEnv())
    env = agent._build_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert env["ANTHROPIC_BASE_URL"] == "http://gw/anthropic"
    # base_url set => keep full provider-prefixed model name
    assert env["ANTHROPIC_MODEL"] == "anthropic/claude-opus-4-7-004"
    for tier in (
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[tier] == env["ANTHROPIC_MODEL"]


def test_env_derivation_strips_prefix_without_base_url():
    m = _Model()
    m.config.model_kwargs = {"api_key": "sk-test"}  # no base_url
    agent = ClaudeCodeAgent(m, FakeEnv())
    env = agent._build_env()
    assert "ANTHROPIC_BASE_URL" not in env
    assert env["ANTHROPIC_MODEL"] == "claude-opus-4-7-004"


def test_run_happy_path_folds_usage_and_messages():
    m = _Model()
    env = FakeEnv()
    agent = make_agent("claude-code", m, env, max_turns=10)
    status, msg = agent.run("Fix the bug")

    assert status == "Completed"
    assert msg == "Fixed the bug"
    # usage folded into token stats + n_calls
    assert m.token_stats.input_tokens == 100
    assert m.token_stats.output_tokens == 20
    assert m.n_calls == 5
    # self.messages stays minimal: prompt + final reply (the full Claude Code
    # session is copied out to claude-code.txt, not parsed into messages).
    roles = [msg["role"] for msg in agent.messages]
    assert roles == ["user", "assistant"]
    assert agent.messages[-1]["content"] == "Fixed the bug"
    # save_traj relies on this being empty for blackbox agents
    assert agent.get_model_query_kwargs() == {}
    # staged the instruction + runner + install script
    staged = [d for _, d in env.copies]
    assert any("install-claude-code" in d for d in staged)
    assert any("run-claude-sdk" in d for d in staged)


def test_copies_session_log_out(tmp_path):
    # The raw claude-code.txt is copied out of the pod, next to the msg file.
    env = FakeEnv()
    msg_path = tmp_path / "agent_msgs" / "main.log"
    agent = ClaudeCodeAgent(_Model(), env, skip_install=True, msg_path=str(msg_path))
    agent.run("Fix the bug")
    assert env.copies_out == [
        ("/tmp/mimo-claude-logs/claude-code.txt", str(tmp_path / "agent_msgs" / "claude-code.txt"))
    ]


def test_copy_log_out_noop_without_msg_path():
    # No msg_path → nothing to copy next to; copy_out is skipped.
    env = FakeEnv()
    agent = ClaudeCodeAgent(_Model(), env, skip_install=True)
    agent.run("Fix the bug")
    assert env.copies_out == []


def test_multi_turn_skips_reinstall_and_resumes():
    env = FakeEnv()
    agent = ClaudeCodeAgent(_Model(), env)
    agent.run("first")
    assert len(agent.messages) == 2  # prompt + reply per turn
    env.cmds.clear()
    agent.run("second")
    assert not any("install-claude-code" in c for c in env.cmds)
    assert any("--continue" in c for c in env.cmds)
    assert len(agent.messages) == 4


def test_runner_nonzero_is_claude_code_error():
    agent = ClaudeCodeAgent(_Model(), FakeEnv(runner_rc=1), skip_install=True)
    status, _ = agent.run("x")
    assert status == "ClaudeCodeError"


def test_result_is_error_flag_marks_error():
    agent = ClaudeCodeAgent(_Model(), FakeEnv(is_error=True), skip_install=True)
    status, _ = agent.run("x")
    assert status == "ClaudeCodeError"


def test_skip_install_does_not_upload_install_script():
    env = FakeEnv()
    agent = ClaudeCodeAgent(_Model(), env, skip_install=True)
    agent.run("x")
    assert not any("install-claude-code" in d for _, d in env.copies)


def test_cli_controls_are_forwarded_through_options_file():
    env = FakeEnv()
    agent = ClaudeCodeAgent(
        _Model(),
        env,
        skip_install=True,
        cli_args={
            "tools": "Bash,Read,Edit",
            "exclude-dynamic-system-prompt-sections": None,
            "setting-sources": "project",
        },
        disallowed_tools=["WebSearch", "WebFetch"],
    )
    agent.run("x")

    options = json.loads(env.copy_contents["/tmp/mimo-claude-options.json"])
    assert options == {
        "cli_args": {
            "tools": "Bash,Read,Edit",
            "exclude-dynamic-system-prompt-sections": None,
            "setting-sources": "project",
        },
        "disallowed_tools": ["WebSearch", "WebFetch"],
    }
    assert any("--options-file=/tmp/mimo-claude-options.json" in command for command in env.cmds)


def test_sdk_runner_applies_cli_controls(monkeypatch):
    class _Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Hook:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        SimpleNamespace(ClaudeAgentOptions=_Options, HookMatcher=_Hook),
    )
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources/run_claude_sdk.py")
    )

    options = runner["_build_options"](
        "model",
        "/testbed",
        10,
        cli_args={"tools": "Bash,Read,Edit", "setting-sources": ""},
        disallowed_tools=["WebSearch", "WebFetch"],
    )
    assert options.kwargs["extra_args"] == {
        "tools": "Bash,Read,Edit",
        "setting-sources": "",
    }
    assert options.kwargs["disallowed_tools"] == ["WebSearch", "WebFetch"]


def test_sdk_runner_rejects_shell_style_cli_arg_keys(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        SimpleNamespace(ClaudeAgentOptions=object, HookMatcher=object),
    )
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources/run_claude_sdk.py")
    )
    with pytest.raises(ValueError, match="without leading '--'"):
        runner["_build_options"]("model", "/testbed", 10, cli_args={"--tools": "Read"})


def test_sdk_runner_never_echoes_session_to_stdout(capsys, tmp_path, monkeypatch):
    """Regression: stdout is the k8s-exec pipe and it is O_NONBLOCK. Echoing
    each SDK message there let a full pipe freeze the runner's event loop in a
    synchronous retry loop, so the CLI issued no further requests and the
    rollout was scored as an agent timeout."""
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        SimpleNamespace(ClaudeAgentOptions=object, HookMatcher=object),
    )
    runner = runpy.run_path(
        str(Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources/run_claude_sdk.py")
    )
    assert "_write_stdout" not in runner

    log = tmp_path / "claude-code.txt"
    with log.open("w") as handle:
        runner["_log_message"](SimpleNamespace(type="assistant", text="x" * 4096), handle)
    assert "assistant" in log.read_text()
    assert capsys.readouterr().out == ""


def test_collect_falls_back_to_pod_session_log():
    """Without sdk_result.json the result text comes from the pod's
    claude-code.txt tail, since stdout no longer carries the session."""
    env = FakeEnv()
    passthrough = env.execute

    def execute(command, cwd="", timeout=None):
        if "sdk_result.json" in command:
            return {"output": "", "returncode": 0}
        if "claude-code.txt" in command:
            return {"output": "tail from pod\n", "returncode": 0}
        return passthrough(command, cwd=cwd, timeout=timeout)

    env.execute = execute
    agent = ClaudeCodeAgent(_Model(), env)
    result_text, status = agent._collect(0, "stderr noise only")
    assert result_text == "tail from pod"
    assert status == agent.IDLE_STATUS


def test_runner_goes_through_the_detached_exec():
    """The Claude Code session is the one exec that can outlive the k8s
    websocket lifetime cap; install, staging and result reads stay attached."""
    env = FakeEnv()
    agent = make_agent("claude-code", _Model(), env, run_timeout=7200)
    agent.run("Fix the bug")

    (call,) = env.detached
    assert "run-claude-sdk" in call["command"]
    assert call["timeout"] == 7200
    assert call["idle_files"] == ["/tmp/mimo-claude-logs/claude-code.txt"]
    assert call["idle_timeout"] == 0  # watchdog off unless configured
    attached = [c for c in env.cmds if c not in {call["command"]}]
    assert any("install-claude-code" in c for c in attached)
    assert any("sdk_result.json" in c for c in attached)


def test_stall_timeout_reaches_the_watchdog():
    env = FakeEnv()
    make_agent("claude-code", _Model(), env, stall_timeout=1800).run("t")
    assert env.detached[0]["idle_timeout"] == 1800


def test_detached_transport_error_is_infra_error():
    from mimoagent.agents.base import InfraError

    env = FakeEnv()

    def execute_detached(command, cwd="", timeout=None, **kw):
        return {"output": "", "returncode": None, "reason": "transport_error"}

    env.execute_detached = execute_detached
    with pytest.raises(InfraError):
        make_agent("claude-code", _Model(), env).run("t")


def test_unknown_outcome_is_never_completed():
    """client_timeout / stall leave rc=None: the agent must not report success
    even though sdk_result.json from an earlier turn might still be readable."""
    env = FakeEnv()
    env.execute_detached = lambda command, cwd="", timeout=None, **kw: {
        "output": "",
        "returncode": None,
        "reason": "stall",
    }
    status, _ = make_agent("claude-code", _Model(), env).run("t")
    assert status == "ClaudeCodeError"


def test_env_without_detached_exec_falls_back_to_attached_run(capsys):
    env = FakeEnv()
    del FakeEnv.execute_detached
    try:
        status, msg = make_agent("claude-code", _Model(), env).run("Fix the bug")
    finally:
        FakeEnv.execute_detached = _FAKE_DETACHED
    assert (status, msg) == ("Completed", "Fixed the bug")
    assert any("run-claude-sdk" in c for c in env.cmds)
    assert "no execute_detached" in capsys.readouterr().out


_FAKE_DETACHED = FakeEnv.execute_detached
