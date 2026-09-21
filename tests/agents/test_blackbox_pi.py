"""Focused host-side tests for the Pi blackbox adapter."""

import json
from pathlib import Path

import pytest

from mimoagent.agents.blackbox.pi import PiAgent, PiAgentConfig
from mimoagent.agents.factory import get_agent_class, list_agent_types, make_agent


class _Stats:
    def __init__(self):
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_creation_tokens = 0


class _ModelConfig:
    model_name = "blackbox-rollout"
    model_kwargs = {"base_url": "http://router/sequence/v1", "api_key": "dummy-key"}


class _Model:
    def __init__(self):
        self.config = _ModelConfig()
        self.n_calls = 0
        self.token_stats = _Stats()

    def query(self, *args, **kwargs):
        return {}

    def get_template_vars(self):
        return {}


def _assistant_event(text="Fixed the bug", stop_reason="stop", *, input_tokens=10, output_tokens=3):
    content = [{"type": "text", "text": text}] if text else []
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": content,
            "stopReason": stop_reason,
            "usage": {
                "input": input_tokens,
                "output": output_tokens,
                "cacheRead": 2,
                "cacheWrite": 1,
            },
        },
    }


class FakeEnv:
    class _Config:
        cwd = "/testbed"

    def __init__(self, *, events=None, runner_rc=0, reason="ok", install_output="PI_INSTALL_OK\n"):
        self.config = self._Config()
        self.events = events or [_assistant_event()]
        self.runner_rc = runner_rc
        self.reason = reason
        self.install_output = install_output
        self.commands = []
        self.copies = []
        self.copies_out = []
        self.detached = []

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

    @property
    def raw_events(self):
        return "\n".join(json.dumps(event) for event in self.events) + "\n"

    def copy_to(self, src, dst, **kwargs):
        self.copies.append((src, dst))

    def copy_out(self, src, dst, **kwargs):
        self.copies_out.append((src, dst))
        Path(dst).write_text(self.raw_events, encoding="utf-8")

    def execute(self, command, cwd="", timeout=None):
        self.commands.append(command)
        if "mimo-install-pi" in command:
            return {"output": self.install_output, "returncode": 0}
        if "tail -c" in command:
            return {"output": self.raw_events, "returncode": 0}
        if "--mode json" in command:
            return {"output": "", "returncode": self.runner_rc, "reason": self.reason}
        return {"output": "", "returncode": 0}


def test_factory_lists_and_resolves_pi():
    assert "pi" in list_agent_types()
    assert get_agent_class("pi") is PiAgent


def test_version_is_pinned():
    assert PiAgentConfig().version == "0.83.0"
    installer = (Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources/install-pi.sh").read_text()
    assert 'PI_VERSION="${PI_VERSION:-0.83.0}"' in installer
    # standalone binary is the only install path: the upstream GitHub release
    # tarball (binary + startup assets); no npm fallback, no node bootstrap
    assert "npm install" not in installer
    assert "PI_NODE_VERSION" not in installer
    assert "PI_PACKAGES" not in installer
    assert "PI_STANDALONE" not in installer
    assert 'fetch_github_release "earendil-works/pi" "v${PI_VERSION}"' in installer
    assert 'ASSET="pi-linux-x64.tar.gz"' in installer
    assert 'payload_override PI "$STANDALONE_DIR"' in installer


def test_provider_is_chat_completions_and_router_compatible():
    agent = PiAgent(_Model(), FakeEnv(), skip_install=True)
    config = json.loads(agent._provider_config())
    provider = config["providers"]["gateway"]
    assert provider["baseUrl"] == "http://router/sequence/v1"
    assert provider["api"] == "openai-completions"
    assert provider["compat"] == {
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "supportsStore": False,
        "maxTokensField": "max_tokens",
    }
    assert provider["models"] == [
        {
            "id": "blackbox-rollout",
            "reasoning": True,
            "input": ["text", "image"],
            "contextWindow": 262144,
            "maxTokens": 32768,
        }
    ]


def test_provider_appends_v1_once():
    model = _Model()
    model.config.model_kwargs = {"base_url": "http://router/sequence/", "api_key": "x"}
    provider = json.loads(PiAgent(model, FakeEnv(), skip_install=True)._provider_config())["providers"]["gateway"]
    assert provider["baseUrl"] == "http://router/sequence/v1"


def test_anthropic_messages_provider_strips_v1_and_keeps_signature_seams():
    # Pi's Anthropic driver appends /v1/messages to baseUrl itself, so a
    # configured .../v1 must be stripped; the OpenAI compat block does not
    # apply, and session-affinity headers give multi-replica gateways a
    # sticky-routing hook for the prompt cache.
    model = _Model()
    model.config.model_kwargs = {"base_url": "http://router/sequence/v1", "api_key": "x"}
    agent = PiAgent(model, FakeEnv(), skip_install=True, api="anthropic-messages")
    provider = json.loads(agent._provider_config())["providers"]["gateway"]
    assert provider["baseUrl"] == "http://router/sequence"
    assert provider["api"] == "anthropic-messages"
    assert provider["compat"] == {"sendSessionAffinityHeaders": True}
    assert provider["models"][0]["reasoning"] is True


def test_unknown_api_is_rejected():
    agent = PiAgent(_Model(), FakeEnv(), skip_install=True, api="openai-responses")
    with pytest.raises(RuntimeError, match="unknown api"):
        agent._provider_config()


def test_adaptive_thinking_mirrors_builtin_opus_metadata():
    agent = PiAgent(
        _Model(),
        FakeEnv(),
        skip_install=True,
        api="anthropic-messages",
        adaptive_thinking=True,
        thinking="max",
    )
    entry = json.loads(agent._provider_config())["providers"]["gateway"]["models"][0]
    assert entry["thinkingLevelMap"] == {"xhigh": "xhigh", "max": "max"}
    assert entry["compat"] == {"forceAdaptiveThinking": True}


def test_adaptive_thinking_requires_anthropic_messages():
    agent = PiAgent(_Model(), FakeEnv(), skip_install=True, adaptive_thinking=True)
    with pytest.raises(RuntimeError, match="anthropic-messages"):
        agent._provider_config()


def test_thinking_budgets_are_staged_as_settings(tmp_path):
    env = FakeEnv()
    agent = PiAgent(
        _Model(),
        env,
        skip_install=True,
        thinking="high",
        thinking_budgets={"high": 32768},
    )
    agent.run("x")
    staged = [dst for _src, dst in env.copies]
    assert "/tmp/mimo-pi-logs/settings.json" in staged
    # Without the override no settings.json is staged (Pi's defaults apply).
    env2 = FakeEnv()
    PiAgent(_Model(), env2, skip_install=True).run("x")
    assert "/tmp/mimo-pi-logs/settings.json" not in [dst for _src, dst in env2.copies]


def test_run_folds_each_assistant_call_and_returns_final_text():
    events = [
        _assistant_event("", "toolUse", input_tokens=7, output_tokens=2),
        _assistant_event("Done", "stop", input_tokens=11, output_tokens=4),
    ]
    model = _Model()
    env = FakeEnv(events=events)
    agent = make_agent("pi", model, env)
    status, result = agent.run("Fix the bug")

    assert (status, result) == ("Completed", "Done")
    assert model.n_calls == 2
    assert model.token_stats.input_tokens == 18
    assert model.token_stats.output_tokens == 6
    assert model.token_stats.cache_read_tokens == 4
    assert model.token_stats.cache_creation_tokens == 2
    assert [message["role"] for message in agent.messages] == ["user", "assistant"]


def test_run_reports_usage_into_shared_stats():
    # Batch injects a GlobalModelStats aggregator via model.shared_stats; pi
    # must report each assistant call into it so its sessions count toward
    # the batch token progress and MIMOAGENT_GLOBAL_CALL_LIMIT.
    from mimoagent.models import GlobalModelStats

    events = [
        _assistant_event("", "toolUse", input_tokens=7, output_tokens=2),
        _assistant_event("Done", "stop", input_tokens=11, output_tokens=4),
    ]
    model = _Model()
    model.shared_stats = GlobalModelStats()
    agent = make_agent("pi", model, FakeEnv(events=events))
    status, _ = agent.run("Fix the bug")

    assert status == "Completed"
    assert model.shared_stats.n_calls == 2
    assert model.shared_stats.input_tokens == 18
    assert model.shared_stats.output_tokens == 6
    assert model.shared_stats.cache_read_tokens == 4
    assert model.shared_stats.cache_creation_tokens == 2


def test_run_is_offline_hermetic_and_resumes_without_reinstall():
    env = FakeEnv()
    agent = PiAgent(_Model(), env)
    agent.run("first")
    first_command = next(command for command in env.commands if "--mode json" in command)
    assert "PI_OFFLINE=1" in first_command
    assert "PI_TELEMETRY=0" in first_command
    assert "--provider gateway" in first_command
    assert "--no-approve" in first_command
    assert "--no-extensions" in first_command
    assert "--no-skills" in first_command
    assert "--continue" not in first_command

    env.commands.clear()
    agent.run("second")
    second_command = next(command for command in env.commands if "--mode json" in command)
    assert "--continue" in second_command
    assert not any("mimo-install-pi" in command for command in env.commands)


def test_install_uses_pinned_version_only():
    env = FakeEnv()
    PiAgent(_Model(), env).run("Fix the bug")

    install_command = next(command for command in env.commands if "mimo-install-pi" in command)
    assert "PI_VERSION=0.83.0" in install_command
    assert "PI_NODE_VERSION" not in install_command
    assert "PI_PACKAGES" not in install_command


def test_install_error_redacts_url_credentials():
    env = FakeEnv(install_output="download failed: http://secret@mirror.invalid/pi\n")
    with pytest.raises(RuntimeError) as error:
        PiAgent(_Model(), env).run("x")
    assert "secret" not in str(error.value)
    assert "http://[redacted]@mirror.invalid/pi" in str(error.value)


def test_json_error_is_pi_error_even_when_cli_returns_zero():
    event = _assistant_event("", "error")
    event["message"]["errorMessage"] = "gateway unavailable"
    status, result = PiAgent(_Model(), FakeEnv(events=[event]), skip_install=True).run("x")
    assert (status, result) == ("PiError", "gateway unavailable")


def test_nonzero_exit_and_missing_events_are_pi_error():
    status, _ = PiAgent(_Model(), FakeEnv(events=[], runner_rc=1), skip_install=True).run("x")
    assert status == "PiError"


def test_copies_event_log_next_to_message_file(tmp_path):
    env = FakeEnv()
    msg_path = tmp_path / "agent_msgs" / "main.log"
    agent = PiAgent(_Model(), env, skip_install=True, msg_path=msg_path)
    agent.run("x")
    assert env.copies_out == [("/tmp/mimo-pi-logs/pi.txt", str(tmp_path / "agent_msgs" / "pi.txt"))]


def test_run_goes_through_the_detached_exec():
    env = FakeEnv()
    PiAgent(_Model(), env, skip_install=True, run_timeout=7200, stall_timeout=900).run("t")
    (call,) = env.detached
    assert "--mode json" in call["command"]
    assert call["cwd"] == "/testbed"
    assert call["timeout"] == 7200
    assert call["idle_files"] == ["/tmp/mimo-pi-logs/pi.txt"]
    assert call["idle_timeout"] == 900
