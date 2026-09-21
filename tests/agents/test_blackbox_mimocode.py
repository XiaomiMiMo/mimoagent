"""Focused version-pin tests for the MiMo-Code blackbox harness."""

import json
from pathlib import Path
from types import SimpleNamespace

from mimoagent.agents.blackbox.mimocode import MimoCodeAgent, MimoCodeAgentConfig


def test_rebuilt_version_is_pinned_in_agent_and_installer():
    expected = "0.1.12"
    assert MimoCodeAgentConfig().version == expected

    installer = (Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources/install-mimocode.sh").read_text()
    assert f'MIMOCODE_VERSION="${{MIMOCODE_VERSION:-{expected}}}"' in installer


class _Env:
    config = SimpleNamespace(cwd="/testbed")

    def __init__(self):
        self.commands = []
        self.detached = []

    def execute(self, command, *args, **kwargs):
        self.commands.append(command)
        return {"output": "", "returncode": 0}

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

    def copy_to(self, *args, **kwargs):
        pass


def _agent(**kwargs) -> MimoCodeAgent:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_name="aws/claude-opus-4-8-006",
            model_kwargs={"base_url": "http://router", "api_key": "k"},
        ),
        n_calls=0,
    )
    return MimoCodeAgent(model, _Env(), **kwargs)


def test_channel_model_names_keep_their_slashes():
    # model_name is the serving name, sent verbatim; channel-prefixed names
    # (aws/claude-...) must reach the gateway intact.
    agent = _agent()
    assert agent._resolved_model == "aws/claude-opus-4-8-006"
    config = json.loads(agent._provider_config())
    assert "aws/claude-opus-4-8-006" in config["provider"]["gateway"]["models"]


def test_thinking_budget_lands_in_model_options():
    # --variant does not apply to custom gateway providers; options.thinking
    # is the mechanism that reaches the request's thinking parameter.
    entry = json.loads(_agent(thinking_budget=8192)._provider_config())["provider"]["gateway"]["models"][
        "aws/claude-opus-4-8-006"
    ]
    assert entry["options"] == {"thinking": {"type": "enabled", "budgetTokens": 8192}}

    plain = json.loads(_agent()._provider_config())["provider"]["gateway"]["models"]["aws/claude-opus-4-8-006"]
    assert "options" not in plain


def test_thinking_effort_lands_as_variant_and_run_flag():
    import pytest

    agent = _agent(thinking_effort="max")
    entry = json.loads(agent._provider_config())["provider"]["gateway"]["models"]["aws/claude-opus-4-8-006"]
    assert entry["variants"] == {"max": {"thinking": {"type": "adaptive"}, "effort": "max"}}
    agent._copy_text_to_pod = lambda *a, **k: None
    agent._run_mimocode("task")
    # the run must select the handwritten variant
    assert any("--variant max" in c for c in getattr(agent.env, "commands", []))

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _agent(thinking_budget=8192, thinking_effort="max")._provider_config()


def test_partial_limit_block_is_rejected():
    # mimo's schema requires context/output together; a partial block would
    # kill every session at startup with a config-validation error.
    import pytest

    with pytest.raises(ValueError, match="must be set together"):
        _agent(context_limit=1000000)._provider_config()
    with pytest.raises(ValueError, match="must be set together"):
        _agent(output_limit=65536)._provider_config()
    entry = json.loads(_agent(context_limit=1000000, output_limit=65536)._provider_config())["provider"]["gateway"][
        "models"
    ]["aws/claude-opus-4-8-006"]
    assert entry["limit"] == {"context": 1000000, "output": 65536}


def test_max_turns_caps_both_primary_agents():
    # mimo counts iterations per agent and defaults to unlimited (steps ?? Inf);
    # plan_enter switches to the other primary mid-run, so capping one is not
    # enough.
    config = json.loads(_agent(max_turns=500)._provider_config())
    assert config["agent"] == {"build": {"steps": 500}, "plan": {"steps": 500}}
    assert "agent" not in json.loads(_agent()._provider_config())


def test_codex_mode_is_disabled_by_default_and_overridable():
    env = _agent()._build_env()
    assert env["MIMOCODE_CODEX_MODE"] == "false"
    overridden = _agent(extra_env={"MIMOCODE_CODEX_MODE": "true"})._build_env()
    assert overridden["MIMOCODE_CODEX_MODE"] == "true"


def test_run_goes_through_the_detached_exec():
    agent = _agent(skip_install=True, run_timeout=7200, stall_timeout=900)
    agent.run("t")
    (call,) = agent.env.detached
    assert "mimo run" in call["command"]
    assert call["cwd"] == "/testbed"
    assert call["timeout"] == 7200
    assert call["idle_files"] == ["/tmp/mimo-mimocode-logs/mimocode.txt"]
    assert call["idle_timeout"] == 900
