"""Bash tool timeout resolution: config default, model override, max cap."""

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.bash import BashTool


@pytest.fixture
def env():
    return LocalEnvironment()


def test_defaults():
    tool = BashTool()
    assert tool.config.timeout == 30
    assert tool.config.max_timeout == 600


def test_model_param_used(env):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    BashTool().execute({"command": "true", "timeout": 120}, {"env": env})
    assert captured["timeout"] == 120


def test_model_param_capped_at_max(env):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    BashTool().execute({"command": "true", "timeout": 99999}, {"env": env})
    assert captured["timeout"] == 600


def test_config_default_when_param_absent(env):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    BashTool({"timeout": 45}).execute({"command": "true"}, {"env": env})
    assert captured["timeout"] == 45


def test_config_default_capped_by_max(env):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    BashTool({"timeout": 1200, "max_timeout": 600}).execute({"command": "true"}, {"env": env})
    assert captured["timeout"] == 600


@pytest.mark.parametrize("bad", ["abc", -5, 0, None])
def test_invalid_param_falls_back_to_default(env, bad):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    params = {"command": "true"}
    if bad is not None:
        params["timeout"] = bad
    BashTool().execute(params, {"env": env})
    assert captured["timeout"] == 30


def test_numeric_string_param_accepted(env):
    captured = {}
    orig = env.execute

    def spy(command, cwd="", timeout=None):
        captured["timeout"] = timeout
        return orig(command, cwd=cwd, timeout=timeout)

    env.execute = spy
    BashTool().execute({"command": "true", "timeout": "90"}, {"env": env})
    assert captured["timeout"] == 90


def test_timeout_surfaced_as_error_output():
    """A timed-out command is an operation failure: error ToolOutput (with the
    partial output and a retry hint), not a ToolException."""

    class TimeoutEnv:
        def execute(self, command, cwd="", timeout=None):
            return {"output": "partial output", "returncode": None, "reason": "pod_timeout"}

    out = BashTool().execute({"command": "sleep 5", "timeout": 1}, {"env": TimeoutEnv()})
    assert not out.success
    assert out.output.startswith("Error:")
    assert "timed out after 1s" in out.output
    assert "partial output" in out.output
    assert "600" in out.output  # hints the max the model may retry with
    assert out.metadata["reason"] == "pod_timeout"


def test_local_timeout_still_raises_something(env):
    """LocalEnvironment raises subprocess.TimeoutExpired — surfaced as a
    ToolException ('Failed to execute command'), not a crash."""
    from mimoagent.tools.base import ToolException

    with pytest.raises(ToolException):
        BashTool().execute({"command": "sleep 5", "timeout": 1}, {"env": env})


def test_schema_advertises_timeout():
    schema = BashTool().get_function_parameters()
    assert "timeout" in schema["properties"]
    assert "timeout" not in schema["required"]
    desc = schema["properties"]["timeout"]["description"]
    assert "30" in desc and "600" in desc
