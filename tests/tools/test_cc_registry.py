"""Tests for the CC-aligned tool catalogue and CCAgent wiring.

Covers what the port from the cc_aligned_tools flag design must preserve:
- the CC registry resolves capitalized AND lowercase names to CC classes;
- the original ToolRegistry is untouched (lowercase catalogue only);
- CCAgent runs an end-to-end scripted rollout over CC tool names;
- Grep stages a static rg into an environment that lacks one.
"""

from __future__ import annotations

import json

import pytest

from mimoagent.agents.cc.cc_agent import CCAgent, CCAgentConfig
from mimoagent.agents.factory import get_agent_class, list_agent_types
from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools import ToolException, ToolRegistry
from mimoagent.tools.cc import CC_TOOL_CLASSES, CCToolRegistry

# ---- registry separation ------------------------------------------------------


def test_cc_registry_resolves_capitalized_and_lowercase():
    reg = CCToolRegistry.from_config([{"tool": "Bash"}, {"tool": "read"}, {"tool": "GREP"}])
    assert reg.list_tools() == ["Bash", "Read", "Grep"]


def test_cc_registry_rejects_unknown():
    with pytest.raises(ToolException, match="Unknown tool type"):
        CCToolRegistry.from_config([{"tool": "TodoWrite"}])


def test_original_registry_untouched():
    # The original catalogue must keep resolving lowercase names to the
    # original classes — and must NOT know the CC-only tools.
    reg = ToolRegistry.from_config([{"tool": "bash"}, {"tool": "read"}])
    assert reg.list_tools() == ["bash", "read"]
    with pytest.raises(ToolException, match="Unknown tool type"):
        ToolRegistry.from_config([{"tool": "grep"}])


def test_cc_catalogue_names_are_capitalized():
    for name, cls in CC_TOOL_CLASSES.items():
        assert name[0].isupper()
        assert cls({}).name == name


def test_factory_registers_cc_agent():
    assert "cc-agent" in list_agent_types()
    assert get_agent_class("cc-agent") is CCAgent


# ---- CCAgent end-to-end over CC schemas ----------------------------------------


class ScriptedModel:
    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._index = -1
        self.n_calls = 0

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.n_calls += 1
        self._index += 1
        return self._responses[self._index]

    def get_template_vars(self) -> dict:
        return {}


def _call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_cc_agent_runs_cc_bash(tmp_path):
    marker = tmp_path / "ran.txt"
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("c1", "Bash", {"command": f"echo hi > {marker}"})]},
            {"content": "done"},
        ]
    )
    agent = CCAgent(model=model, env=LocalEnvironment(), tools=[{"tool": "Bash"}], step_limit=5)
    status, _ = agent.run("do it")
    assert status == "Idle"
    assert marker.read_text().strip() == "hi"
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1


def test_cc_agent_read_uses_file_path_schema(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("print('hello')\n")
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("c1", "Read", {"file_path": str(f)})]},
            {"content": "done"},
        ]
    )
    agent = CCAgent(model=model, env=LocalEnvironment(), tools=[{"tool": "Read"}], step_limit=5)
    status, _ = agent.run("read it")
    assert status == "Idle"
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert "print('hello')" in tool_msgs[0]["content"]


def test_cc_agent_default_config_is_cc_names():
    cfg = CCAgentConfig()
    assert {spec["tool"] for spec in cfg.tools} == {"Bash", "Read", "Write", "Edit", "Agent"}


def test_stray_tool_call_block_raises_feedback_turn():
    stray = '<tool_call>{"name": "TodoWrite", "arguments": {}}</tool_call>'
    model = ScriptedModel(
        [
            {"content": stray},
            {"content": "ok, stopping"},
        ]
    )
    agent = CCAgent(model=model, env=LocalEnvironment(), tools=[{"tool": "Bash"}], step_limit=5)
    status, _ = agent.run("task")
    # FormatError becomes a user feedback turn, then the next response idles.
    assert status == "Idle"
    feedback = [m for m in agent.messages if m["role"] == "user" and "TodoWrite" in str(m.get("content"))]
    assert feedback, "expected a feedback turn naming the stray tool"
    assert agent.tool_call_errors[0] is True


# ---- Grep: static-rg staging ---------------------------------------------------


class _NoRgEnv(LocalEnvironment):
    """LocalEnvironment that pretends rg is absent until one is copied in."""

    def __init__(self):
        super().__init__()
        self.copied: list[tuple[str, str]] = []
        self._staged = False

    def execute(self, command: str, cwd: str = "", timeout: int = None):
        if "command -v rg" in command and not self._staged:
            return {"output": "", "returncode": 1}
        if command.startswith("chmod +x") and "command -v rg" in command:
            return {"output": "/usr/local/bin/rg", "returncode": 0}
        return super().execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, **kwargs):
        self.copied.append((src_path, dest_path))
        self._staged = True


def test_grep_stages_a_static_rg_when_missing(monkeypatch, tmp_path):
    from mimoagent.tools.cc import grep as grep_mod
    from mimoagent.tools.cc.grep import GrepTool

    # the host-side resolver is exercised in tests/tools/test_ripgrep.py; here
    # it hands back a fake binary so the test needs no network
    fake_rg = tmp_path / "rg"
    fake_rg.write_text("#!/bin/sh\nexit 0\n")
    fake_rg.chmod(0o755)
    monkeypatch.setattr(grep_mod, "resolve_host_rg", lambda: fake_rg)
    env = _NoRgEnv()
    tool = GrepTool({})
    out = tool.execute({"pattern": "def ", "path": "src/mimoagent/tools/cc"}, {"env": env})
    assert out.success
    assert env.copied and env.copied[0][1] == "/usr/local/bin/rg"
    # Second call must not re-copy (memoized per env).
    tool.execute({"pattern": "class ", "path": "src/mimoagent/tools/cc"}, {"env": env})
    assert len(env.copied) == 1


def test_grep_runs_directly_when_rg_present():
    from mimoagent.tools.cc.grep import GrepTool

    env = LocalEnvironment()
    which = env.execute("command -v rg").get("output", "").strip()
    if not which:
        pytest.skip("rg not installed on the host")
    tool = GrepTool({})
    out = tool.execute(
        {"pattern": "class GrepTool", "path": "src/mimoagent/tools/cc", "output_mode": "files_with_matches"},
        {"env": env},
    )
    assert out.success
    assert "grep.py" in out.output
