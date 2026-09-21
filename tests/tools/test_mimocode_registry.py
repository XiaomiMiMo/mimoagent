"""Focused tests for the stage-1 MiMo native catalogue."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
from pathlib import Path

import pytest

import mimoagent.agents.mimocode
from mimoagent.agents.factory import get_agent_class, list_agent_types
from mimoagent.agents.mimocode import MimocodeAgent, MimocodeAgentConfig
from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.base import ToolException
from mimoagent.tools.mimocode import (
    MIMOCODE_CORE_TOOL_NAMES,
    ActorTool,
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    MimocodeToolRegistry,
    ReadTool,
    TaskTool,
    WriteTool,
)


class ScriptedModel:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.index = -1

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.index += 1
        return self.responses[self.index]

    def get_template_vars(self) -> dict:
        return {}


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_mimocode_registry_uses_lowercase_protocol_names():
    registry = MimocodeToolRegistry.from_config(
        [{"tool": "bash"}, {"tool": "READ"}, {"tool": "grep"}, {"tool": "Glob"}]
    )

    assert registry.list_tools() == ["bash", "read", "grep", "glob"]
    assert set(MIMOCODE_CORE_TOOL_NAMES) == {"bash", "read", "write", "edit", "grep", "glob", "task"}


def test_glob_matches_mimocode_schema_and_observation(tmp_path):
    schema = GlobTool({}).get_function_parameters()
    assert list(schema["properties"]) == ["pattern", "path"]
    assert schema["required"] == ["pattern"]
    assert schema["additionalProperties"] is False

    (tmp_path / "visible.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("pass\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    tool = GlobTool({})

    result = tool.execute({"pattern": "*.py"}, context)
    assert result.success
    assert result.metadata == {"count": 2, "truncated": False}
    assert str(tmp_path / "visible.py") in result.output
    assert str(tmp_path / ".hidden.py") in result.output

    missing = tool.execute({"pattern": "*.missing"}, context)
    assert missing.output == "No files found"
    assert missing.metadata == {"count": 0, "truncated": False}

    file_path = tool.execute({"pattern": "*.py", "path": "visible.py"}, context)
    assert not file_path.success
    assert "glob path must be a directory" in file_path.output


def _host_rg() -> str:
    """A real rg for the tests that run searches on the host (skip when absent)."""
    from mimoagent.tools.ripgrep import RipgrepUnavailable, resolve_host_rg

    found = shutil.which("rg")
    if found:
        return found
    try:
        return str(resolve_host_rg(download=False))
    except RipgrepUnavailable:
        pytest.skip("rg not installed on the host and no cached static binary")


class _NoRgMimocodeEnv(LocalEnvironment):
    """Exercise lazy static-rg staging: rg looks absent until one is copied in,
    after which searches run with the host's rg."""

    def __init__(self, cwd):
        super().__init__(cwd=str(cwd))
        self.copied: list[tuple[str, str]] = []
        self.staged = False

    def execute(self, command: str, cwd: str = "", timeout: int = None):
        if command.strip() == "command -v rg || true" and not self.staged:
            return {"output": "", "returncode": 0}
        if command.startswith("chmod +x /usr/local/bin/rg && command -v rg"):
            self.staged = True
            return {"output": "/usr/local/bin/rg\n", "returncode": 0}
        if self.staged and command.startswith("rg "):
            command = f"{_host_rg()}{command[2:]}"
        return super().execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, **kwargs):
        self.copied.append((src_path, dest_path))


def test_task_schema_uses_action_specific_operations_without_session_scope():
    schema = TaskTool({}).get_function_parameters()
    operations = schema["properties"]["operation"]["anyOf"]
    assert [op["properties"]["action"]["const"] for op in operations] == [
        "create",
        "list",
        "get",
        "start",
        "block",
        "unblock",
        "done",
        "abandon",
        "rename",
    ]
    create = operations[0]
    assert create["required"] == ["action", "summary"]
    assert create["properties"]["summary"]["minLength"] == 1
    assert "session_id" not in create["properties"]
    assert "include_archived" not in operations[1]["properties"]
    assert all(op["additionalProperties"] is False for op in operations)
    # The description must document the verbs the schema actually accepts.
    description = TaskTool({}).description
    for verb in ("create", "list", "get", "start", "block", "unblock", "done", "abandon", "rename"):
        assert f"- {verb}:" in description
    documented_options = {line.split("optional:")[1] for line in description.splitlines() if "optional:" in line}
    assert not any("session_id" in opt or "include_archived" in opt for opt in documented_options)


def test_task_unblock_returns_task_to_open():
    tool = TaskTool({})
    context = {"state": {}}
    tool.execute({"operation": {"action": "create", "summary": "Track a fix"}}, context)
    tool.execute({"operation": {"action": "block", "id": "T1", "event_summary": "waiting"}}, context)

    result = tool.execute({"operation": {"action": "unblock", "id": "T1"}}, context)

    assert result.success
    assert context["state"]["tasks"]["T1"]["status"] == "open"


def test_mimocode_grep_matches_content_contract_and_truncates(tmp_path):
    from mimoagent.tools.mimocode.grep import GrepTool

    _host_rg()
    source = tmp_path / "source.py"
    source.write_text("\n".join(f"needle {i}" for i in range(105)) + "\n", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("needle outside include\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    tool = GrepTool({})

    schema = tool.get_function_parameters()
    assert list(schema["properties"]) == ["pattern", "path", "include"]
    result = tool.execute({"pattern": "needle", "include": "*.py"}, context)
    assert result.success
    assert result.metadata == {"matches": 105, "truncated": True}
    assert result.output.startswith("Found 105 matches (showing first 100)")
    assert str(source) in result.output
    assert str(other) not in result.output
    assert "(Results truncated: showing 100 of 105 matches (5 hidden)." in result.output

    empty = tool.execute({"pattern": "does-not-exist"}, context)
    assert empty.output == "No files found"
    assert empty.metadata == {"matches": 0, "truncated": False}


def test_mimocode_grep_rejects_invalid_regex(tmp_path):
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = GrepTool({}).execute({"pattern": "["}, context)

    assert not result.success
    assert "ripgrep failed" in result.output


def test_mimocode_glob_truncates_large_result_sets(tmp_path):
    for index in range(105):
        (tmp_path / f"file_{index:03d}.txt").write_text("x\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = GlobTool({}).execute({"pattern": "*.txt"}, context)

    assert result.success
    assert result.metadata == {"count": 100, "truncated": True}
    assert "(Results are truncated: showing first 100 results." in result.output


def test_mimocode_glob_cap_is_applied_to_a_path_ordered_candidate_set(monkeypatch, tmp_path):
    """Structural guard: the cap must not depend on ripgrep's emission order.

    The reproducibility tests below observe real ``rg`` output, so they only catch
    the bug while that walker happens to be unordered. This one feeds a shuffled
    stream directly and asserts which paths survive, so it holds regardless.
    """
    from mimoagent.tools.mimocode import glob as glob_module

    names = [f"f{index:03d}.txt" for index in range(150)]
    shuffled = names[75:] + names[:75]
    monkeypatch.setattr(GlobTool, "_ensure_rg", lambda self, env: None)
    monkeypatch.setattr(GlobTool, "_path_type", staticmethod(lambda env, path, context: "DIR"))
    monkeypatch.setattr(glob_module.GlobTool, "_file_mtimes", staticmethod(lambda env, paths, context: {}))

    class _Env:
        def execute(self, command, **kwargs):
            return {"returncode": 0, "output": "\n".join(shuffled)}

    result = GlobTool({}).execute({"pattern": "*.txt"}, {"env": _Env(), "state": {"cwd": str(tmp_path)}})

    shown = [line.rsplit("/", 1)[-1] for line in result.output.splitlines() if line.endswith(".txt")]
    assert shown == names[:100]
    assert result.metadata == {"count": 100, "truncated": True}


def test_mimocode_glob_picks_a_reproducible_slice_when_mtimes_tie(tmp_path):
    """Which 100 paths survive the cap must not depend on ripgrep's walk order.

    A freshly checked-out testbed gives nearly every file one identical mtime, so
    without a deterministic key the shown slice varies between identical runs and
    the rollout stops being replayable.
    """
    for index in range(105):
        path = tmp_path / f"file_{index:03d}.txt"
        path.write_text("x\n", encoding="utf-8")
        os.utime(path, (1_600_000_000, 1_600_000_000))
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    runs = [GlobTool({}).execute({"pattern": "*.txt"}, context).output for _ in range(3)]

    assert len(set(runs)) == 1
    shown = [line for line in runs[0].splitlines() if line.endswith(".txt")]
    assert shown == sorted(shown)
    assert len(shown) == 100
    # The cap keeps the first 100 by path, so the tail of the tree is what drops.
    assert shown[0].endswith("file_000.txt")
    assert "file_104.txt" not in runs[0]


def test_mimocode_glob_orders_by_mtime_within_the_kept_slice(tmp_path):
    for index in range(3):
        path = tmp_path / f"file_{index}.txt"
        path.write_text("x\n", encoding="utf-8")
        os.utime(path, (1_600_000_000 + index, 1_600_000_000 + index))
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    output = GlobTool({}).execute({"pattern": "*.txt"}, context).output

    assert [line.rsplit("/", 1)[-1] for line in output.splitlines()] == [
        "file_2.txt",
        "file_1.txt",
        "file_0.txt",
    ]


def test_mimocode_grep_match_order_is_reproducible_when_mtimes_tie(tmp_path):
    for index in range(4):
        path = tmp_path / f"mod_{index}.py"
        path.write_text("needle\nneedle\n", encoding="utf-8")
        os.utime(path, (1_600_000_000, 1_600_000_000))
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    runs = [GrepTool({}).execute({"pattern": "needle"}, context).output for _ in range(3)]

    assert len(set(runs)) == 1
    paths = [line[:-1] for line in runs[0].splitlines() if line.endswith(":")]
    assert paths == sorted(paths)


def test_mimocode_grep_and_glob_stage_the_same_static_rg(tmp_path, monkeypatch):
    from mimoagent.tools.mimocode import glob as glob_mod
    from mimoagent.tools.mimocode import grep as grep_mod
    from mimoagent.tools.mimocode.glob import GlobTool
    from mimoagent.tools.mimocode.grep import GrepTool

    host_rg = pathlib.Path(_host_rg())
    monkeypatch.setattr(grep_mod, "resolve_host_rg", lambda: host_rg)
    monkeypatch.setattr(glob_mod, "resolve_host_rg", lambda: host_rg)

    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    context = {"state": {"cwd": str(tmp_path)}}

    grep_env = _NoRgMimocodeEnv(tmp_path)
    grep_context = {**context, "env": grep_env}
    grep_tool = GrepTool({})
    assert grep_tool.execute({"pattern": "needle"}, grep_context).success
    assert grep_tool.execute({"pattern": "needle"}, grep_context).success
    assert len(grep_env.copied) == 1
    assert all(dest == "/usr/local/bin/rg" for _, dest in grep_env.copied)

    glob_env = _NoRgMimocodeEnv(tmp_path)
    glob_context = {**context, "env": glob_env}
    assert GlobTool({}).execute({"pattern": "*.py"}, glob_context).success
    assert len(glob_env.copied) == 1
    assert glob_env.copied[0][1] == "/usr/local/bin/rg"


def test_actor_is_optional_and_replaces_legacy_agent_tool():
    defaults = MimocodeToolRegistry.from_config(MimocodeAgentConfig().tools)
    assert "actor" not in defaults.list_tools()

    registry = MimocodeToolRegistry.from_config([{"tool": "actor"}])
    assert registry.list_tools() == ["actor"]
    assert isinstance(registry.get("actor"), ActorTool)

    with pytest.raises(ToolException, match="Unknown tool type: agent"):
        MimocodeToolRegistry.from_config([{"tool": "agent"}])


def test_actor_schema_only_exposes_synchronous_run():
    schema = ActorTool({}).get_function_parameters()
    operation = schema["properties"]["operation"]

    assert operation["properties"]["action"]["enum"] == ["run"]
    assert operation["properties"]["subagent_type"]["enum"] == ["explore", "general"]
    assert set(operation["required"]) == {"action", "subagent_type", "description", "prompt"}
    assert "actor_id" not in operation["properties"]


def test_mimocode_actor_child_tool_sets_are_not_cc_roster():
    from mimoagent.agents.mimocode import (
        available_mimocode_subagent_types,
        mimocode_subagent_kwargs,
    )

    assert available_mimocode_subagent_types() == ["explore", "general"]
    # The roster is the single source of truth for a child's tool set: the actor
    # tool advertises it verbatim, so explore must carry the dedicated search
    # tools here rather than have them patched in at dispatch time.
    assert [spec["tool"] for spec in mimocode_subagent_kwargs("explore")["tools"]] == [
        "bash",
        "read",
        "grep",
        "glob",
        "task",
    ]
    assert [spec["tool"] for spec in mimocode_subagent_kwargs("general")["tools"]] == [
        "bash",
        "read",
        "write",
        "edit",
        "task",
    ]
    with pytest.raises(KeyError, match="Unknown Mimocode subagent type"):
        mimocode_subagent_kwargs("plan")


def test_mimocode_actor_description_matches_the_roster_tool_sets():
    """The advertised tool list is rendered from the roster, so it cannot drift."""
    from mimoagent.agents.mimocode import MIMOCODE_SUBAGENT_PRESETS, mimocode_subagent_kwargs
    from mimoagent.tools.mimocode.actor import ActorTool

    description = ActorTool({}).description
    for name, preset in MIMOCODE_SUBAGENT_PRESETS.items():
        assert preset["description"] in description
        registered = [spec["tool"] for spec in mimocode_subagent_kwargs(name)["tools"]]
        assert f"Tools: {', '.join(registered)}." in description


def test_mimocode_actor_prompts_only_describe_exposed_capabilities():
    """Each child prompt must name exactly the tools that child gets registered."""
    from mimoagent.agents.mimocode import mimocode_subagent_kwargs
    from mimoagent.tools.mimocode import MIMOCODE_TOOL_CLASSES
    from mimoagent.tools.mimocode.actor import ActorTool

    for name in ("explore", "general"):
        kwargs = mimocode_subagent_kwargs(name)
        registered = {spec["tool"] for spec in kwargs["tools"]}
        prompt = kwargs["system_template"]
        named = {tool for tool in MIMOCODE_TOOL_CLASSES if f"`{tool}`" in prompt}
        assert named == registered, f"{name} prompt names {named} but registers {registered}"

    description = ActorTool({}).description
    assert "`run` is the only supported operation" in description
    schema = ActorTool({}).get_function_parameters()
    assert schema["properties"]["operation"]["properties"]["action"]["enum"] == ["run"]


def test_actor_run_is_available_only_when_enabled_and_returns_child_result():
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _call(
                        "a1",
                        "actor",
                        {
                            "operation": {
                                "action": "run",
                                "subagent_type": "explore",
                                "description": "Inspect one path",
                                "prompt": "Read the target and report the answer.",
                            }
                        },
                    )
                ],
            },
            {"content": "child answer"},
            {"content": "parent finished"},
        ]
    )
    agent = MimocodeAgent(
        model=model,
        env=LocalEnvironment(),
        tools=[{"tool": "actor"}],
        step_limit=5,
    )

    status, message = agent.run("delegate one lookup")

    assert status == "Idle"
    assert message == "parent finished"
    assert len(agent.subagents) == 1
    assert "actor" not in agent.subagents[0].tool_registry.list_tools()
    actor_message = next(item for item in agent.messages if item.get("role") == "tool")
    assert actor_message["content"].startswith("child answer")


def test_actor_child_read_state_does_not_authorize_parent_write(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("original\n", encoding="utf-8")
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _call(
                        "a1",
                        "actor",
                        {
                            "operation": {
                                "action": "run",
                                "subagent_type": "explore",
                                "description": "Read target",
                                "prompt": "Read target.txt and report its contents.",
                            }
                        },
                    )
                ],
            },
            {"content": "", "tool_calls": [_call("r1", "read", {"file_path": "target.txt"})]},
            {"content": "child read original"},
            {
                "content": "",
                "tool_calls": [
                    _call(
                        "w1",
                        "write",
                        {"file_path": "target.txt", "content": "parent overwrite\n"},
                    )
                ],
            },
            {"content": "parent finished"},
        ]
    )
    agent = MimocodeAgent(
        model=model,
        env=LocalEnvironment(cwd=str(tmp_path)),
        tools=[{"tool": "actor"}, {"tool": "write"}],
        step_limit=8,
    )

    status, message = agent.run("delegate a read, then try to overwrite the same file")

    assert status == "Idle"
    assert message == "parent finished"
    assert target.read_text(encoding="utf-8") == "original\n"
    write_result = [m for m in agent.messages if m.get("tool_call_id") == "w1"][0]
    assert "must read" in write_result["content"]


def test_mimocode_agent_is_registered_and_defaults_to_core_non_task_tools():
    assert "mimocode-agent" in list_agent_types()
    assert get_agent_class("mimocode-agent") is MimocodeAgent
    assert {spec["tool"] for spec in MimocodeAgentConfig().tools} == {
        "bash",
        "read",
        "write",
        "edit",
        "grep",
        "glob",
        "task",
    }


def test_mimocode_system_prompt_only_injects_cwd(tmp_path):
    model = ScriptedModel([{"content": "done"}])
    agent = MimocodeAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[])

    status, _ = agent.run("inspect the environment")

    assert status == "Idle"
    system = agent.messages[0]["content"]
    # The packaged prompt is rendered with the session cwd and nothing else:
    # LocalEnvironment exposes platform and os.environ as template vars, none
    # of which may reach the model.
    packaged = (Path(mimoagent.agents.mimocode.__file__).parent / "prompts" / "mimocode_core.txt").read_text(
        encoding="utf-8"
    )
    assert system == packaged.rstrip().replace("{{cwd}}", str(tmp_path))
    assert "{{" not in system


def test_bash_cd_does_not_persist_and_cwd_metadata(tmp_path):
    workdir = tmp_path / "nested"
    workdir.mkdir()
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_call("c1", "bash", {"command": "cd / && pwd", "description": "cd then pwd"})],
            },
            {"content": "", "tool_calls": [_call("c2", "bash", {"command": "pwd", "description": "pwd"})]},
            {"content": "done"},
        ]
    )

    agent = MimocodeAgent(
        model=model,
        env=LocalEnvironment(cwd=str(workdir)),
        tools=[{"tool": "bash"}],
        step_limit=6,
    )
    status, _ = agent.run("check cwd")

    assert status == "Idle"
    assert agent.tool_state["cwd"] == str(workdir)
    tool_messages = [message for message in agent.messages if message["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("/\n")
    assert str(workdir) in tool_messages[1]["content"]


def test_task_operation_state_persists_across_calls():
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_call("t1", "task", {"operation": {"action": "create", "summary": "Fix bug"}})],
            },
            {"content": "", "tool_calls": [_call("t2", "task", {"operation": {"action": "start", "id": "T1"}})]},
            {
                "content": "",
                "tool_calls": [
                    _call("t3", "task", {"operation": {"action": "done", "id": "T1", "event_summary": "tests pass"}})
                ],
            },
            {
                "content": "",
                "tool_calls": [_call("t4", "task", {"operation": {"action": "list", "include_terminal": True}})],
            },
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(model=model, env=LocalEnvironment(), tools=[{"tool": "task"}], step_limit=6)
    status, _ = agent.run("track work")

    assert status == "Idle"
    assert agent.tool_state["tasks"]["T1"]["status"] == "done"


def test_read_and_write_match_mimocode_observation_shapes(tmp_path):
    env = LocalEnvironment()
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")

    read = ReadTool({}).execute({"file_path": str(path)}, {"env": env, "state": {"cwd": str(tmp_path)}})
    assert "<path>" in read.output
    assert "<type>file</type>" in read.output
    assert "1: one" in read.output
    assert "</content>" in read.output

    target = tmp_path / "written.txt"
    written = WriteTool({}).execute(
        {"file_path": str(target), "content": "written\n"},
        {"env": env, "state": {"cwd": str(tmp_path)}},
    )
    assert written.output == "Wrote file successfully."
    assert target.read_text(encoding="utf-8") == "written\n"

    state = {"cwd": str(tmp_path), "read_files": set()}
    blocked = WriteTool({}).execute(
        {"file_path": str(target), "content": "blocked\n"},
        {"env": env, "state": state},
    )
    assert not blocked.success
    ReadTool({}).execute({"file_path": str(target)}, {"env": env, "state": state})
    edited = EditTool({}).execute(
        {"file_path": str(target), "old_string": "written", "new_string": "edited"},
        {"env": env, "state": state},
    )
    assert edited.success
    assert edited.output == "Edit applied successfully."
    assert edited.metadata["diff"]
    assert target.read_text(encoding="utf-8") == "edited\n"


def test_file_tools_resolve_paths_relative_to_agent_cwd(tmp_path):
    source = tmp_path / "relative.txt"
    source.write_text("before\n", encoding="utf-8")
    env = LocalEnvironment(cwd=str(tmp_path))
    state = {"cwd": str(tmp_path), "read_files": set()}
    context = {"env": env, "state": state}

    read = ReadTool({}).execute({"file_path": "relative.txt"}, context)
    assert read.success
    assert "1: before" in read.output

    written = WriteTool({}).execute({"file_path": "created.txt", "content": "created\n"}, context)
    assert written.success
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"

    edited = EditTool({}).execute(
        {"file_path": "relative.txt", "old_string": "before", "new_string": "after"},
        context,
    )
    assert edited.success
    assert source.read_text(encoding="utf-8") == "after\n"


def test_read_directory_and_long_line_follow_mimocode_shape(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    long_file = tmp_path / "long.txt"
    long_file.write_text("x" * 2100, encoding="utf-8")
    env = LocalEnvironment()
    context = {"env": env, "state": {"cwd": str(tmp_path)}}

    directory = ReadTool({}).execute({"file_path": str(tmp_path), "offset": 2, "limit": 1}, context)
    assert "<type>directory</type>" in directory.output
    assert "Showing 1 of" in directory.output
    assert "b.txt" in directory.output
    assert str(tmp_path / "b.txt") not in directory.output

    file_output = ReadTool({}).execute({"file_path": str(long_file)}, context)
    assert "line truncated to 2000 chars" in file_output.output

    no_newline = tmp_path / "no-newline.txt"
    no_newline.write_bytes(b"one\ntwo")
    no_newline_output = ReadTool({}).execute({"file_path": str(no_newline)}, context)
    assert "End of file - total 2 lines" in no_newline_output.output


def test_mimocode_edit_diff_stays_within_a_bounded_metadata_payload(tmp_path):
    """A diff rides in metadata, which is appended after output truncation.

    Editing a minified single-line file used to emit a ~400k-character
    observation that no context window can absorb, ending the rollout.
    """
    target = tmp_path / "min.js"
    target.write_text("a" * 200_000 + "\n", encoding="utf-8")
    context = {
        "env": LocalEnvironment(cwd=str(tmp_path)),
        "state": {"cwd": str(tmp_path), "read_files": [str(target)]},
    }

    result = EditTool({}).execute(
        {"file_path": str(target), "old_string": "a" * 200_000, "new_string": "b" * 200_000},
        context,
    )

    assert result.success
    assert result.output == "Edit applied successfully."
    assert len(result.metadata["diff"]) < 3_000
    assert "diff truncated" in result.metadata["diff"]


def test_mimocode_task_ledger_metadata_stays_constant_size(tmp_path):
    """Capping entry count is not enough — summaries are model-authored.

    metadata is appended after the output is truncated, so anything model-sized in
    there escapes max_observation_length entirely.
    """
    state: dict = {"cwd": str(tmp_path), "tasks": {}, "read_files": set()}
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": state}
    tool = TaskTool({})
    for _ in range(60):
        tool.execute({"operation": {"action": "create", "summary": "P" * 800}}, context)

    result = tool.execute({"operation": {"action": "list"}}, context)

    assert result.metadata["task_count"] == 60
    assert result.metadata["task_counts_by_status"] == {"open": 60}
    assert len(str(result.metadata)) < 200
    agent = MimocodeAgent(model=ScriptedModel([]), env=LocalEnvironment(cwd=str(tmp_path)), tools=[])
    observation = agent._format_observation(result.to_dict())
    assert "task_state" not in observation
    # No inherited char cut for this catalogue: the ledger listing reaches the
    # model as the tool wrote it, and only the constant-size metadata rides along.
    assert "TRUNCATION WARNING" not in observation
    assert observation.count("[open]") == 60


def test_mimocode_edit_empty_old_string_refuses_an_existing_file(tmp_path):
    """`old_string=""` is the documented shorthand for a new file only.

    The engine underneath refuses an empty old_string; routing past that into a
    write would turn one typo into silent truncation of the file being fixed,
    since the read gate only asks that the path was read.
    """
    target = tmp_path / "important.py"
    target.write_text("def critical():\n    return 1\n", encoding="utf-8")
    context = {
        "env": LocalEnvironment(cwd=str(tmp_path)),
        "state": {"cwd": str(tmp_path), "tasks": {}, "read_files": {str(target)}},
    }

    result = EditTool({}).execute({"file_path": str(target), "old_string": "", "new_string": "# oops\n"}, context)

    assert not result.success
    assert "already exists" in result.output
    assert target.read_text(encoding="utf-8") == "def critical():\n    return 1\n"

    fresh = tmp_path / "brand_new.py"
    created = EditTool({}).execute({"file_path": str(fresh), "old_string": "", "new_string": "print(1)\n"}, context)
    assert created.success
    assert created.output == "Edit applied successfully."
    assert fresh.read_text(encoding="utf-8") == "print(1)\n"


def test_mimocode_write_and_create_register_the_file_as_read(tmp_path):
    """A file the model just authored must be editable without a separate read.

    Only ``read`` used to register ``read_files``, so ``write x`` then ``edit x``
    was refused with "You must read x before editing it" — 41% of mimocode
    trajectories hit it, most on files the agent had just created.
    """
    state: dict = {"cwd": str(tmp_path), "tasks": {}, "read_files": set()}
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": state}

    written = tmp_path / "written.py"
    assert WriteTool({}).execute({"file_path": str(written), "content": "x = 1\n"}, context).success
    assert str(written) in state["read_files"]
    edited = EditTool({}).execute({"file_path": str(written), "old_string": "x = 1", "new_string": "x = 2"}, context)
    assert edited.success
    assert written.read_text(encoding="utf-8") == "x = 2\n"

    created = tmp_path / "created.py"
    assert EditTool({}).execute({"file_path": str(created), "old_string": "", "new_string": "y = 1\n"}, context).success
    assert str(created) in state["read_files"]
    edited = EditTool({}).execute({"file_path": str(created), "old_string": "y = 1", "new_string": "y = 2"}, context)
    assert edited.success
    assert created.read_text(encoding="utf-8") == "y = 2\n"

    # A relative path is registered under its resolved absolute form, which is
    # also what the edit gate looks up.
    assert WriteTool({}).execute({"file_path": "rel.py", "content": "z = 1\n"}, context).success
    assert str(tmp_path / "rel.py") in state["read_files"]
    assert EditTool({}).execute({"file_path": "rel.py", "old_string": "z = 1", "new_string": "z = 2"}, context).success


def test_mimocode_failed_write_does_not_register_the_file_as_read(tmp_path):
    state: dict = {"cwd": str(tmp_path), "tasks": {}, "read_files": set()}
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": state}
    existing = tmp_path / "existing.py"
    existing.write_text("a\n", encoding="utf-8")

    blocked = WriteTool({}).execute({"file_path": str(existing), "content": "b\n"}, context)

    assert not blocked.success
    assert str(existing) not in state["read_files"]


def test_mimocode_grep_survives_more_files_than_one_shell_command_can_carry(tmp_path):
    """The mtime probe carries every matched path inside the command string.

    One `set -- "$@" <path>` per file crosses Linux MAX_ARG_STRLEN at a few
    thousand files, and a wide pattern on a real testbed reaches that easily.
    """
    nested = tmp_path / ("d" * 40) / ("e" * 40)
    nested.mkdir(parents=True)
    for index in range(3000):
        (nested / f"module_name_{index:05d}.py").write_text("needle\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = GrepTool({}).execute({"pattern": "needle"}, context)

    assert result.success
    assert result.output.startswith("Found 3000 matches (showing first 100)")
    shown = [line[:-1] for line in result.output.splitlines() if line.endswith(":")]
    assert len(shown) == 100
    assert shown == sorted(shown)


def test_mimocode_core_prompt_does_not_promise_absent_machinery():
    """The prompt must not assert a mechanism this build does not have.

    Describing a tool that is off by default is fine — the model simply won't be
    offered it. What is not fine is stating that something happens for the model
    when nothing does: there is no threshold-triggered compaction here, so a
    promise of unlimited context makes it skip economising until the real limit
    ends the rollout. ``task`` is also a plain ledger, not a dispatcher.
    """
    prompt = MimocodeAgentConfig().system_template

    assert "automatically compress" not in prompt
    assert "not limited by the context window" not in prompt
    assert "context window is finite" in prompt
    assert "via the Task /" not in prompt
    assert "Prefer dedicated tools over shelling out" in prompt


def test_mimocode_default_tools_come_from_the_core_tool_names(tmp_path):
    """One source of truth, so the constant and the default config cannot drift."""
    agent = MimocodeAgent(model=ScriptedModel([]), env=LocalEnvironment(cwd=str(tmp_path)))

    assert [spec["tool"] for spec in MimocodeAgentConfig().tools] == list(MIMOCODE_CORE_TOOL_NAMES)
    assert [d["function"]["name"] for d in agent._tool_definitions] == list(MIMOCODE_CORE_TOOL_NAMES)


def test_mimocode_read_rejects_a_directory_offset_past_the_end(tmp_path):
    """An out-of-range offset must fail like the file branch, not look complete."""
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = ReadTool({}).execute({"file_path": str(tmp_path), "offset": 9}, context)

    assert not result.success
    assert "out of range for this directory (2 entries)" in result.output


def test_mimocode_edit_creating_a_file_keeps_edits_success_line(tmp_path):
    target = tmp_path / "new.py"
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = EditTool({}).execute({"file_path": str(target), "old_string": "", "new_string": "print(1)\n"}, context)

    assert result.success
    assert result.output == "Edit applied successfully."
    assert target.read_text(encoding="utf-8") == "print(1)\n"


def test_mimocode_blocked_tool_result_does_not_advertise_the_guard(tmp_path):
    """A blocked call must look like a plain failure, or the guard is probeable."""
    agent = MimocodeAgent(
        model=ScriptedModel([]),
        env=LocalEnvironment(cwd=str(tmp_path)),
        tools=[{"tool": "bash"}],
        antihack={"enabled": True},
    )

    blocked = agent.execute_action({"tool": "bash", "params": {"command": "git clone https://x/y", "description": "d"}})

    assert agent.antihack.blocks
    assert not blocked.get("metadata")
    assert "Tool metadata" not in agent._format_observation(blocked)


def test_mimocode_bash_description_states_the_configured_timeout():
    """The upstream resource hardcodes 120000ms; a configured run must say its own."""
    description = BashTool({"timeout": 60, "max_timeout": 300}).description

    assert "60000ms" in description
    assert "300000ms" in description
    assert "120000ms" not in description


def test_mimocode_read_marks_subdirectories_with_a_trailing_slash(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "plain.txt").write_text("x\n", encoding="utf-8")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    output = ReadTool({}).execute({"file_path": str(tmp_path)}, context).output

    assert "subdir/" in output
    assert "\nplain.txt\n" in output


def test_antihack_covers_the_mimocode_lowercase_schemas():
    """mimocode reuses the lowercase ids with file_path, so the guard must scan both."""
    from mimoagent.agents.antihack import AntiHackConfig, AntiHackGuard

    guard = AntiHackGuard(AntiHackConfig(enabled=True))
    artifact = "/testbed/tests/test_patch.diff"

    assert guard.inspect({"tool": "read", "params": {"file_path": artifact}}) is not None
    assert guard.inspect({"tool": "read", "params": {"path": artifact}}) is not None
    assert guard.inspect({"tool": "grep", "params": {"pattern": "test_patch"}}) is not None
    assert guard.inspect({"tool": "glob", "params": {"pattern": "**/test_patch*"}}) is not None
    assert guard.inspect({"tool": "read", "params": {"file_path": "/testbed/src/app.py"}}) is None


def test_child_state_shares_the_ledger_but_not_the_read_gate():
    from mimoagent.tools.mimocode.state import derive_child_state, new_tool_state

    parent = new_tool_state("/testbed")
    parent["read_files"].add("/testbed/app.py")
    child = derive_child_state(parent)

    assert child["cwd"] == "/testbed"
    assert child["tasks"] is parent["tasks"]
    assert child["read_files"] == set()
    child["read_files"].add("/testbed/other.py")
    assert parent["read_files"] == {"/testbed/app.py"}


def test_actor_child_cannot_overwrite_a_file_only_the_parent_read(tmp_path):
    """The child never sees the parent's conversation, so the parent's reads grant
    it nothing — it must look at a file before overwriting it."""
    target = tmp_path / "important.py"
    target.write_text("original\n", encoding="utf-8")
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("r1", "read", {"file_path": str(target)})]},
            {
                "content": "",
                "tool_calls": [
                    _call(
                        "a1",
                        "actor",
                        {
                            "operation": {
                                "action": "run",
                                "subagent_type": "general",
                                "description": "overwrite it",
                                "prompt": "overwrite the file",
                            }
                        },
                    )
                ],
            },
            {
                "content": "",
                "tool_calls": [_call("w1", "write", {"file_path": str(target), "content": "child\n"})],
            },
            {"content": "child done"},
            {"content": "parent done"},
        ]
    )
    agent = MimocodeAgent(
        model=model,
        env=LocalEnvironment(cwd=str(tmp_path)),
        tools=[{"tool": "read"}, {"tool": "actor"}],
        step_limit=8,
    )

    status, _ = agent.run("read the file, then delegate an overwrite")

    assert status == "Idle"
    assert target.read_text(encoding="utf-8") == "original\n"
    child = agent.subagents[0]
    child_write = [m for m in child.messages if m.get("tool_call_id") == "w1"][0]
    assert "must read" in child_write["content"]


def test_actor_child_reads_and_updates_the_shared_task_ledger(tmp_path):
    """The ledger is shared by reference, so a child needs the tool to reach it."""
    from mimoagent.agents.mimocode import mimocode_subagent_kwargs
    from mimoagent.tools.mimocode.state import derive_child_state, new_tool_state

    parent = new_tool_state(str(tmp_path))
    parent_ctx = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": parent}
    TaskTool({}).execute({"operation": {"action": "create", "summary": "planned by parent"}}, parent_ctx)

    child = derive_child_state(parent)
    child_ctx = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": child}
    listed = TaskTool({}).execute({"operation": {"action": "list"}}, child_ctx)
    TaskTool({}).execute({"operation": {"action": "start", "id": "T1"}}, child_ctx)

    assert "planned by parent" in listed.output
    assert parent["tasks"]["T1"]["status"] == "in_progress"
    for preset in ("explore", "general"):
        assert "task" in {spec["tool"] for spec in mimocode_subagent_kwargs(preset)["tools"]}


def test_mimocode_task_list_status_filter_returns_terminal_tasks(tmp_path):
    """An explicit status is the whole filter.

    Layering the default terminal exclusion on top made `status="done"`
    self-cancelling: the schema offers done/abandoned, so a legal call returned
    nothing and reported success.
    """
    context = {
        "env": LocalEnvironment(cwd=str(tmp_path)),
        "state": {"cwd": str(tmp_path), "tasks": {}, "read_files": set()},
    }
    tool = TaskTool({})
    tool.execute({"operation": {"action": "create", "summary": "shipped"}}, context)
    tool.execute({"operation": {"action": "create", "summary": "pending"}}, context)
    tool.execute({"operation": {"action": "done", "id": "T1"}}, context)

    assert tool.execute({"operation": {"action": "list", "status": "done"}}, context).output == "T1 [done] shipped"
    assert tool.execute({"operation": {"action": "list", "status": "open"}}, context).output == "T2 [open] pending"
    assert tool.execute({"operation": {"action": "list", "status": "abandoned"}}, context).output == "No tasks."
    # The default listing still hides terminal tasks.
    assert tool.execute({"operation": {"action": "list"}}, context).output == "T2 [open] pending"
    both = tool.execute({"operation": {"action": "list", "include_terminal": True}}, context)
    assert both.metadata["count"] == 2


def test_mimocode_concurrency_defaults_match_other_whitebox_agents():
    config = MimocodeAgentConfig()
    assert config.tool_parallel_workers == 8
    assert config.actor_max_concurrency == 1


def test_mimocode_actor_semaphore_clamps_non_positive_concurrency():
    agent = MimocodeAgent(
        model=ScriptedModel([{"content": "done"}]),
        env=LocalEnvironment(),
        actor_max_concurrency=0,
    )
    # Clamped to 1: first acquire succeeds, second does not.
    assert agent.actor_semaphore.acquire(blocking=False)
    assert not agent.actor_semaphore.acquire(blocking=False)


def test_mimocode_actor_semaphore_honors_configured_concurrency():
    agent = MimocodeAgent(
        model=ScriptedModel([{"content": "done"}]),
        env=LocalEnvironment(),
        actor_max_concurrency=2,
    )
    assert agent.actor_semaphore.acquire(blocking=False)
    assert agent.actor_semaphore.acquire(blocking=False)
    assert not agent.actor_semaphore.acquire(blocking=False)


def test_actor_semaphore_limits_concurrent_children_to_one():
    """Default actor_max_concurrency=1 serializes concurrent actor.run calls."""
    import threading
    import time
    from unittest.mock import patch

    from mimoagent.tools.base import ToolOutput
    from mimoagent.tools.mimocode.actor import ActorTool

    active = 0
    max_active = 0
    lock = threading.Lock()

    def tracking_child(self, operation, context, env, model, subagent_type):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        # Hold the slot long enough for the peer to attempt entry.
        time.sleep(0.1)
        with lock:
            active -= 1
        return ToolOutput(output="child done", success=True, metadata={})

    agent = MimocodeAgent(
        model=ScriptedModel([{"content": "done"}]),
        env=LocalEnvironment(),
        tools=[{"tool": "actor"}],
    )
    tool = agent.tool_registry.get("actor")
    params = {
        "operation": {
            "action": "run",
            "subagent_type": "explore",
            "description": "noop",
            "prompt": "noop",
        }
    }
    context = agent.get_tool_context()

    with patch.object(ActorTool, "_execute_child", tracking_child):
        threads = [threading.Thread(target=tool.execute, args=(params, context)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert max_active == 1


def test_actor_semaphore_allows_configured_concurrency():
    import threading
    from unittest.mock import patch

    from mimoagent.tools.base import ToolOutput
    from mimoagent.tools.mimocode.actor import ActorTool

    active = 0
    max_active = 0
    lock = threading.Lock()
    both_inside = threading.Barrier(2, timeout=5)

    def tracking_child(self, operation, context, env, model, subagent_type):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        both_inside.wait()
        with lock:
            active -= 1
        return ToolOutput(output="child done", success=True, metadata={})

    agent = MimocodeAgent(
        model=ScriptedModel([{"content": "done"}]),
        env=LocalEnvironment(),
        tools=[{"tool": "actor"}],
        actor_max_concurrency=2,
    )
    tool = agent.tool_registry.get("actor")
    params = {
        "operation": {
            "action": "run",
            "subagent_type": "explore",
            "description": "noop",
            "prompt": "noop",
        }
    }
    context = agent.get_tool_context()

    with patch.object(ActorTool, "_execute_child", tracking_child):
        threads = [threading.Thread(target=tool.execute, args=(params, context)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert max_active == 2


def test_actor_runs_ungated_without_parent_agent():
    """Direct ActorTool construction (no context['agent']) still executes."""
    from unittest.mock import patch

    from mimoagent.tools.base import ToolOutput
    from mimoagent.tools.mimocode.actor import ActorTool

    calls = []

    def recording_child(self, operation, context, env, model, subagent_type):
        calls.append(subagent_type)
        return ToolOutput(output="ok", success=True, metadata={})

    tool = ActorTool({})
    params = {
        "operation": {
            "action": "run",
            "subagent_type": "explore",
            "description": "noop",
            "prompt": "noop",
        }
    }
    context = {"env": LocalEnvironment(), "model": ScriptedModel([])}

    with patch.object(ActorTool, "_execute_child", recording_child):
        result = tool.execute(params, context)

    assert result.success
    assert calls == ["explore"]


def test_actor_semaphore_releases_after_child_raises():
    """`with semaphore` must release on TransportError / any exception."""
    from unittest.mock import patch

    from mimoagent.environments import TransportError
    from mimoagent.tools.mimocode.actor import ActorTool

    def boom(self, operation, context, env, model, subagent_type):
        raise TransportError("pod gone")

    agent = MimocodeAgent(
        model=ScriptedModel([{"content": "done"}]),
        env=LocalEnvironment(),
        tools=[{"tool": "actor"}],
        actor_max_concurrency=1,
    )
    tool = agent.tool_registry.get("actor")
    params = {
        "operation": {
            "action": "run",
            "subagent_type": "explore",
            "description": "noop",
            "prompt": "noop",
        }
    }
    context = agent.get_tool_context()

    with patch.object(ActorTool, "_execute_child", boom):
        with pytest.raises(TransportError, match="pod gone"):
            tool.execute(params, context)

    # Slot is free again after the failed child.
    assert agent.actor_semaphore.acquire(blocking=False)


# ---- observation size is bounded by the tools, not by the inherited char cut ----


def _tool_messages(agent: MimocodeAgent) -> list[str]:
    return [message["content"] for message in agent.messages if message["role"] == "tool"]


def test_mimocode_agent_disables_the_inherited_char_cut():
    """The catalogue budgets its own output (like CodexAgentConfig); a second
    cut on top kept the tool's own continuation footer while deleting the lines
    it claimed to show."""
    assert MimocodeAgentConfig().max_observation_length == 0


def test_mimocode_bash_keeps_head_and_tail_of_oversized_output(tmp_path):
    """The tool's own cut has the shape of the agent-level truncate_middle:
    beginning and end survive, a marker names the gap, and the whole thing
    stays within MiMo-Code's 2000-line / 50 KB budget."""
    env = LocalEnvironment(cwd=str(tmp_path))
    context = {"env": env, "state": {"cwd": str(tmp_path)}}

    result = BashTool({}).execute({"command": "seq 1 5000", "description": "big output"}, context)

    assert result.success
    assert result.output.startswith("...output truncated...\n1\n2\n")
    head, tail = result.output[len("...output truncated...\n") :].split(
        "\n...3001 lines (15004 bytes) omitted here...\n"
    )
    assert head.split("\n") == [str(i) for i in range(1, 1001)]
    # 1000 trailing newline-separated fields: 999 numbers plus the empty field
    # after the final newline, exactly like MiMo-Code's split.
    assert tail.split("\n") == [str(i) for i in range(4002, 5001)] + [""]
    assert len(result.output.encode("utf-8")) <= 51200 + len("...output truncated...\n") + 60
    # Metadata is kept on the object (logs/tests) even though the agent no longer renders it.
    assert set(result.metadata) == {"returncode", "reason", "cwd"}


def test_mimocode_bash_head_and_tail_share_the_byte_budget_on_long_lines(tmp_path):
    """Lines of ~85 bytes: the byte budget binds before the line budget, each
    half gets about 25 KB, and the first and last lines both survive."""
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    cmd = (
        'python3 -c "'
        "print('===== session starts: FIRST LINE =====');"
        "[print(f'tests/test_mod_{i//50}.py::test_case_{i} PASSED' + ' '*40 + f'[{i*100//3000:3d}%]') for i in range(3000)];"
        "print('===== 1 failed, 3000 passed: LAST LINE =====')\""
    )

    result = BashTool({}).execute({"command": cmd, "description": "fake pytest"}, context)

    out = result.output
    assert out.startswith("...output truncated...\n===== session starts: FIRST LINE =====\n")
    assert out.endswith("===== 1 failed, 3000 passed: LAST LINE =====\n")
    head, tail = out[len("...output truncated...\n") :].split("\n...", 1)
    marker, tail = tail.split("\n", 1)
    assert marker.endswith("omitted here...")
    assert len(head.encode()) <= 51200 // 2 and len(tail.encode()) <= 51200 - len(head.encode())
    assert not set(head.split("\n")) & {line for line in tail.split("\n") if line}
    assert "[TRUNCATION WARNING]" not in out


def test_mimocode_bash_byte_budget_keeps_the_end_of_one_long_line(tmp_path):
    """A single line over the budget has no head to keep; the tail gets the
    whole budget and the marker reports the cut in bytes only."""
    env = LocalEnvironment(cwd=str(tmp_path))
    context = {"env": env, "state": {"cwd": str(tmp_path)}}

    result = BashTool({}).execute(
        {"command": "python3 -c \"print('a' * 60000 + 'END')\"", "description": "one long line"},
        context,
    )

    assert result.output.startswith("...output truncated...\n...1 lines (")
    marker, body = result.output[len("...output truncated...\n") :].split("\n", 1)
    assert marker.endswith("bytes) omitted here...")
    assert len(body.encode("utf-8")) <= 51200
    assert body.endswith("END\n")


def test_mimocode_bash_output_within_budget_is_untouched(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path))
    context = {"env": env, "state": {"cwd": str(tmp_path)}}

    result = BashTool({}).execute({"command": "seq 1 10", "description": "small output"}, context)

    assert result.output == "".join(f"{i}\n" for i in range(1, 11))


def test_mimocode_bash_description_states_the_head_and_tail_shape():
    description = BashTool({}).description
    assert "keeps the beginning and the end within those limits" in description
    assert "omitted here..." in description


def test_mimocode_bash_budget_is_configurable_and_rejects_nonsense():
    small = BashTool({"max_output_lines": 3, "max_output_bytes": 1024})
    assert small.config.max_output_lines == 3
    with pytest.raises(ValueError):
        BashTool({"max_output_lines": 0})
    with pytest.raises(ValueError):
        BashTool({"max_output_bytes": 0})


def test_mimocode_bash_description_states_the_output_budget():
    """The prompt must describe the cut the tool makes, from the same config."""
    default = BashTool({}).description
    assert "2000 lines or 51200 bytes" in default
    assert "...output truncated..." in default
    assert "${max_output_lines}" not in default

    custom = BashTool({"max_output_lines": 100, "max_output_bytes": 4096}).description
    assert "100 lines or 4096 bytes" in custom


def test_mimocode_read_caps_a_large_file_at_50kb_with_a_continuation_offset(tmp_path):
    """MiMo-Code's read.ts stops at 50 KB and tells the model where to resume.
    Without this cap the inherited char cut used to fire instead, leaving the
    tool's own "Showing lines 1-2000" footer describing lines that were gone."""
    big = tmp_path / "big.py"
    big.write_text(
        "\n".join(f"line_{i:05d} = 'x' * 40  # padding to make the line about sixty chars" for i in range(1, 3001))
    )
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    first = ReadTool({}).execute({"file_path": str(big)}, context)

    assert first.success
    numbered = [line for line in first.output.splitlines() if line[:1].isdigit() and ": " in line]
    last_shown = int(numbered[-1].split(":", 1)[0])
    assert numbered[0].startswith("1: ")
    assert 700 < last_shown < 800  # ~68 bytes/line under a 51200-byte budget
    content = "\n".join(line.split(": ", 1)[1] for line in numbered)
    assert len(content.encode("utf-8")) <= 51200
    assert (
        f"(Output capped at 50 KB. Showing lines 1-{last_shown}. Use offset={last_shown + 1} to continue.)"
        in first.output
    )
    assert "of 3000" not in first.output  # the capped footer is MiMo's, not the line-limit one

    second = ReadTool({}).execute({"file_path": str(big), "offset": last_shown + 1}, context)

    assert second.success
    assert f"{last_shown + 1}: line_{last_shown + 1:05d}" in second.output
    assert "Output capped at 50 KB" in second.output  # still more than 50 KB left


def test_mimocode_read_below_the_byte_cap_keeps_the_line_limit_footers(tmp_path):
    """Files under 50 KB behave exactly as before: the 2000-line footer or the
    end-of-file footer, with nothing removed."""
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    small = tmp_path / "small.txt"
    small.write_text("\n".join(f"row {i}" for i in range(1, 2501)))

    limited = ReadTool({}).execute({"file_path": str(small)}, context)
    assert "2000: row 2000" in limited.output
    assert "(Showing lines 1-2000 of 2500. Use offset=2001 to continue.)" in limited.output

    tail = ReadTool({}).execute({"file_path": str(small), "offset": 2001}, context)
    assert "2500: row 2500" in tail.output
    assert "(End of file - total 2500 lines)" in tail.output


def test_mimocode_read_over_40k_chars_reaches_the_model_intact(tmp_path):
    """A ~45k-char read is under MiMo's byte cap and must not be cut by the agent."""
    big = tmp_path / "mid.py"
    big.write_text("\n".join(f"value_{i:04d} = {i}  # {'p' * 40}" for i in range(1, 701)))
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("c1", "read", {"file_path": str(big)})]},
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "read"}], step_limit=4)

    status, _ = agent.run("read the file")

    assert status == "Idle"
    (observation,) = _tool_messages(agent)
    assert 40_000 < len(observation) < 51_200 + 500
    assert "[TRUNCATION WARNING]" not in observation
    assert "[...OUTPUT OMITTED" not in observation
    assert "350: value_0350" in observation
    assert "(End of file - total 700 lines)" in observation


def test_mimocode_agent_renders_oversized_bash_with_only_the_tool_cut(tmp_path):
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("c1", "bash", {"command": "seq 1 20000", "description": "big"})]},
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "bash"}], step_limit=4)

    status, _ = agent.run("dump numbers")

    assert status == "Idle"
    (observation,) = _tool_messages(agent)
    assert observation.startswith("...output truncated...\n1\n2\n")
    assert "\n...18001 lines (99006 bytes) omitted here...\n19002\n" in observation
    assert "[TRUNCATION WARNING]" not in observation
    assert "\n20000\n" in observation


# ---- the harness must not inject noise or teach a contract it does not enforce ----


def test_mimocode_edit_diff_names_the_target_and_carries_no_scratch_path(tmp_path):
    """The scratch copy's random /tmp/mimo_edit_<hex>/ path used to appear in
    the diff header: per-call random tokens in the observation."""
    target = tmp_path / "f.py"
    target.write_text("\n".join(f"line {i}" for i in range(80)) + "\n")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path), "read_files": {str(target)}}}

    result = EditTool({}).execute(
        {"file_path": str(target), "old_string": "line 0\n", "new_string": "LINE 0\n"}, context
    )

    assert result.success
    diff = result.metadata["diff"]
    assert "mimo_edit" not in diff
    assert f"+++ b{target}" in diff
    assert "+LINE 0" in diff


def test_mimocode_edit_diff_is_cut_once_by_the_char_cap(tmp_path):
    """No `head -50` line cut underneath the 2000-char cap, so the reported
    total is the real diff length."""
    target = tmp_path / "f.py"
    old = "\n".join(f"line {i:03d} {'x' * 30}" for i in range(80))
    target.write_text(old + "\n")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path), "read_files": {str(target)}}}

    result = EditTool({}).execute(
        {"file_path": str(target), "old_string": old, "new_string": old.upper()},
        context,
    )

    diff = result.metadata["diff"]
    assert "diff truncated" in diff
    assert len(diff) < 2_200
    total = int(diff.rsplit("diff truncated: ", 1)[1].split(" ")[0])
    assert total > 6_000  # 160 changed lines of ~40 chars, not the 50 lines `head -50` used to leave


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        ({"command": "true"}, "missing required parameter 'description'"),
        (
            {"command": "true", "description": "x", "timeout": "abc"},
            "parameter 'timeout' must be of type number, got str",
        ),
        (
            {"command": "true", "description": "x", "timeout": True},
            "parameter 'timeout' must be of type number, got bool",
        ),
    ],
)
def test_mimocode_agent_rejects_arguments_that_violate_the_advertised_schema(tmp_path, args, problem):
    """Schema `required`/`type` are enforced as FormatError so the model gets
    feedback and the turn is counted in tool_call_errors."""
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_call("c1", "bash", args)]},
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "bash"}], step_limit=4)

    status, _ = agent.run("run it")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    (observation,) = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert observation == f"Invalid arguments for tool 'bash': {problem}"


def test_mimocode_agent_accepts_arguments_that_satisfy_the_schema(tmp_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_call("c1", "bash", {"command": "echo ok", "description": "x", "timeout": 5000})],
            },
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "bash"}], step_limit=4)

    status, _ = agent.run("run it")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False]
    assert any(m["role"] == "tool" and m["content"].startswith("ok\n") for m in agent.messages)


def test_mimocode_tool_descriptions_name_tools_by_their_registered_ids():
    assert "`read` tool" in EditTool({}).description
    assert "`Read`" not in EditTool({}).description


def test_mimocode_bash_description_states_the_sandbox_not_the_host(monkeypatch):
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    from mimoagent.tools.mimocode.prompts import load_bash_prompt

    load_bash_prompt.cache_clear()
    description = BashTool({}).description

    assert "OS: linux, Shell: bash" in description


# ---- metadata is not part of the model-visible observation ----


def test_mimocode_agent_does_not_render_tool_metadata(tmp_path):
    """Every metadata field is constant per rollout, a duplicate of the output,
    or a harness-only signal (bash returncode); none of it reaches the model."""
    assert MimocodeAgentConfig().show_tool_metadata is False
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _call("c1", "bash", {"command": "echo hi", "description": "say hi"}),
                    _call("c2", "bash", {"command": "true", "description": "silent"}),
                    _call("c3", "glob", {"pattern": "*.nothing"}),
                ],
            },
            {"content": "done"},
        ]
    )
    agent = MimocodeAgent(
        model=model, env=LocalEnvironment(cwd=str(tmp_path)), tools=[{"tool": "bash"}, {"tool": "glob"}], step_limit=4
    )

    status, _ = agent.run("probe")

    assert status == "Idle"
    observations = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert observations == ["hi\n", "(no output)", "No files found"]


# ---- exit status stays visible without metadata; budgets cannot wedge a rollout ----


def test_mimocode_bash_marks_non_zero_exit_in_the_body(tmp_path):
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    run = lambda cmd: BashTool({}).execute({"command": cmd, "description": "x"}, context)  # noqa: E731

    assert run("true").output == "(no output)"
    assert run("false").output == "(no output)\n(exit code 1)"
    failed = run("ls /nonexistent_dir_zz")
    assert failed.output.endswith("\n(exit code 2)")
    assert "No such file" in failed.output
    assert run("echo ok").output == "ok\n"


def test_mimocode_bash_one_byte_budget_stays_within_budget(tmp_path):
    """max_bytes=1 with a trailing newline used to compute bytes[-0:] and return the whole line."""
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = BashTool({"max_output_lines": 3, "max_output_bytes": 1}).execute(
        {"command": "printf 'xxxxxxxxxxxxxxxxxxxx\\n'", "description": "x"}, context
    )

    body = result.output[len("...output truncated...\n") :]
    marker, body = body.split("\n", 1)
    assert marker.endswith("omitted here...")
    assert len(body.encode("utf-8")) <= 1


def test_mimocode_read_rejects_non_positive_budgets_and_always_advances(tmp_path):
    with pytest.raises(ValueError):
        ReadTool({"max_output_bytes": 0})
    with pytest.raises(ValueError):
        ReadTool({"max_view_lines": 0})

    target = tmp_path / "wide.txt"
    target.write_text("\n".join("w" * 50 for _ in range(3)) + "\n")
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}
    tool = ReadTool({"max_output_bytes": 10})  # smaller than one line

    first = tool.execute({"file_path": str(target)}, context)
    assert "1: " + "w" * 10 + "... (line truncated to 10 bytes)" in first.output
    assert "w" * 11 not in first.output  # the budget holds even for the first line
    assert "(Output capped at 10 bytes. Showing lines 1-1. Use offset=2 to continue.)" in first.output

    second = tool.execute({"file_path": str(target), "offset": 2}, context)
    assert "2: " + "w" * 10 + "... (line truncated to 10 bytes)" in second.output
    assert "Showing lines 2-2. Use offset=3 to continue." in second.output


def test_mimocode_read_first_line_cut_lands_on_a_utf8_boundary(tmp_path):
    target = tmp_path / "cjk.txt"
    target.write_text("\u4e2d" * 20 + "\n")  # 3 bytes per char
    context = {"env": LocalEnvironment(cwd=str(tmp_path)), "state": {"cwd": str(tmp_path)}}

    result = ReadTool({"max_output_bytes": 10}).execute({"file_path": str(target)}, context)

    assert "1: " + "\u4e2d" * 3 + "... (line truncated to 10 bytes)" in result.output  # 9 bytes, no torn char
    assert "(End of file - total 1 lines)" in result.output
