"""Public-contract tests for the native Codex agent's code mode.

The real Codex implementation exposes ``exec`` as a Responses custom/freeform
tool.  JavaScript passed to it can compose the regular Codex tools without
making those implementation details part of the model conversation.  These
tests drive the public ``CodexAgent`` surface with a scripted model and a local
environment. They need neither a gateway nor a Kubernetes pod; runtime cases
use a locally installed ``codex-code-mode-host`` and skip when it is absent.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from mimoagent.agents.codex.codex_agent import CodexAgent, CodexAgentConfig
from mimoagent.compaction import CODEX_SUMMARY_PREFIX
from mimoagent.environments import TransportError
from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.codex.code_mode_host import find_code_mode_host


class ScriptedModel:
    def __init__(self, responses, *, protocol="responses"):
        self.config = SimpleNamespace(
            model_name="test-codex",
            model_kwargs={},
            protocol=protocol,
        )
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []
        self.seen_kwargs: list[dict] = []

    def query(self, messages, **kwargs):
        self.seen_messages.append([dict(message) for message in messages])
        self.seen_kwargs.append(kwargs)
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response

    def get_template_vars(self):
        return {}


_ACTIVE_AGENTS: list[CodexAgent] = []


@pytest.fixture(autouse=True)
def _close_code_mode_agents():
    yield
    while _ACTIVE_AGENTS:
        _ACTIVE_AGENTS.pop().close()


@pytest.fixture
def code_mode_host_path() -> str:
    try:
        return str(find_code_mode_host())
    except FileNotFoundError:
        pytest.skip("codex-code-mode-host is not installed")


def _tool_call(call_id: str, name: str, arguments: str, *, call_type: str = "function") -> dict:
    return {
        "id": call_id,
        "type": call_type,
        "function": {"name": name, "arguments": arguments},
    }


def _exec_call(call_id: str, source: str) -> dict:
    return _tool_call(call_id, "exec", source, call_type="custom")


def _tool_name(definition: dict) -> str | None:
    if definition.get("type") == "function":
        return (definition.get("function") or {}).get("name") or definition.get("name")
    return definition.get("name")


def _last_tool_text(messages: list[dict]) -> str:
    return next(message["content"] for message in reversed(messages) if message.get("role") == "tool")


def _agent(tmp_path, model, **overrides) -> CodexAgent:
    overrides.setdefault("ptc", True)
    agent = CodexAgent(
        model,
        LocalEnvironment(cwd=str(tmp_path)),
        step_limit=20,
        **overrides,
    )
    _ACTIVE_AGENTS.append(agent)
    return agent


def test_ptc_surface_is_exactly_freeform_exec_and_wait(tmp_path):
    assert CodexAgentConfig().ptc is True
    assert CodexAgentConfig().nested_tool_call_errors is False

    model = ScriptedModel([{"content": "done"}])
    agent = _agent(tmp_path, model)
    status, _ = agent.run("inspect tools")

    assert status == "Idle"
    definitions = model.seen_kwargs[0]["tools"]
    by_name = {_tool_name(definition): definition for definition in definitions}
    assert set(by_name) == {"exec", "wait"}

    exec_definition = by_name["exec"]
    assert exec_definition["type"] == "custom"
    assert exec_definition["format"]["type"] == "grammar"
    assert exec_definition["format"]["syntax"] == "lark"
    assert "pragma_source" in exec_definition["format"]["definition"]
    for contract in (
        "await tools.exec_command",
        "text(value",
        "store(key",
        "load(key",
        "ALL_TOOLS",
    ):
        assert contract in exec_definition["description"]

    wait_schema = by_name["wait"]["function"]["parameters"]
    assert wait_schema["required"] == ["cell_id"]
    assert {"cell_id", "yield_time_ms", "max_tokens", "terminate"} <= wait_schema["properties"].keys()


def test_codex_compaction_is_independent_and_uses_codex_history_shape(tmp_path):
    model = ScriptedModel([{"content": "handoff"}, {"content": "done"}])
    agent = _agent(
        tmp_path,
        model,
        ptc=False,
        compaction_enabled=True,
        compaction_trigger_tokens=1,
    )

    status, result = agent.run("complete the task")

    assert (status, result) == ("Idle", "done")
    assert agent._compact_count == 1
    roles = [message["role"] for message in agent.messages]
    assert roles == ["system", "user", "user", "assistant"]
    assert agent.messages[2]["content"].startswith(CODEX_SUMMARY_PREFIX)
    assert "complete the task" in agent.messages[1]["content"]
    assert "handoff" in agent.messages[2]["content"]
    assert "CONTEXT CHECKPOINT COMPACTION" in model.seen_messages[0][-1]["content"]


def test_codex_observation_is_cut_once_by_the_tool_budget(tmp_path):
    # A direct exec_command result is cut to the model policy (10000 tokens,
    # codex-rs ``to_response_item``); the 48000-character history cut is the
    # 12000-token upstream bound and does not fire again on top of it.
    assert CodexAgentConfig().max_observation_length == 48_000

    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call("big-1", "exec_command", json.dumps({"cmd": "python3 -c \"print('a' * 500000)\""}))
                ],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, ptc=False)

    status, _ = agent.run("emit a large output")

    assert status == "Idle"
    observation = _last_tool_text(agent.messages)
    assert observation.startswith("Warning: truncated output (original token count: 125001)")
    assert observation.count("tokens truncated…") == 1
    assert "[TRUNCATION WARNING]" not in observation
    assert "[...OUTPUT OMITTED" not in observation
    # The 10000-token policy keeps 40000 payload bytes; a second cut on top
    # would have halved that.
    assert 40_000 <= observation.count("a") < 40_100


def test_ptc_rejects_hallucinated_direct_call(tmp_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("direct-1", "exec_command", json.dumps({"cmd": "touch bypass"}))],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model)

    status, _ = agent.run("do not bypass exec")

    assert status == "Idle"
    assert not (tmp_path / "bypass").exists()
    assert any(
        message.get("role") == "tool" and "Available tools: ['exec', 'wait']" in str(message.get("content"))
        for message in agent.messages
    )
    assert agent.tool_call_errors == [True, False]


def test_nested_unknown_tool_is_not_a_tool_call_error_by_default(tmp_path, code_mode_host_path):
    # Default-off: the outer exec turn already reports script failure. Nested
    # contract slips only enter tool_call_errors when the switch is on.
    source = 'const r = await tools.nonexistent_tool({cmd: "printf x"}); text(r.output);'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("call a tool that does not exist from JS")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False]
    assert "nonexistent_tool is not a function" in _last_tool_text(model.seen_messages[1])


def test_nested_unknown_tool_marks_the_turn_as_a_tool_call_error(tmp_path, code_mode_host_path):
    source = 'const r = await tools.nonexistent_tool({cmd: "printf x"}); text(r.output);'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("call a tool that does not exist from JS")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    assert agent._nested_contract_errors == []
    assert "nonexistent_tool is not a function" in _last_tool_text(model.seen_messages[1])


def test_nested_computed_unknown_tool_marks_the_turn_as_a_tool_call_error(tmp_path, code_mode_host_path):
    source = 'const name = "nonexistent_tool"; await tools[name]({});'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("call an unknown tool through a computed property")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    assert agent._nested_contract_errors == []
    assert "tools[name] is not a function" in _last_tool_text(model.seen_messages[1])


def test_nested_known_tool_does_not_mark_the_turn(tmp_path, code_mode_host_path):
    source = 'text((await tools.exec_command({cmd: "printf ok"})).output);'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("call a real nested tool")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False]


def test_nested_bad_arg_type_marks_the_turn(tmp_path, code_mode_host_path):
    source = 'await tools.exec_command("ls -la");'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("pass a string where an object is required")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    assert "expects a JSON object" in _last_tool_text(model.seen_messages[1])


def test_nested_semantic_param_error_does_not_mark_the_turn(tmp_path, code_mode_host_path):
    source = 'await tools.exec_command({command: "ls -la"});'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("use the wrong parameter name")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False]
    assert "unsupported field" in _last_tool_text(model.seen_messages[1])


def test_nested_unknown_tool_can_be_caught_for_recovery(tmp_path, code_mode_host_path):
    source = (
        "try { await tools.nonexistent_tool({}); } catch (_) { text('recovered'); } "
        'text((await tools.exec_command({cmd: "printf still-ok"})).output);'
    )
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("swallow the unknown-tool error and keep going")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False]
    output = _last_tool_text(model.seen_messages[1])
    assert "Script completed" in output
    assert "still-ok" in output


def test_nested_error_surfacing_through_wait_marks_the_wait_turn(tmp_path, code_mode_host_path):
    # ``wait`` is the second way a script result reaches _format_response: the
    # cell yields on one turn and only fails on a later one.
    source = (
        'text("pre"); yield_control(); await new Promise(r => setTimeout(r, 20)); await tools.nonexistent_tool({});'
    )

    def wait_for_cell(messages):
        match = re.search(r"Script running with cell ID ([^\s]+)", _last_tool_text(messages))
        assert match, _last_tool_text(messages)
        return {
            "content": "",
            "tool_calls": [
                _tool_call("wait-1", "wait", json.dumps({"cell_id": match.group(1), "yield_time_ms": 2000}))
            ],
        }

    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            wait_for_cell,
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("yield, then fail on the wait turn")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, True, False]
    assert "nonexistent_tool is not a function" in _last_tool_text(model.seen_messages[2])


def test_callers_own_object_is_not_read_as_the_nested_surface(tmp_path, code_mode_host_path):
    # Same class as the semantic-param case above: the script fails, but the
    # failure is ordinary JS on the caller's own object, not a tool contract
    # error. `mytools` only shares a suffix; `ctx.tools` is the caller's own map.
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", "const mytools = {}; await mytools.foo();")]},
            {"content": "", "tool_calls": [_exec_call("exec-2", "const ctx = {tools: {}}; await ctx.tools.foo();")]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("call methods on objects the caller made")

    assert status == "Idle"
    assert agent.tool_call_errors == [False, False, False]
    assert "mytools.foo is not a function" in _last_tool_text(model.seen_messages[1])
    assert "ctx.tools.foo is not a function" in _last_tool_text(model.seen_messages[2])


def test_nested_error_is_counted_even_when_the_output_is_truncated(tmp_path, code_mode_host_path):
    # The scan reads error_text, so a zero budget hides the message from the
    # model without hiding the contract error from tool_call_errors.
    source = '// @exec: {"max_output_tokens": 0}\ntext("x".repeat(5000)); await tools.nonexistent_tool({});'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("bury the error under a truncated output")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    output = _last_tool_text(model.seen_messages[1])
    assert "truncated" in output
    assert "nonexistent_tool" not in output


def test_parallel_exec_calls_mark_the_turn_once_and_drain(tmp_path, code_mode_host_path):
    # Two failing cells run on worker threads and land in one turn; the next
    # turn must start from an empty ledger.
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _exec_call("exec-1", "await tools.nope_one({});"),
                    _exec_call("exec-2", "await tools.nope_two({});"),
                ],
            },
            {
                "content": "",
                "tool_calls": [_exec_call("exec-3", 'text((await tools.exec_command({cmd: "printf ok"})).output);')],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("two failing cells in one turn, then a clean one")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False, False]
    assert agent._nested_contract_errors == []
    assert "Script completed" in _last_tool_text(model.seen_messages[2])


def test_a_rethrown_contract_error_still_counts(tmp_path, code_mode_host_path):
    # Only the message survives a rethrow, so a wrapper that re-raises the real
    # failure is counted — and so is a message the model writes by hand. That
    # penalises nobody but the author, and keeping it means wrappers stay honest.
    source = "try { await tools.nonexistent_tool({}); } catch (e) { throw new Error('wrapped: ' + e.message); }"
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("catch and rethrow the contract error")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    assert "wrapped: " in _last_tool_text(model.seen_messages[1])


def test_bad_args_covers_the_freeform_string_contract(tmp_path, code_mode_host_path):
    # The bad-args pattern has two branches because the host words the freeform
    # and JSON-schema contracts differently. exec_command covers "JSON object
    # for arguments"; apply_patch is the only tool on the "string input" side.
    source = "await tools.apply_patch({patch: 1});"
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, nested_tool_call_errors=True)

    status, _ = agent.run("pass an object to a freeform tool")

    assert status == "Idle"
    assert agent.tool_call_errors == [True, False]
    assert "expects a string input" in _last_tool_text(model.seen_messages[1])


def test_disabling_code_mode_restores_direct_tools_only(tmp_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("direct-1", "exec_command", json.dumps({"cmd": "printf direct-ok"}))],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, ptc=False)

    status, _ = agent.run("use direct tools")

    names = {_tool_name(definition) for definition in model.seen_kwargs[0]["tools"]}
    assert names == {"exec_command", "apply_patch"}
    assert status == "Idle"
    assert "direct-ok" in _last_tool_text(model.seen_messages[1])
    assert "Process exited" not in _last_tool_text(model.seen_messages[1])


def test_direct_exec_command_keeps_the_exit_code_visible_without_metadata(tmp_path):
    """Metadata is not rendered; a silent non-zero exit must still differ from a silent success."""
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_tool_call("direct-1", "exec_command", json.dumps({"cmd": "false"}))]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, ptc=False)

    status, _ = agent.run("run a silently failing command")

    assert status == "Idle"
    text = _last_tool_text(model.seen_messages[1])
    assert "Process exited with code 1" in text
    assert "Tool metadata" not in text


def test_codex_agent_does_not_render_tool_metadata():
    assert CodexAgentConfig().show_tool_metadata is False


def test_direct_exec_command_request_above_the_model_policy_is_cut_to_the_policy(tmp_path):
    """codex-rs ``model_output_policy``: a direct call's request only lowers the
    budget; above the policy the policy wins. The description says so in
    upstream's words and never states the number."""

    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "big-1",
                        "exec_command",
                        json.dumps({"cmd": "python3 -c \"print('a' * 500000)\"", "max_output_tokens": 50_000}),
                    )
                ],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, ptc=False)

    status, _ = agent.run("emit a large output with a raised budget")

    assert status == "Idle"
    observation = _last_tool_text(agent.messages)
    assert "tokens truncated…" in observation
    assert 40_000 <= observation.count("a") < 40_100
    exec_command_def = next(d for d in model.seen_kwargs[0]["tools"] if _tool_name(d) == "exec_command")
    parameters = exec_command_def.get("function", exec_command_def)["parameters"]
    doc = parameters["properties"]["max_output_tokens"]["description"]
    assert doc == "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."


def test_the_history_cut_follows_the_agent_config(tmp_path):
    """Layer 4: ``max_observation_length`` is the hard bound, applied by the
    base class when the observation is formatted, on top of the tool's own cut."""

    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "big-1",
                        "exec_command",
                        json.dumps({"cmd": "python3 -c \"print('a' * 500000)\"", "max_output_tokens": 50_000}),
                    )
                ],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, ptc=False, max_observation_length=4_000)

    status, _ = agent.run("emit a large output under a small limit")

    assert status == "Idle"
    observation = _last_tool_text(agent.messages)
    assert 3_900 <= observation.count("a") < 4_100
    assert observation.count("Warning: truncated output") == 1
    assert "[TRUNCATION WARNING]" in observation


def test_ptc_nested_exec_command_budget_is_honoured_and_the_cell_cut_follows(tmp_path, code_mode_host_path):
    """Layer 2 honours the nested request as given (50000 tokens -> 200000
    bytes reach JavaScript); layer 3 then cuts what the cell ``text()``s to the
    pragma default of 10000 tokens."""

    source = """
const r = await tools.exec_command({cmd: "python3 -c \\"print('a' * 500000)\\"", max_output_tokens: 50000});
text("nested_length=" + r.output.length);
text(r.output);
"""
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("measure then dump a large nested output")

    assert status == "Idle"
    output = _last_tool_text(model.seen_messages[1])
    assert "Script completed" in output
    nested_length = int(re.search(r"nested_length=(\d+)", output).group(1))
    assert 200_000 <= nested_length < 200_200
    assert "tokens truncated…" in output
    assert 39_800 <= output.count("a") < 40_100
    exec_def = next(d for d in model.seen_kwargs[0]["tools"] if _tool_name(d) == "exec")
    assert "Values above" not in exec_def["description"]  # the old cap sentence


def test_ptc_cell_budget_above_the_old_cap_is_honoured_up_to_the_history_limit(tmp_path, code_mode_host_path):
    """The pragma budget has no ceiling of its own; only the observation cut
    bounds it. With that limit raised the whole 200000-byte cell result reaches
    the model; at the default 48000 characters it is cut once more."""

    source = """// @exec: {"max_output_tokens": 50000}
const r = await tools.exec_command({cmd: "python3 -c \\"print('a' * 500000)\\""});
text(r.output);
"""
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, max_observation_length=1_000_000)

    status, _ = agent.run("dump a large output under a raised cell budget")

    assert status == "Idle"
    output = _last_tool_text(model.seen_messages[1])
    assert 200_000 <= output.count("a") < 200_100
    assert output.count("tokens truncated…") == 1

    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    assert agent.run("dump a large output under the default history limit")[0] == "Idle"
    output = _last_tool_text(model.seen_messages[1])
    assert 47_000 <= output.count("a") < 48_100
    assert output.count("Warning: truncated output") == 1
    assert "[TRUNCATION WARNING]" in output


def test_non_responses_protocol_is_rejected(tmp_path):
    model = ScriptedModel([], protocol="chat")
    with pytest.raises(ValueError, match="CodexAgent requires protocol='responses'"):
        _agent(tmp_path, model)


def test_exec_can_compose_exec_command_and_lists_nested_tools(tmp_path, code_mode_host_path):
    source = """
const result = await tools.exec_command({cmd: "printf nested-ok"});
text(result.output);
text(JSON.stringify(ALL_TOOLS.map(({name}) => name).sort()));
"""
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, result = agent.run("compose tools")

    assert (status, result) == ("Idle", "done")
    output = _last_tool_text(model.seen_messages[1])
    assert "Script completed" in output
    assert "nested-ok" in output
    # Default catalogue is exactly exec_command + apply_patch; ptc must not
    # silently inject view_image / update_plan.
    assert '["apply_patch","exec_command"]' in output.replace(" ", "")
    assert "view_image" not in output
    assert "update_plan" not in output


def test_ptc_true_nested_surface_follows_configured_tools(tmp_path):
    model = ScriptedModel([{"content": "done"}])
    agent = _agent(tmp_path, model, tools=[{"tool": "exec_command"}])

    status, _ = agent.run("inspect nested tools")
    assert status == "Idle"

    exec_definition = next(
        definition for definition in model.seen_kwargs[0]["tools"] if _tool_name(definition) == "exec"
    )
    description = exec_definition["description"]
    assert "### `exec_command`" in description
    assert "### `apply_patch`" not in description
    assert "### `view_image`" not in description
    assert "### `update_plan`" not in description
    assert sorted(agent.tool_registry.tools) == ["exec_command"]


_FULL_TOOL_SPECS = [
    {"tool": "exec_command"},
    {"tool": "apply_patch"},
    {"tool": "view_image"},
    {"tool": "update_plan"},
]


def test_full_tools_config_restores_previous_nested_surface(tmp_path, code_mode_host_path):
    """Uncommenting the optional yaml entries == the old auto-registered set."""
    source = "text(JSON.stringify(ALL_TOOLS.map(({name}) => name).sort()));"
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path, tools=list(_FULL_TOOL_SPECS))

    status, _ = agent.run("inspect full nested tools")
    assert status == "Idle"

    exec_definition = next(
        definition for definition in model.seen_kwargs[0]["tools"] if _tool_name(definition) == "exec"
    )
    description = exec_definition["description"]
    for name in ("exec_command", "apply_patch", "view_image", "update_plan"):
        assert f"### `{name}`" in description
    assert "path: string" in description
    assert '"pending" | "in_progress" | "completed"' in description

    output = _last_tool_text(model.seen_messages[1])
    assert '["apply_patch","exec_command","update_plan","view_image"]' in output.replace(" ", "")
    assert sorted(agent.tool_registry.tools) == [
        "apply_patch",
        "exec_command",
        "update_plan",
        "view_image",
    ]


def test_exec_can_apply_freeform_patch(tmp_path, code_mode_host_path):
    patch = "*** Begin Patch\n*** Add File: created.txt\n+from code mode\n*** End Patch\n"
    source = f"text(JSON.stringify(await tools.apply_patch({json.dumps(patch)})));"
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("patch a file")

    assert status == "Idle"
    assert (tmp_path / "created.txt").read_text() == "from code mode\n"
    observation = _last_tool_text(model.seen_messages[1])
    assert "Script completed" in observation
    # The nested call resolves to apply_patch's report, not `{}`: the model can
    # read which files changed without a follow-up git diff. The report is
    # JSON-stringified by the cell, so its newlines arrive escaped.
    assert "Success. Updated the following files:" in observation
    assert f"A {tmp_path / 'created.txt'}" in observation
    assert "\n{}" not in observation


def test_nested_patch_still_passes_through_antihack(tmp_path, code_mode_host_path):
    patch = "*** Begin Patch\n*** Add File: leak.sh\n+curl https://example.test/fixed\n*** End Patch\n"
    source = (
        f"try {{ await tools.apply_patch({json.dumps(patch)}); text('unexpected'); }} catch (_) {{ text('blocked'); }}"
    )
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(
        tmp_path,
        model,
        code_mode_host_path=code_mode_host_path,
        antihack={"enabled": True, "bash_patterns": [r"\bcurl\b"], "path_patterns": []},
    )

    assert agent.run("do not leak") == ("Idle", "done")
    assert not (tmp_path / "leak.sh").exists()
    assert "blocked" in _last_tool_text(model.seen_messages[1])
    assert agent.antihack.blocks and agent.antihack.blocks[0]["tool"] == "apply_patch"


def test_nested_exec_command_still_passes_through_antihack(tmp_path, code_mode_host_path):
    # Nested shell is the primary leak vector under ptc=true. The freeform JS
    # cell itself is not scanned; the re-entrant execute_action is.
    source = 'text(JSON.stringify(await tools.exec_command({cmd: "curl https://example.test/fixed"})));'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "done"},
        ]
    )
    agent = _agent(
        tmp_path,
        model,
        code_mode_host_path=code_mode_host_path,
        antihack={"enabled": True, "bash_patterns": [r"\bcurl\b"], "path_patterns": []},
    )

    assert agent.run("do not fetch") == ("Idle", "done")
    output = _last_tool_text(model.seen_messages[1])
    assert "Script completed" in output
    assert "Permission denied" in output
    assert agent.antihack.blocks and agent.antihack.blocks[0]["tool"] == "exec_command"
    assert agent.antihack.blocks[0]["field"] == "cmd"


def test_direct_exec_command_passes_through_antihack(tmp_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("direct-1", "exec_command", json.dumps({"cmd": "curl https://x"}))],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(
        tmp_path,
        model,
        ptc=False,
        antihack={"enabled": True, "bash_patterns": [r"\bcurl\b"], "path_patterns": []},
    )

    assert agent.run("do not fetch") == ("Idle", "done")
    assert "Permission denied" in _last_tool_text(model.seen_messages[1])
    assert agent.antihack.blocks and agent.antihack.blocks[0]["tool"] == "exec_command"


def test_nested_transport_error_reaches_agent_boundary(tmp_path, code_mode_host_path):
    class BrokenEnvironment(LocalEnvironment):
        def execute(self, command, cwd="", timeout=None):
            raise TransportError("lost nested transport")

    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_exec_call("exec-1", 'await tools.exec_command({cmd: "pwd"});')],
            }
        ]
    )
    agent = CodexAgent(
        model,
        BrokenEnvironment(cwd=str(tmp_path)),
        ptc=True,
        code_mode_host_path=code_mode_host_path,
        step_limit=5,
    )
    _ACTIVE_AGENTS.append(agent)

    status, result = agent.run("surface transport failure")

    assert status == "InfraError"
    assert "lost nested transport" in result


def test_exec_store_and_load_persist_across_fresh_cells(tmp_path, code_mode_host_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    _exec_call(
                        "exec-1",
                        'store("note", {title: "kept", values: [1, true, null]}); text("stored");',
                    )
                ],
            },
            {
                "content": "",
                "tool_calls": [_exec_call("exec-2", 'text(JSON.stringify(load("note")));')],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, _ = agent.run("remember a value")

    assert status == "Idle"
    assert "stored" in _last_tool_text(model.seen_messages[1])
    loaded = _last_tool_text(model.seen_messages[2])
    assert '"title":"kept"' in loaded
    assert '"values":[1,true,null]' in loaded


def test_exec_preserves_output_and_reports_script_failure(tmp_path, code_mode_host_path):
    source = 'text("before crash"); throw new Error("boom");'
    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            {"content": "handled"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, result = agent.run("surface failure")

    assert (status, result) == ("Idle", "handled")
    output = _last_tool_text(model.seen_messages[1])
    assert "Script failed" in output
    assert "before crash" in output
    assert "boom" in output


def test_exec_forwards_image_output_as_media(tmp_path, code_mode_host_path):
    image_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    def inspect_media(messages):
        content = next(message["content"] for message in reversed(messages) if message.get("role") == "tool")
        assert content[0]["type"] == "text"
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }
        return {"content": "done"}

    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", f"image({json.dumps(image_url)});")]},
            inspect_media,
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    assert agent.run("return an image") == ("Idle", "done")


def test_notify_is_a_separate_output_for_the_original_exec_call(tmp_path, code_mode_host_path):
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [_exec_call("exec-1", 'notify("progress"); text("final");')],
            },
            {"content": "done"},
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    assert agent.run("notify me") == ("Idle", "done")
    outputs = [message for message in agent.messages if message.get("role") == "tool"]
    assert len(outputs) == 2
    assert outputs[0]["tool_call_id"] == outputs[1]["tool_call_id"] == "exec-1"
    assert outputs[0]["content"] == "progress"
    assert "Script completed" in outputs[1]["content"]
    assert "final" in outputs[1]["content"]


def test_chat_protocol_is_rejected_even_when_ptc_is_disabled(tmp_path):
    model = ScriptedModel([{"content": "done"}], protocol="chat")
    with pytest.raises(ValueError, match="CodexAgent requires protocol='responses'"):
        _agent(tmp_path, model, ptc=False)


def test_yielded_exec_can_be_resumed_with_wait(tmp_path, code_mode_host_path):
    source = """
text("before-yield");
yield_control();
await new Promise(resolve => setTimeout(() => { text("after-yield"); resolve(); }, 50));
"""

    def wait_for_cell(messages):
        output = _last_tool_text(messages)
        match = re.search(r"Script running with cell ID ([^\s]+)", output)
        assert match, output
        return {
            "content": "",
            "tool_calls": [
                _tool_call(
                    "wait-1",
                    "wait",
                    json.dumps({"cell_id": match.group(1), "yield_time_ms": 1000}),
                )
            ],
        }

    def finish_after_wait(messages):
        output = _last_tool_text(messages)
        assert "Script completed" in output
        assert "after-yield" in output
        return {"content": "done"}

    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            wait_for_cell,
            finish_after_wait,
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    status, result = agent.run("yield and wait")

    assert (status, result) == ("Idle", "done")
    assert "before-yield" in _last_tool_text(model.seen_messages[1])


def test_zero_output_budget_keeps_yielded_cell_id(tmp_path, code_mode_host_path):
    source = '// @exec: {"max_output_tokens": 0}\ntext("hidden"); yield_control(); await new Promise(() => {});'

    def verify_header(messages):
        output = _last_tool_text(messages)
        assert re.search(r"Script running with cell ID \S+", output), output
        assert "hidden" not in output
        return {"content": "done"}

    model = ScriptedModel(
        [
            {"content": "", "tool_calls": [_exec_call("exec-1", source)]},
            verify_header,
        ]
    )
    agent = _agent(tmp_path, model, code_mode_host_path=code_mode_host_path)

    assert agent.run("yield with no output budget") == ("Idle", "done")


# --- The native Codex agent ships a distilled core system prompt as its
# default, rather than relying on a hand-written yaml template.


def test_codex_agent_default_system_prompt_is_the_packaged_core():
    from pathlib import Path

    import mimoagent.agents.codex.codex_agent as codex_module

    packaged = (Path(codex_module.__file__).parent / "prompts" / "codex_core.txt").read_text(encoding="utf-8").rstrip()

    # Default equals the packaged resource verbatim — no drift between a copy in
    # code and the file that ships.
    assert CodexAgentConfig().system_template == packaged
    assert packaged  # non-empty


def test_codex_core_prompt_keeps_behaviour_rules_and_drops_product_prose():
    prompt = CodexAgentConfig().system_template

    # Behaviour-shaping content that must survive the distillation.
    assert "apply_patch" in prompt
    assert "rg --files" in prompt
    assert "git reset --hard" in prompt  # destructive-command guard
    # scope-of-action rules survive (verbatim wording from the upstream section).
    assert "Diagnose:" in prompt
    assert "Change or build:" in prompt

    # Product / CLI prose that headless RL must not carry.
    for dropped in ("Personality", "commentary channel", "Visualization", "SKILL.md", "final channel"):
        assert dropped not in prompt


def test_codex_core_prompt_drops_user_midturn_and_compaction_contracts():
    """RL does not train these; promising them misleads the model.

    - Mid-turn user messages → models wait on / request a user reply.
    - Auto-compaction / "time never runs out" → longer rollouts that then hit
      hard truncation when compaction is off (the default).
    """
    prompt = CodexAgentConfig().system_template

    for dropped in (
        "Working with the user",
        "The user may send a new message",
        "automatically summarized",
        "compaction occurred",
        "time never runs out",
        "spanning compactions",
    ):
        assert dropped not in prompt


def test_codex_system_prompt_states_the_environment_cwd(tmp_path):
    """codex-rs sends `<environment_context><cwd>`; without an equivalent, 64%
    of ptc trajectories guessed `/workspace` and hit `cd: No such file`."""
    model = ScriptedModel([{"content": "done"}])
    agent = _agent(tmp_path, model)

    assert agent.run("where am I") == ("Idle", "done")

    system = next(message["content"] for message in model.seen_messages[0] if message["role"] == "system")
    assert f"Your working directory is `{tmp_path}`." in system
    assert "{{cwd}}" not in system
    # The tool docs point at that statement rather than an undefined "turn cwd".
    exec_description = model.seen_kwargs[0]["tools"][0]["description"]
    assert "turn cwd" not in exec_description
    assert "working directory stated in your instructions" in exec_description


def test_stray_tool_call_text_is_not_rescued(tmp_path):
    """Whitebox does not parse <tool_call> from assistant text (unlike CCAgent).

    A structured tool-call never arrived → Idle. The model must learn the real
    tool surface instead of relying on a CC-style FormatError rescue.
    """
    stray = '<tool_call>{"name": "exec_command", "arguments": {"cmd": "ls"}}</tool_call>'
    model = ScriptedModel([{"content": stray}])
    agent = _agent(tmp_path, model, ptc=False)

    status, result = agent.run("do not emit tool-call text")

    assert status == "Idle"
    assert result == stray
    assert agent.tool_call_errors == [False]
