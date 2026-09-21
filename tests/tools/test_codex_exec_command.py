"""Contract tests for the Codex ``exec_command`` tool.

These pin the ways the tool's model-facing contract used to be wrong: it
advertised a PTY and a session ID it never provides, it stated a 30s ceiling
unrelated to the configured one, it accepted a ``max_output_tokens`` budget it
never applied (and later applied it in the wrong layer, under a cap the model
was told about), it silently raised any deadline below the configured default,
and for one run it advertised the deadline as ``timeout_ms`` while the trained
checkpoint kept sending ``yield_time_ms`` (silently ignored). The deadline now
keeps upstream's ``yield_time_ms`` name, is honoured as given, and the
description states that nothing can be resumed. Each test asserts the real
behaviour rather than only the absence of the old wording, so none of them can
pass on an empty description.
"""

from __future__ import annotations

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.base import ToolException
from mimoagent.tools.codex.exec_command import ExecCommandTool
from mimoagent.tools.codex.output_budget import DEFAULT_MAX_OUTPUT_TOKENS, UNIFIED_EXEC_OUTPUT_MAX_BYTES


class _FixedOutputEnvironment:
    """Returns a canned payload so budget behaviour is isolated from the shell."""

    def __init__(self, output: str, *, returncode: int = 0, reason: str = "ok"):
        self.output = output
        self.returncode = returncode
        self.reason = reason
        self.config = type("_Config", (), {"cwd": "/testbed"})()
        self.timeouts: list[int | None] = []

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict:
        self.timeouts.append(timeout)
        return {"output": self.output, "returncode": self.returncode, "reason": self.reason}


def _run(params: dict, *, env) -> object:
    return ExecCommandTool().execute(params, {"env": env})


def _effective_timeout(params: dict, *, config: dict | None = None) -> int | None:
    """The seconds the tool actually asked the environment to wait."""

    env = _FixedOutputEnvironment("")
    ExecCommandTool(config or {}).execute({"cmd": "noop", **params}, {"env": env})
    return env.timeouts[-1]


def test_description_states_run_to_completion_and_claims_no_pty_session():
    description = ExecCommandTool({"timeout": 30, "max_timeout": 300}).description

    # Positive first: the description must actually describe what happens, so
    # the absence checks below cannot be satisfied by an empty string.
    assert "Runs a shell command to completion" in description
    assert "no session to resume" in description.lower()
    # The cell-level pragma shares the parameter name and IS resumable; the
    # description has to draw that line itself.
    assert "`// @exec:` pragma of the same name" in description

    assert "PTY" not in description
    assert "session ID" not in description


@pytest.mark.parametrize(
    ("timeout", "max_timeout"),
    [(30, 300), (60, 120)],
)
def test_timeout_text_is_rendered_from_config(timeout, max_timeout):
    """Structural, not lexical: hardcoded numbers fail one of the two cases."""

    tool = ExecCommandTool({"timeout": timeout, "max_timeout": max_timeout})
    properties = tool.get_function_parameters()["properties"]
    timeout_doc = properties["yield_time_ms"]["description"]

    assert f"{timeout * 1000}ms" in timeout_doc
    assert f"capped at {max_timeout * 1000}ms" in timeout_doc
    # The upstream figures describe unified exec, which this tool is not.
    assert "250-30000" not in timeout_doc
    assert f"{timeout * 1000}ms" in tool.description


def test_the_deadline_keeps_upstreams_yield_name_and_says_it_cannot_resume():
    """The checkpoint was trained on ``yield_time_ms``; the one-shot rename to
    ``timeout_ms`` (cfef3f1d) left 56–61% of calls carrying a silently ignored
    field and seeded a ``timeout_time_ms`` blend. The name stays, and the
    parameter text carries the one-shot meaning."""

    properties = ExecCommandTool().get_function_parameters()["properties"]

    assert "yield_time_ms" in properties
    assert "timeout_ms" not in properties
    assert "cannot be resumed" in properties["yield_time_ms"]["description"]


def test_a_timeout_below_the_configured_default_is_honoured():
    """The old ``max(requested, default)`` floor mapped every value under the
    default onto it, so a model asking for a short probe waited the full
    default instead."""

    assert _effective_timeout({"yield_time_ms": 5_000}, config={"timeout": 60, "max_timeout": 300}) == 5


@pytest.mark.parametrize(
    ("yield_time_ms", "expected"),
    [(1, 1), (0, 1), (1_500, 1), (10_000, 10), (120_000, 120), (300_000, 300), (999_000, 300)],
)
def test_the_requested_deadline_is_honoured_as_given_and_clamped_only_at_the_ends(yield_time_ms, expected):
    """Upstream's ``yield_time_ms: 10000`` means "preview after 10s, keep
    running"; here it is a 10s deadline, by decision, so the model's value is
    the value that is used."""

    assert _effective_timeout({"yield_time_ms": yield_time_ms}, config={"timeout": 60, "max_timeout": 300}) == expected


@pytest.mark.parametrize("bad", [None, "soon", True, float("nan")])
def test_an_unusable_timeout_falls_back_to_the_configured_default(bad):
    assert _effective_timeout({"yield_time_ms": bad}, config={"timeout": 60, "max_timeout": 300}) == 60


@pytest.mark.parametrize(
    "params",
    [{"timeout_ms": 5_000}, {"timeout_ms": 5_000, "yield_time_ms": 10_000}, {"timeout_time_ms": 1_000}],
    ids=["timeout_ms", "timeout_ms-next-to-yield", "blend"],
)
def test_timeout_ms_and_other_unknown_fields_are_rejected_before_anything_runs(params):
    """``timeout_ms`` was the advertised name for one run and 18–25% of calls
    picked it up; accepting it silently would keep two names alive. The generic
    unsupported-field error names the offending field, and nothing executes."""

    env = _FixedOutputEnvironment("")

    with pytest.raises(ToolException, match=r"unsupported field\(s\)") as excinfo:
        ExecCommandTool({"timeout": 60, "max_timeout": 300}).execute({"cmd": "noop", **params}, {"env": env})

    offending = [k for k in params if k != "yield_time_ms"]
    assert all(k in str(excinfo.value) for k in offending)
    assert env.timeouts == []


@pytest.mark.parametrize("reason", ["pod_timeout", "client_timeout"])
def test_a_timed_out_command_returns_a_result_instead_of_raising(reason):
    """A raised timeout became a rejected promise in code mode, so one slow
    command discarded every sibling result in the same ``Promise.all``."""

    env = _FixedOutputEnvironment("partial-output-here", returncode=124, reason=reason)

    result = ExecCommandTool({"timeout": 60, "max_timeout": 300}).execute({"cmd": "sleep 999"}, {"env": env})

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert result.metadata["reason"] == reason
    assert "partial-output-here" in result.output
    assert "timed out" in result.output
    # The model needs to know the knob and its ceiling to recover on its own.
    assert "yield_time_ms" in result.output
    assert "300000" in result.output


def test_a_timed_out_command_reports_the_deadline_it_actually_waited_for():
    env = _FixedOutputEnvironment("", returncode=124, reason="pod_timeout")

    result = ExecCommandTool({"timeout": 60, "max_timeout": 300}).execute(
        {"cmd": "sleep 999", "yield_time_ms": 5_000}, {"env": env}
    )

    assert "5s" in result.output


def test_a_timed_out_result_carries_the_requested_budget_instead_of_applying_it():
    """Layer 2 belongs to the caller: the tool hands back the collected text and
    the budget the model asked for; the runtime (nested) or agent (direct)
    applies it. Nothing is cut here, so a 1 MB partial output survives intact."""

    env = _FixedOutputEnvironment("X" * 1_000_000, returncode=124, reason="pod_timeout")

    result = ExecCommandTool().execute({"cmd": "noop", "max_output_tokens": 10}, {"env": env})

    assert result.metadata["timed_out"] is True
    assert result.metadata["max_output_tokens"] == 10
    assert result.metadata["output_omitted_bytes"] == 0
    assert result.output.count("X") == 1_000_000
    assert "tokens truncated" not in result.output


def test_the_requested_budget_is_recorded_not_applied():
    env = _FixedOutputEnvironment("X" * 100_000)

    result = _run({"cmd": "noop", "max_output_tokens": 10}, env=env)

    assert result.success
    assert result.output == "X" * 100_000
    assert result.metadata["max_output_tokens"] == 10


def test_without_a_request_the_budget_is_absent_so_javascript_gets_the_raw_text():
    """codex-rs ``code_mode_result``: ``max_output_tokens: None`` means the raw
    collected output goes to JavaScript untouched."""

    env = _FixedOutputEnvironment("X" * 1_000_000)

    result = _run({"cmd": "noop"}, env=env)

    assert result.metadata["max_output_tokens"] is None
    assert result.output == "X" * 1_000_000


@pytest.mark.parametrize("bad", [-1, "many", True, float("inf")])
def test_an_unusable_budget_resolves_to_the_default_instead_of_failing(bad):
    env = _FixedOutputEnvironment("X" * 100)

    result = _run({"cmd": "noop", "max_output_tokens": bad}, env=env)

    assert result.metadata["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_collection_keeps_one_mebibyte_head_and_tail_with_upstreams_omission_marker():
    """Layer 1: ``HeadTailBuffer`` at 1 MiB, half head, half tail, joined by
    ``... N bytes omitted ...``. This is the only bound the tool itself applies."""

    payload = "H" * 700_000 + "M" * 700_000 + "T" * 700_000
    env = _FixedOutputEnvironment(payload)

    result = _run({"cmd": "noop"}, env=env)

    omitted = len(payload) - UNIFIED_EXEC_OUTPUT_MAX_BYTES
    assert result.metadata["output_omitted_bytes"] == omitted
    assert result.output.startswith("H" * 100)
    assert result.output.endswith("T" * 100)
    assert f"\n... {omitted} bytes omitted ...\n" in result.output
    assert len(result.output.encode()) == UNIFIED_EXEC_OUTPUT_MAX_BYTES + len(f"\n... {omitted} bytes omitted ...\n")


def test_the_budget_description_is_upstreams_sentence_with_no_cap():
    """The model is told the default and that policy may cap larger requests
    (shell_spec.rs wording); the cap itself is never stated, as upstream."""

    doc = ExecCommandTool().get_function_parameters()["properties"]["max_output_tokens"]["description"]

    assert (
        doc
        == f"Output token budget. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens; larger requests may be capped by policy."
    )


def test_the_tool_no_longer_takes_an_output_cap():
    with pytest.raises(TypeError):
        ExecCommandTool({"max_output_tokens_cap": 10})


def test_exit_code_and_wall_time_are_reported_next_to_the_budget():
    env = _FixedOutputEnvironment("X" * 100_000, returncode=2)

    result = _run({"cmd": "false", "max_output_tokens": 5}, env=env)

    assert result.success is False
    assert result.metadata["exit_code"] == 2
    assert "wall_time_seconds" in result.metadata
    assert result.metadata["max_output_tokens"] == 5


def test_real_command_output_is_unaffected_when_small(tmp_path):
    result = _run({"cmd": "printf exec-ok"}, env=LocalEnvironment(cwd=str(tmp_path)))

    assert result.success
    assert "exec-ok" in result.output
