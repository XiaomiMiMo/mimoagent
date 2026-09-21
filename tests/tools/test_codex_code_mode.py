from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from mimoagent.tools.base import ToolException
from mimoagent.tools.codex.code_mode import CodeModeRuntime
from mimoagent.tools.codex.output_budget import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    collect_head_tail,
    formatted_truncate_text,
    truncate_exec_output,
    truncate_middle_tokens,
)


class _Registry:
    tools = {}

    def get_function_definitions(self):
        return []


class _Agent:
    tool_registry = _Registry()


class _Session:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _runtime() -> CodeModeRuntime:
    agent = _Agent()
    runtime = CodeModeRuntime(agent, host_path=None)
    runtime._test_agent = agent
    return runtime


def test_output_budget_never_truncates_status_header_or_cell_id():
    runtime = _runtime()

    output = runtime._format_response(
        {"Yielded": {"cell_id": "cell-7", "content_items": [{"type": "input_text", "text": "hidden"}]}},
        0,
        time.monotonic(),
    )

    assert output.output.startswith("Script running with cell ID cell-7\nWall time")
    assert "hidden" not in output.output


def test_token_budget_uses_utf8_bytes_and_keeps_both_ends():
    truncated = truncate_middle_tokens("甲乙丙丁戊", 2)

    assert "tokens truncated" in truncated
    assert truncated.startswith("甲")
    assert truncated.endswith("戊")


def _result(text: str) -> dict:
    return {"Result": {"cell_id": "cell-1", "content_items": [{"type": "input_text", "text": text}]}}


def test_the_cell_budget_is_the_pragmas_and_has_no_ceiling():
    """Layer 3, codex-rs ``truncate_code_mode_result``: ``resolve_max_tokens``
    is the pragma value or 10000, nothing caps it. The old runtime-level cap
    (and the "Values above N are capped" sentence) is gone; the hard bound is
    the agent's history cut."""

    runtime = _runtime()

    honoured = runtime._format_response(_result("x" * 400_000), 50_000, time.monotonic())
    assert honoured.output.count("x") == 200_000
    assert "Warning: truncated output (original token count: 100000)" in honoured.output

    default = runtime._format_response(_result("x" * 400_000), DEFAULT_MAX_OUTPUT_TOKENS, time.monotonic())
    assert default.output.count("x") == 40_000


def test_exec_and_wait_definitions_state_only_the_defaults():
    exec_def, wait_def = _runtime().model_definitions()

    assert (
        f"`max_output_tokens` sets the token budget for direct `exec` results. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens."
        in exec_def["description"]
    )
    wait_doc = wait_def["function"]["parameters"]["properties"]["max_tokens"]["description"]
    assert f"Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens" in wait_doc
    assert "Values above" not in exec_def["description"]  # the old cap sentence
    assert "Values above" not in wait_doc


def test_failed_session_cleanup_is_identity_compare_and_swap():
    runtime = _runtime()
    old_session = _Session()
    new_session = _Session()
    runtime._session = new_session

    runtime._discard_failed_session(old_session)

    assert runtime._session is new_session
    assert old_session.closed
    assert not new_session.closed
    runtime.close()


def test_close_permanently_prevents_session_reopen():
    runtime = _runtime()
    runtime.close()

    with pytest.raises(ToolException, match="runtime is closed"):
        runtime._get_session()


def test_close_racing_session_creation_reaps_the_created_host(monkeypatch):
    creation_started = threading.Event()
    allow_creation = threading.Event()
    created = []

    class FakeHostSession(_Session):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            creation_started.set()
            assert allow_creation.wait(timeout=2)
            created.append(self)

    monkeypatch.setattr("mimoagent.tools.codex.code_mode.find_code_mode_host", lambda host_path=None: "/fake/host")
    monkeypatch.setattr("mimoagent.tools.codex.code_mode.CodeModeHostSession", FakeHostSession)
    runtime = _runtime()
    result = SimpleNamespace(error=None)

    def create_session():
        try:
            runtime._get_session()
        except Exception as error:
            result.error = error

    creator = threading.Thread(target=create_session)
    creator.start()
    assert creation_started.wait(timeout=2)
    closer = threading.Thread(target=runtime.close)
    closer.start()
    second_closer = threading.Thread(target=runtime.close)
    second_closer.start()
    allow_creation.set()
    creator.join(timeout=2)
    closer.join(timeout=2)
    second_closer.join(timeout=2)

    assert result.error is None
    assert created and created[0].closed
    assert not second_closer.is_alive()
    assert runtime._closed and runtime._session is None


# --- A-3: ptc=true exposes only exec, so exec's description is the sole place
# the model learns the nested tools. It must carry each tool's own description
# (apply_patch's V4A grammar in particular) plus a schema-derived signature.
# Nested tools follow the configured catalogue only — no silent injection.


def _exec_description_for(tool_specs):
    from mimoagent.tools.codex import CodexToolRegistry
    from mimoagent.tools.codex.code_mode import build_exec_tool_definition

    registry = CodexToolRegistry.from_config(tool_specs)
    definition = build_exec_tool_definition(nested_tools=list(registry.tools.values()))
    return definition["description"]


def test_exec_description_carries_apply_patch_v4a_grammar():
    description = _exec_description_for([{"tool": "exec_command"}, {"tool": "apply_patch"}])

    # The whole point of A-3: without these lines the model cannot author a V4A
    # patch, and apply_patch is its only way to edit files.
    assert "*** Begin Patch" in description
    assert "*** Move to" in description
    for action in ("Add File", "Delete File", "Update File"):
        assert action in description


def test_exec_description_carries_exec_command_shell_discipline():
    description = _exec_description_for([{"tool": "exec_command"}, {"tool": "apply_patch"}])

    assert "fresh shell" in description
    assert "no session to resume" in description.lower()
    # The stale upstream unified-exec wording must not reappear via rendering.
    assert "PTY" not in description
    assert "session ID" not in description


def test_exec_description_only_lists_configured_tools():
    description = _exec_description_for([{"tool": "exec_command"}, {"tool": "apply_patch"}])

    assert "### `exec_command`" in description
    assert "### `apply_patch`" in description
    assert "### `view_image`" not in description
    assert "### `update_plan`" not in description
    assert "path: string" not in description
    assert '"pending" | "in_progress" | "completed"' not in description


def test_full_catalogue_matches_previous_auto_registered_surface():
    """yaml tools with optional entries == the old ptc=true auto-register set."""
    description = _exec_description_for(
        [
            {"tool": "exec_command"},
            {"tool": "apply_patch"},
            {"tool": "view_image"},
            {"tool": "update_plan"},
        ]
    )

    for name in ("exec_command", "apply_patch", "view_image", "update_plan"):
        assert f"### `{name}`" in description
    assert "*** Begin Patch" in description
    assert "path: string" in description
    assert '"pending" | "in_progress" | "completed"' in description
    assert "apply_patch(input: string)" in description


def test_exec_description_signatures_are_derived_from_live_schema():
    description = _exec_description_for(
        [
            {"tool": "exec_command"},
            {"tool": "apply_patch"},
            {"tool": "view_image"},
            {"tool": "update_plan"},
        ]
    )

    # update_plan's status enum and view_image's path come straight from the
    # tools' schemas; a hand-written signature is what let max_output_tokens
    # drift, so assert the enum/field survive the JSON-schema -> TS rendering.
    assert '"pending" | "in_progress" | "completed"' in description
    assert "path: string" in description
    assert "apply_patch(input: string)" in description


def test_schema_to_ts_degrades_unknown_shapes_rather_than_guessing():
    from mimoagent.tools.codex.code_mode import _schema_to_ts

    assert _schema_to_ts({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}) == (
        "{\n  x: number;\n}"
    )
    assert _schema_to_ts({"type": "array", "items": {"type": "string"}}) == "Array<string>"
    assert _schema_to_ts({"enum": ["a", "b"]}) == '"a" | "b"'
    assert _schema_to_ts({"type": "object"}) == "Record<string, unknown>"
    assert _schema_to_ts({"type": "weird"}) == "unknown"


def test_schema_to_ts_carries_parameter_descriptions_as_doc_comments():
    """Under ptc=true the declaration is the only route for parameter docs.

    Dropping them hid the exec_command output budget default, so a model that
    raised ``max_output_tokens`` there could not know what it was raising it
    from — or that the cell budget would clip it again.
    """
    from mimoagent.tools.codex.code_mode import _schema_to_ts

    rendered = _schema_to_ts(
        {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "budget": {"type": "number", "description": "Defaults to 10000 tokens;\n  cut in the middle. */ x"},
                "plain": {"type": "boolean"},
            },
            "required": ["cmd"],
        },
        "  ",
    )

    assert rendered == (
        "{\n"
        "    /** Shell command to execute. */\n"
        "    cmd: string;\n"
        "    /** Defaults to 10000 tokens; cut in the middle. * / x */\n"
        "    budget?: number;\n"
        "    plain?: boolean;\n"
        "  }"
    )


def test_exec_description_states_both_budget_defaults_and_return_types():
    from mimoagent.tools.codex.output_budget import DEFAULT_MAX_OUTPUT_TOKENS

    description = _exec_description_for([{"tool": "exec_command"}, {"tool": "apply_patch"}, {"tool": "update_plan"}])

    # The cell budget sentence is upstream's; the nested exec_command budget
    # doc is upstream's shell_spec sentence. Neither mentions a cap.
    assert (
        f"`max_output_tokens` sets the token budget for direct `exec` results. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens."
        in description
    )
    assert (
        f"/** Output token budget. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens; larger requests may be capped by policy. */"
        in description
    )
    assert "Values above" not in description  # the old cap sentence
    assert "Promise<unknown>" not in description
    assert "apply_patch(input: string): Promise<string>;" in description
    assert "Promise<{ output: string; exit_code?: number; wall_time_seconds?: number;" in description
    assert "original_token_count?: number" in description
    assert "}): Promise<string>;" in description  # update_plan
    # The stale "string or object" claim is gone: function tools take an object.
    assert "either a string or an object" not in description


def test_exec_and_wait_definitions_quote_the_runtime_configured_yield_defaults():
    from mimoagent.tools.codex.code_mode import WaitTool, build_exec_tool_definition

    exec_definition = build_exec_tool_definition(nested_tools=[], yield_time_ms=300_000)
    assert "`yield_time_ms` (default 300000)" in exec_definition["description"]

    wait_properties = WaitTool.definition(yield_time_ms=100_000)["function"]["parameters"]["properties"]
    assert "Defaults to 100000" in wait_properties["yield_time_ms"]["description"]
    assert "Defaults to 10000 tokens" in wait_properties["max_tokens"]["description"]

    agent = _Agent()  # the runtime holds only a weakref
    runtime = CodeModeRuntime(agent, host_path=None, exec_yield_time_ms=42_000, wait_yield_time_ms=7_000)
    exec_def, wait_def = runtime.model_definitions()
    assert "`yield_time_ms` (default 42000)" in exec_def["description"]
    assert "Defaults to 7000" in wait_def["function"]["parameters"]["properties"]["yield_time_ms"]["description"]


def test_nested_apply_patch_and_update_plan_return_their_report_text():
    """An empty object told the model nothing; 99% of ptc trajectories then
    re-checked with git diff whether the patch had landed."""
    nested = CodeModeRuntime._nested_result

    assert nested(
        "apply_patch",
        {"success": True, "output": "Success. Updated the following files:\nM /testbed/app.py\nA /testbed/new.py"},
    ) == ("Success. Updated the following files:\nM /testbed/app.py\nA /testbed/new.py")
    assert nested("update_plan", {"success": True, "output": "Plan updated", "metadata": {"plan": []}}) == (
        "Plan updated"
    )
    # Layer 2 (``code_mode_result``): no budget -> the collected text as is.
    assert nested(
        "exec_command",
        {
            "success": True,
            "output": "x" * 100_000,
            "metadata": {
                "exit_code": 0,
                "wall_time_seconds": 0.5,
                "max_output_tokens": None,
                "output_omitted_bytes": 0,
            },
        },
    ) == {"output": "x" * 100_000, "exit_code": 0, "wall_time_seconds": 0.5}
    # With a budget -> cut under the header, and the original count reported.
    budgeted = nested(
        "exec_command",
        {
            "success": True,
            "output": "x" * 100_000,
            "metadata": {"exit_code": 0, "wall_time_seconds": 0.5, "max_output_tokens": 10, "output_omitted_bytes": 0},
        },
    )
    assert budgeted["original_token_count"] == 25_000
    assert budgeted["output"].startswith(
        "Warning: truncated output (original token count: 25000)\nTotal output lines: 1\n\n"
    )
    assert budgeted["output"].count("x") == 40
    assert "…24990 tokens truncated…" in budgeted["output"]


def test_a_nested_timeout_reaches_javascript_as_a_flagged_result_not_a_rejection():
    """``exec_command`` no longer raises on a deadline, so the cell keeps running
    and its siblings in a ``Promise.all`` keep their output. The flag is what
    lets the script tell a timeout from a plain non-zero exit."""

    result = CodeModeRuntime._nested_result(
        "exec_command",
        {
            "success": False,
            "output": "Error: Command timed out after 60s. Partial output below. ...",
            "metadata": {"exit_code": 124, "wall_time_seconds": 60.1, "reason": "pod_timeout", "timed_out": True},
        },
    )

    assert result["timed_out"] is True
    assert result["exit_code"] == 124
    assert "timed out" in result["output"]


# --- The four truncation layers reproduce codex-rs text byte for byte: marker
# wording, header wording, and the removed-token arithmetic (bytes over budget,
# not bytes removed), so a checkpoint trained on upstream traces sees the same
# shapes here.


def test_middle_cut_matches_upstream_marker_and_arithmetic():
    cut = truncate_middle_tokens("a" * 50_000 + "b" * 50_000, 10_000)

    assert cut == "a" * 20_000 + "…15000 tokens truncated…" + "b" * 20_000


def test_zero_budget_is_the_marker_alone_with_the_full_count():
    assert truncate_middle_tokens("a" * 100_000, 0) == "…25000 tokens truncated…"


def test_formatted_cut_carries_upstreams_header_and_leaves_fitting_text_alone():
    text = "line\n" * 20_000

    assert formatted_truncate_text(text, 100_000) == text
    cut = formatted_truncate_text(text, 10)
    assert cut.startswith("Warning: truncated output (original token count: 25000)\nTotal output lines: 20000\n\n")
    assert "…24990 tokens truncated…" in cut


def test_collection_layer_keeps_head_and_tail_under_the_byte_cap():
    kept, omitted = collect_head_tail("H" * 600 + "M" * 600 + "T" * 600, 1_000)

    assert omitted == 800
    assert kept == "H" * 500 + "\n... 800 bytes omitted ...\n" + "T" * 500
    assert collect_head_tail("short", 1_000) == ("short", 0)


def test_budgeted_cut_over_collected_text_keeps_the_omission_marker_visible():
    """``truncated_output_with_policy``: when the collection marker would fall
    inside the cut, it is restated under the header; when the text fits, the
    marker is prepended if the text lost it."""

    kept, omitted = collect_head_tail("H" * 600 + "M" * 600 + "T" * 600, 1_000)

    cut, original = truncate_exec_output(kept, omitted, 100)
    assert original == 257
    assert cut.startswith("Warning: truncated output (original token count: 257)\n... 800 bytes omitted ...\n\n")
    assert "…157 tokens truncated…" in cut

    untouched, none = truncate_exec_output(kept, omitted, 10_000)
    assert none is None and untouched == kept
    assert truncate_exec_output("plain", 0, None) == ("plain", None)


# --- image() forwarded whatever the cell passed straight to the gateway, so an
# undecodable payload came back as a ModelQueryError instead of a tool error.

_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def _image_result(url: str) -> dict:
    return {
        "Result": {
            "cell_id": "cell-1",
            "content_items": [
                {"type": "input_text", "text": "here is the shot"},
                {"type": "input_image", "image_url": url, "detail": "high"},
            ],
            "error_text": None,
        }
    }


def test_a_valid_data_url_is_still_forwarded_as_media():
    runtime = _runtime()

    output = runtime._format_response(_image_result(f"data:image/png;base64,{_PNG_BASE64}"), None, time.monotonic())

    assert len(output.media) == 1
    assert output.media[0]["media_type"] == "image/png"
    assert output.media[0]["data"] == _PNG_BASE64
    assert "here is the shot" in output.output


@pytest.mark.parametrize(
    "payload",
    ["not base64 at all!!", "aGVsbG8", "iVBORw0KGgo=extra"],
)
def test_an_undecodable_payload_is_reported_instead_of_sent(payload):
    runtime = _runtime()

    output = runtime._format_response(_image_result(f"data:image/png;base64,{payload}"), None, time.monotonic())

    assert output.media == []
    # Silently dropping it would leave the cell believing the image was sent.
    assert "image" in output.output.lower()
    assert "base64" in output.output.lower()
    assert "here is the shot" in output.output


def test_a_malformed_data_url_is_reported_too():
    runtime = _runtime()

    output = runtime._format_response(_image_result("https://example.com/cat.png"), None, time.monotonic())

    assert output.media == []
    assert "data:" in output.output
