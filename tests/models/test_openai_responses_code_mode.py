"""Responses API round-trip tests for Codex custom/freeform tool calls."""

from types import SimpleNamespace

from mimoagent.models.openai_responses import _build_assistant_payload, _convert_messages


def _item(**fields):
    item = SimpleNamespace(**fields)
    item.model_dump = lambda exclude_none=False: dict(fields)
    return item


def test_custom_tool_call_becomes_raw_exec_arguments():
    item = _item(
        type="custom_tool_call",
        call_id="call-1",
        name="exec",
        input='text("ok");',
    )

    payload = _build_assistant_payload(SimpleNamespace(output=[item]))

    assert payload["tool_calls"] == [
        {
            "id": "call-1",
            "type": "custom",
            "function": {"name": "exec", "arguments": 'text("ok");'},
        }
    ]
    assert payload["responses_items"] == [
        {
            "type": "custom_tool_call",
            "call_id": "call-1",
            "name": "exec",
            "input": 'text("ok");',
        }
    ]


def test_custom_tool_call_and_output_replay_with_wire_types():
    raw_call = {
        "type": "custom_tool_call",
        "call_id": "call-1",
        "name": "exec",
        "input": 'text("ok");',
    }
    messages = [
        {
            "role": "assistant",
            "content": "",
            "responses_items": [raw_call],
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "custom",
                    "function": {"name": "exec", "arguments": 'text("ok");'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "exec",
            "tool_call_id": "call-1",
            "content": "Script completed\nOutput:\nok",
        },
    ]

    _, items = _convert_messages(messages)

    assert items == [
        raw_call,
        {
            "type": "custom_tool_call_output",
            "call_id": "call-1",
            "output": "Script completed\nOutput:\nok",
        },
    ]


def test_reconstructed_custom_tool_call_keeps_raw_source():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "custom",
                    "function": {"name": "exec", "arguments": 'text("ok");'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "exec",
            "tool_call_id": "call-1",
            "content": "ok",
        },
    ]

    _, items = _convert_messages(messages)

    assert items == [
        {
            "type": "custom_tool_call",
            "call_id": "call-1",
            "name": "exec",
            "input": 'text("ok");',
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call-1",
            "output": "ok",
        },
    ]


def test_code_mode_notification_keeps_exec_name_on_custom_output():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "custom",
                    "function": {"name": "exec", "arguments": 'notify("progress");'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "exec",
            "tool_call_id": "call-1",
            "content": "progress",
            "code_mode_notification": True,
        },
    ]

    _, items = _convert_messages(messages)

    assert items[-1] == {
        "type": "custom_tool_call_output",
        "call_id": "call-1",
        "name": "exec",
        "output": "progress",
    }
