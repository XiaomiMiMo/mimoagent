"""AnthropicModel: chat->Messages-API conversion, cache breakpoints, payload building."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from mimoagent.models import TokenStats
from mimoagent.models.anthropic import (
    AnthropicModel,
    _apply_cache_control,
    _build_assistant_payload,
    _convert_messages,
    _convert_tool_choice,
    _convert_tool_definition,
)

_IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def _model(**kwargs) -> AnthropicModel:
    kwargs.setdefault("model_name", "claude-test")
    kwargs.setdefault("model_kwargs", {"base_url": "http://gateway/v1", "api_key": "k"})
    return AnthropicModel(**kwargs)


class TestConstruction:
    def test_base_url_v1_suffix_stripped(self):
        """The SDK appends /v1/messages itself; shared openai-style configs keep working."""
        assert str(_model().client.base_url) == "http://gateway"

    def test_model_kwargs_not_mutated(self):
        model = _model()
        assert model.config.model_kwargs == {"base_url": "http://gateway/v1", "api_key": "k"}


class TestConvertMessages:
    def test_system_becomes_system_blocks(self):
        system, turns = _convert_messages([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
        assert system == [{"type": "text", "text": "sys"}]
        assert turns == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_assistant_tool_calls_become_tool_use(self):
        _, turns = _convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "on it",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
                    ],
                }
            ]
        )
        assert turns[0]["content"] == [
            {"type": "text", "text": "on it"},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
        ]

    def test_thinking_blocks_replay_first(self):
        thinking = {"type": "thinking", "thinking": "hmm", "signature": "sig"}
        _, turns = _convert_messages([{"role": "assistant", "content": "answer", "thinking_blocks": [thinking]}])
        assert turns[0]["content"][0] == thinking
        assert turns[0]["content"][1] == {"type": "text", "text": "answer"}

    def test_raw_blocks_replay_verbatim(self):
        """With ``anthropic_blocks`` present, replay ignores the chat projection:
        interleaved order, multiple text blocks and unparsed block types survive."""
        raw = [
            {"type": "thinking", "thinking": "a", "signature": "s1"},
            {"type": "text", "text": "part 1"},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
            {"type": "thinking", "thinking": "b", "signature": "s2"},
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "text", "text": "part 2"},
        ]
        msg = {
            "role": "assistant",
            "content": "part 1part 2",
            "thinking_blocks": [raw[0], raw[3]],
            "anthropic_blocks": raw,
        }
        _, turns = _convert_messages([msg])
        assert turns == [{"role": "assistant", "content": raw}]
        # replayed blocks are copies: mutating them must not touch the stash
        _apply_cache_control([], turns)
        turns[0]["content"][0]["mutated"] = True
        assert "mutated" not in msg["anthropic_blocks"][0]

    def test_tool_results_merge_into_one_user_turn(self):
        """Parallel tool results + a trailing user nudge must form a single user turn."""
        _, turns = _convert_messages(
            [
                {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "a", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "out1"},
                {"role": "tool", "tool_call_id": "c2", "content": "out2"},
                {"role": "user", "content": "continue"},
            ]
        )
        assert [t["role"] for t in turns] == ["assistant", "user"]
        user_blocks = turns[1]["content"]
        assert [b["type"] for b in user_blocks] == ["tool_result", "tool_result", "text"]
        assert user_blocks[0]["tool_use_id"] == "c1"
        assert user_blocks[0]["content"] == [{"type": "text", "text": "out1"}]

    def test_images_in_user_and_tool_messages(self):
        _, turns = _convert_messages(
            [
                {"role": "user", "content": [{"type": "text", "text": "look"}, _IMG]},
                {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "img"}, _IMG]},
            ]
        )
        image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
        # user + tool both map to role=user, so they merge into one turn
        assert len(turns) == 1
        assert turns[0]["content"][1] == image_block
        tool_result = turns[0]["content"][2]
        assert tool_result["type"] == "tool_result"
        assert tool_result["content"][1] == image_block

    def test_audio_video_degrade_to_placeholder_text(self):
        """The gateway silently drops audio/video blocks; an explicit placeholder is more honest."""
        _, turns = _convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,QUJD"}},
                    ],
                }
            ]
        )
        blocks = turns[0]["content"]
        assert [b["type"] for b in blocks] == ["text", "text"]
        assert "audio attachment omitted" in blocks[0]["text"]
        assert "video attachment omitted" in blocks[1]["text"]
        assert "QUJD" not in blocks[0]["text"] + blocks[1]["text"]

    def test_http_image_url_becomes_url_source(self):
        _, turns = _convert_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
        )
        assert turns[0]["content"][0]["source"] == {"type": "url", "url": "https://x/y.png"}


class TestCacheControl:
    def test_breakpoints_on_last_two_user_turns(self):
        system, turns = _convert_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ]
        )
        _apply_cache_control(system, turns)
        marked = [t["content"][-1].get("cache_control") for t in turns]
        assert marked == [None, None, {"type": "ephemeral"}, None, {"type": "ephemeral"}]
        assert "cache_control" not in system[0]

    def test_stale_marks_cleared_and_mark_lands_on_last_block(self):
        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "new"},
                ],
            }
        ]
        _apply_cache_control([], turns)
        assert "cache_control" not in turns[0]["content"][0]
        assert turns[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


class TestToolConversion:
    def test_function_tool_converted(self):
        converted = _convert_tool_definition(
            {"type": "function", "function": {"name": "bash", "description": "d", "parameters": {"type": "object"}}}
        )
        assert converted == {"name": "bash", "description": "d", "input_schema": {"type": "object"}}

    def test_tool_choice_mapping(self):
        assert _convert_tool_choice("auto") == {"type": "auto"}
        assert _convert_tool_choice("required") == {"type": "any"}
        assert _convert_tool_choice("none") == {"type": "none"}
        assert _convert_tool_choice({"type": "function", "function": {"name": "bash"}}) == {
            "type": "tool",
            "name": "bash",
        }
        assert _convert_tool_choice({"type": "any"}) == {"type": "any"}


class _FakeBlock(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        data = dict(vars(self))
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class TestPayload:
    def test_payload_from_response(self):
        response = SimpleNamespace(
            content=[
                _FakeBlock(type="thinking", thinking="hmm", signature="sig"),
                _FakeBlock(type="text", text="hello"),
                _FakeBlock(type="tool_use", id="c1", name="bash", input={"command": "ls"}),
            ],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=2
            ),
        )
        payload = _build_assistant_payload(response)
        assert payload["content"] == "hello"
        assert payload["reasoning_content"] == "hmm"
        assert payload["thinking_blocks"] == [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]
        assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
        assert [b["type"] for b in payload["anthropic_blocks"]] == ["thinking", "text", "tool_use"]

    def test_response_round_trips_verbatim(self):
        """response -> payload -> assistant message -> replay reproduces the wire
        blocks exactly, including a block type this module doesn't parse."""
        blocks = [
            _FakeBlock(type="thinking", thinking="t1", signature="s1"),
            _FakeBlock(type="text", text="first"),
            _FakeBlock(type="tool_use", id="c1", name="bash", input={"command": "ls"}),
            _FakeBlock(type="thinking", thinking="t2", signature="s2"),
            _FakeBlock(type="web_search_tool_result", tool_use_id="s1", content=[{"type": "x"}]),
            _FakeBlock(type="text", text="second"),
        ]
        payload = _build_assistant_payload(SimpleNamespace(content=blocks))
        _, turns = _convert_messages([{"role": "assistant", **payload}])
        assert turns[0]["content"] == [b.model_dump(exclude_none=True) for b in blocks]

    def test_query_records_stats_and_passes_system(self):
        model = _model()
        response = SimpleNamespace(
            content=[_FakeBlock(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=2
            ),
        )
        with patch.object(model.client.messages, "create", return_value=response) as mock_create:
            result = model.query(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {}}}],
                tool_choice="auto",
            )
        assert result == {"content": "ok", "anthropic_blocks": [{"type": "text", "text": "ok"}]}
        assert model.token_stats == TokenStats(10, 5, 3, 2)
        kwargs = mock_create.call_args.kwargs
        assert kwargs["system"] == [{"type": "text", "text": "sys"}]
        assert kwargs["tools"] == [{"name": "bash", "description": "d", "input_schema": {}}]
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["max_tokens"] == model.config.default_max_tokens
        # the sole user turn carries the cache breakpoint
        assert kwargs["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_stream_uses_sdk_helper_and_final_message(self):
        """``stream: true`` goes through ``messages.stream()``; the accumulated
        final message feeds the same payload/stats path as non-streaming."""
        model = _model(model_kwargs={"api_key": "k", "stream": True})
        response = SimpleNamespace(
            content=[_FakeBlock(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
        )

        class _FakeStream:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: response)

            def __exit__(self, *exc):
                return False

        with (
            patch.object(model.client.messages, "stream", return_value=_FakeStream()) as mock_stream,
            patch.object(model.client.messages, "create") as mock_create,
        ):
            result = model.query([{"role": "user", "content": "hi"}])
        assert result["content"] == "ok"
        assert model.token_stats == TokenStats(10, 5, 0, 0)
        mock_create.assert_not_called()
        assert "stream" not in mock_stream.call_args.kwargs
