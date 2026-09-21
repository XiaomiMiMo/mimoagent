"""OpenAIResponsesModel: multimodal input conversion, reasoning extraction, streaming."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mimoagent.models.openai_responses import (
    OpenAIResponsesModel,
    _build_assistant_payload,
    _convert_messages,
    _convert_tool_output,
)

_IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


class TestConvertMessages:
    def test_first_system_becomes_instructions(self):
        instructions, items = _convert_messages(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        )
        assert instructions == "sys"
        assert items == [{"role": "user", "content": "hi"}]

    def test_user_images_become_input_image_parts(self):
        _, items = _convert_messages([{"role": "user", "content": [{"type": "text", "text": "look"}, _IMG]}])
        assert items[0]["content"] == [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
        ]

    def test_image_detail_is_preserved(self):
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,QUJD", "detail": "original"},
        }
        _, items = _convert_messages([{"role": "user", "content": [image]}])
        assert items[0]["content"] == [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,QUJD",
                "detail": "original",
            }
        ]

    def test_tool_output_plain_text_stays_string(self):
        assert _convert_tool_output("done") == "done"
        assert _convert_tool_output([{"type": "text", "text": "done"}]) == "done"

    def test_audio_video_degrade_to_placeholder_text(self):
        """The gateway 400s on audio/video input items; placeholders keep the request alive."""
        _, items = _convert_messages(
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
        parts = items[0]["content"]
        assert [p["type"] for p in parts] == ["input_text", "input_text"]
        assert "audio attachment omitted" in parts[0]["text"]
        assert "video attachment omitted" in parts[1]["text"]

    def test_tool_output_with_images_becomes_parts(self):
        out = _convert_tool_output([{"type": "text", "text": "img"}, _IMG])
        assert out == [
            {"type": "input_text", "text": "img"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
        ]


class TestPayload:
    def test_reasoning_from_content_reasoning_text(self):
        """vLLM and some gateways put reasoning under item.content[], not item.summary."""
        item = SimpleNamespace(
            type="reasoning",
            summary=[],
            content=[SimpleNamespace(type="reasoning_text", text="thought")],
        )
        item.model_dump = lambda exclude_none=False: {"type": "reasoning"}
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="hello")],
        )
        message.model_dump = lambda exclude_none=False: {"type": "message"}
        payload = _build_assistant_payload(SimpleNamespace(output=[item, message]))
        assert payload["content"] == "hello"
        assert payload["reasoning_content"] == "thought"

    def test_reasoning_from_summary_still_works(self):
        item = SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="sum")], content=None)
        item.model_dump = lambda exclude_none=False: {"type": "reasoning"}
        message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="hi")])
        message.model_dump = lambda exclude_none=False: {"type": "message"}
        payload = _build_assistant_payload(SimpleNamespace(output=[item, message]))
        assert payload["reasoning_content"] == "sum"


def _model(**model_kwargs) -> OpenAIResponsesModel:
    model_kwargs.setdefault("api_key", "k")
    return OpenAIResponsesModel(model_name="gpt-test", model_kwargs=model_kwargs)


def _final_response():
    message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="ok")])
    message.model_dump = lambda exclude_none=False: {"type": "message"}
    return SimpleNamespace(
        output=[message],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, input_tokens_details=None),
    )


class TestStreaming:
    def test_stream_returns_terminal_event_response(self):
        """``stream: true`` consumes SSE events; the ``response.completed`` event
        carries the full Response, feeding the same payload path as non-streaming."""
        model = _model(stream=True)
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="o"),
            SimpleNamespace(type="response.output_text.delta", delta="k"),
            SimpleNamespace(type="response.completed", response=_final_response()),
        ]
        with patch.object(model.client.responses, "create", return_value=iter(events)) as mock_create:
            result = model.query([{"role": "user", "content": "hi"}])
        assert result["content"] == "ok"
        assert model.token_stats.input_tokens == 10
        assert mock_create.call_args.kwargs["stream"] is True

    def test_stream_failure_event_raises(self):
        model = _model()
        events = [SimpleNamespace(type="response.failed", response=SimpleNamespace(error="boom"))]
        with patch.object(model.client.responses, "create", return_value=iter(events)):
            with pytest.raises(ValueError, match="failure"):
                model._stream_final_response([], {})

    def test_stream_without_terminal_event_raises(self):
        model = _model()
        with patch.object(model.client.responses, "create", return_value=iter([])):
            with pytest.raises(ValueError, match="without a terminal"):
                model._stream_final_response([], {})
