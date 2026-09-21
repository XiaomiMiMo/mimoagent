"""OpenAIChatModel: kwarg routing and payload building."""

from unittest.mock import Mock

from mimoagent.models.openai_chat import (
    OpenAIChatModel,
    _build_assistant_payload,
    _route_unknown_kwargs,
)


def _model(**kwargs) -> OpenAIChatModel:
    kwargs.setdefault("model_name", "mimo-test")
    kwargs.setdefault("model_kwargs", {"base_url": "http://gateway/v1", "api_key": "k"})
    return OpenAIChatModel(**kwargs)


class TestConstruction:
    def test_model_name_sent_verbatim(self):
        """No prefix stripping: model_name is the serving name (slashes allowed)."""
        assert _model().config.model_name == "mimo-test"
        assert _model(model_name="z-ai/glm-x").config.model_name == "z-ai/glm-x"

    def test_model_kwargs_not_mutated(self):
        """Blackbox harnesses read config.model_kwargs[base_url/api_key] after construction."""
        model = _model()
        assert model.config.model_kwargs == {"base_url": "http://gateway/v1", "api_key": "k"}


class TestKwargRouting:
    def test_unknown_kwargs_tunnelled_to_extra_body(self):
        routed = _route_unknown_kwargs({"temperature": 1.0, "thinking": {"type": "adaptive"}})
        assert routed["temperature"] == 1.0
        assert "thinking" not in routed
        assert routed["extra_body"]["thinking"] == {"type": "adaptive"}

    def test_explicit_extra_body_wins_on_conflict(self):
        routed = _route_unknown_kwargs({"foo": 1, "extra_body": {"foo": 2}})
        assert routed["extra_body"]["foo"] == 2


class TestPayload:
    def test_payload_from_message(self):
        message = Mock()
        message.content = "hello"
        message.reasoning_content = "thinking..."
        call = Mock()
        call.id, call.type = "c1", "function"
        call.function.name, call.function.arguments = "bash", '{"command": "ls"}'
        message.tool_calls = [call]
        payload = _build_assistant_payload(message)
        assert payload["content"] == "hello"
        assert payload["reasoning_content"] == "thinking..."
        assert payload["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
        ]
