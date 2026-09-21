"""DefaultAgent: tool outcomes with media become multimodal tool messages."""

from mimoagent.agents.default import DefaultAgent
from mimoagent.environments.local import LocalEnvironment
from mimoagent.models.test_models import DeterministicModel
from mimoagent.tools.base import ToolOutput


def _agent(**kwargs) -> DefaultAgent:
    return DefaultAgent(DeterministicModel(outputs=["unused"]), LocalEnvironment(), **kwargs)


def _call(call_id: str = "c1") -> dict:
    return {"id": call_id, "function": {"name": "read", "arguments": "{}"}}


def test_plain_outcome_stays_string():
    agent = _agent()
    agent._emit_outcome(_call(), ToolOutput(output="hello").to_dict())
    msg = agent.messages[-1]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "c1"
    assert msg["content"] == "hello"


def test_image_outcome_becomes_content_parts():
    agent = _agent()
    outcome = ToolOutput(
        output="Read image", media=[{"kind": "image", "media_type": "image/png", "data": "QUJD"}]
    ).to_dict()
    agent._emit_outcome(_call(), outcome)
    content = agent.messages[-1]["content"]
    assert content[0] == {"type": "text", "text": "Read image"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def test_audio_outcome_becomes_input_audio_part():
    agent = _agent()
    outcome = ToolOutput(
        output="Read audio",
        media=[{"kind": "audio", "media_type": "audio/wav", "data": "QUJD", "format": "wav"}],
    ).to_dict()
    agent._emit_outcome(_call(), outcome)
    content = agent.messages[-1]["content"]
    assert content[1] == {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}


def test_video_outcome_becomes_video_url_part():
    agent = _agent()
    outcome = ToolOutput(
        output="Read video", media=[{"kind": "video", "media_type": "video/mp4", "data": "QUJD"}]
    ).to_dict()
    agent._emit_outcome(_call(), outcome)
    content = agent.messages[-1]["content"]
    assert content[1] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,QUJD"}}


def test_media_survives_output_truncation():
    """The text half is truncated by max_observation_length; media ride along untouched."""
    agent = _agent(max_observation_length=100)
    outcome = ToolOutput(
        output="x" * 10_000, media=[{"kind": "image", "media_type": "image/png", "data": "QUJD"}]
    ).to_dict()
    agent._emit_outcome(_call(), outcome)
    content = agent.messages[-1]["content"]
    assert len(content[0]["text"]) < 10_000
    assert content[1]["type"] == "image_url"


def test_run_accepts_user_message_payload():
    """A dict task is the full first user message (may carry multimodal parts)."""
    agent = _agent()
    payload = {
        "content": [
            {"type": "text", "text": "what color?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]
    }
    status, message = agent.run(payload)
    assert status == "Idle"
    assert agent.messages[0]["role"] == "system"
    user_msg = agent.messages[1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == payload["content"]
    # instance_template is bypassed, but {{task}} still carries the text
    assert agent.extra_template_vars["task"] == "what color?"


def test_run_dict_payload_on_follow_up_turn():
    agent = _agent()
    agent.model.config.outputs = ["turn1", "turn2"]
    agent.run("first task")
    agent.run({"content": [{"type": "text", "text": "follow-up"}]})
    user_msgs = [m for m in agent.messages if m["role"] == "user"]
    assert user_msgs[1]["content"] == [{"type": "text", "text": "follow-up"}]
