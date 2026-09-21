"""CC Read image attachments and their path through the agent loop."""

import base64
import json
from unittest.mock import Mock

import pytest

from mimoagent.agents.cc.cc_agent import CCAgent
from mimoagent.environments.local import LocalEnvironment
from mimoagent.models.test_models import DeterministicModel
from mimoagent.tools.cc.read import ReadTool

_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=")


@pytest.fixture
def env():
    return LocalEnvironment()


@pytest.fixture
def tool():
    return ReadTool({"enable_media": True})


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "screenshot.png"
    path.write_bytes(_PNG)
    return path


def test_images_are_opt_in(env, image_path):
    tool = ReadTool()
    out = tool.execute({"file_path": str(image_path)}, {"env": env})
    assert out.media == []
    assert "image" not in tool.description.lower()


@pytest.mark.parametrize(
    ("suffix", "media_type"),
    [
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".JPEG", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
    ],
)
def test_supported_suffix_attaches_original_bytes(tool, env, tmp_path, suffix, media_type):
    path = tmp_path / f"screenshot{suffix}"
    path.write_bytes(_PNG)
    out = tool.execute({"file_path": str(path), "offset": 100, "limit": 1}, {"env": env})
    assert out.success
    assert out.media == [{"kind": "image", "media_type": media_type, "data": base64.b64encode(_PNG).decode()}]


def test_image_path_is_not_interpreted_by_shell(tool, env, tmp_path):
    path = tmp_path / "a \"quoted\" 'image' $(touch injected) `touch injected2` $HOME.png"
    path.write_bytes(_PNG)
    env.config.cwd = str(tmp_path)
    out = tool.execute({"file_path": str(path)}, {"env": env})
    assert out.success
    assert base64.b64decode(out.media[0]["data"]) == _PNG
    assert not (tmp_path / "injected").exists()
    assert not (tmp_path / "injected2").exists()


@pytest.mark.parametrize("max_bytes", [len(_PNG) - 1, len(_PNG)])
def test_image_size_limit(env, image_path, max_bytes):
    tool = ReadTool({"enable_media": True, "max_image_bytes": max_bytes})
    out = tool.execute({"file_path": str(image_path)}, {"env": env})
    if max_bytes < len(_PNG):
        assert not out.success
        assert out.media == []
        assert "byte limit" in out.output
    else:
        assert out.success
        assert base64.b64decode(out.media[0]["data"]) == _PNG


def test_empty_image_rejected(tool, env, tmp_path):
    path = tmp_path / "empty.png"
    path.touch()
    out = tool.execute({"file_path": str(path)}, {"env": env})
    assert not out.success
    assert out.media == []
    assert "empty" in out.output


def test_missing_image_rejected(tool, env, tmp_path):
    out = tool.execute({"file_path": str(tmp_path / "missing.png")}, {"env": env})
    assert not out.success
    assert out.media == []
    assert "does not exist" in out.output


@pytest.mark.parametrize(
    "encoded",
    [
        {"output": "base64: read error", "returncode": 1},
        {"output": "head: permission denied", "returncode": 0},
        {"output": "", "returncode": 0},
        {"output": base64.b64encode(_PNG[:-1]).decode(), "returncode": 0},
        {"output": base64.b64encode(_PNG).decode(), "returncode": 1},
    ],
)
def test_failed_or_incomplete_read_never_attaches_image(tool, encoded):
    env = Mock()
    env.execute.side_effect = [
        {"output": "FILE"},
        {"output": "OK"},
        {"output": str(len(_PNG)), "returncode": 0},
        encoded,
    ]
    out = tool.execute({"file_path": "/testbed/screenshot.png"}, {"env": env})
    assert not out.success
    assert out.media == []


def test_image_growing_after_stat_is_bounded_and_rejected(env, image_path, monkeypatch):
    tool = ReadTool({"enable_media": True, "max_image_bytes": len(_PNG)})
    execute = env.execute
    read_sizes = []

    def grow_before_read(command):
        if "| base64" in command:
            image_path.write_bytes(_PNG * 100)
            result = execute(command)
            read_sizes.append(len(base64.b64decode(result["output"])))
            return result
        return execute(command)

    monkeypatch.setattr(env, "execute", grow_before_read)
    out = tool.execute({"file_path": str(image_path)}, {"env": env})
    assert not out.success
    assert out.media == []
    assert read_sizes == [len(_PNG) + 1]


def test_media_mode_preserves_text_ranges_and_directory_listing(tool, env, tmp_path):
    directory = tmp_path / "assets.png"
    directory.mkdir()
    text = directory / "notes.txt"
    text.write_text("first\nsecond\nthird\n")
    out = tool.execute({"file_path": str(text), "offset": 2, "limit": 1}, {"env": env})
    assert out.success
    assert out.media == []
    assert out.output.splitlines()[0].split() == ["2", "second"]
    assert "first" not in out.output and "third" not in out.output
    out = tool.execute({"file_path": str(directory)}, {"env": env})
    assert out.success
    assert out.media == []
    assert "notes.txt" in out.output


def test_cc_agent_delivers_image_to_next_model_query(env, image_path, tmp_path, monkeypatch):
    model = DeterministicModel(outputs=[])
    call = {
        "id": "image-1",
        "type": "function",
        "function": {"name": "Read", "arguments": json.dumps({"file_path": str(image_path)})},
    }
    query = Mock(side_effect=[{"content": "", "tool_calls": [call]}, {"content": "done"}])
    monkeypatch.setattr(model, "query", query)
    agent = CCAgent(
        model,
        env,
        tools=[{"tool": "Read", "config": {"enable_media": True}}, {"tool": "Compact"}],
        max_observation_length=20,
        context_usage_warn_fraction=0,
        step_limit=3,
        msg_path=tmp_path / "messages.txt",
    )

    assert agent.run("Inspect the screenshot") == ("Idle", "done")
    assert query.call_count == 2
    messages = query.call_args_list[1].args[0]
    result = next(message for message in messages if message["role"] == "tool")
    assert result["name"] == "Read"
    assert result["tool_call_id"] == "image-1"
    assert "<context_usage>" in result["content"][0]["text"]
    assert result["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(_PNG).decode()},
    }
    assert base64.b64encode(_PNG).decode() not in agent.msg_path.read_text()
