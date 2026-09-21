"""Read tool: file-only reads (no tool-side truncation) and view_range."""

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.read import ReadTool


@pytest.fixture
def env():
    return LocalEnvironment()


@pytest.fixture
def tool():
    return ReadTool()


@pytest.fixture
def media_tool():
    return ReadTool({"enable_media": True})


def test_reads_whole_large_file(tool, env, tmp_path):
    """No 500-line clipping: the tool returns every line; truncation is the
    agent layer's job (utils.truncate)."""
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 2001)) + "\n")
    out = tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert "line-2000" in out.output
    assert "<response clipped>" not in out.output


def test_line_numbers_prefixed(tool, env, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("alpha\nbeta\n")
    out = tool.execute({"path": str(f)}, {"env": env})
    lines = out.output.splitlines()
    assert lines[0].split() == ["1", "alpha"]
    assert lines[1].split() == ["2", "beta"]


def test_view_range(tool, env, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 101)) + "\n")
    out = tool.execute({"path": str(f), "view_range": [10, 12]}, {"env": env})
    assert out.success
    assert "line-10" in out.output and "line-12" in out.output
    assert "line-9\n" not in out.output and "line-13" not in out.output


def test_view_range_to_eof(tool, env, tmp_path):
    """[start, -1] reads to EOF with no line cap."""
    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 1001)) + "\n")
    out = tool.execute({"path": str(f), "view_range": [900, -1]}, {"env": env})
    assert out.success
    assert "line-900" in out.output
    assert "line-1000" in out.output


def test_directory_rejected(tool, env, tmp_path):
    """Directories are not readable — the error points the model at bash."""
    out = tool.execute({"path": str(tmp_path)}, {"env": env})
    assert not out.success
    assert "directory" in out.output
    assert "bash" in out.output


def test_missing_path(tool, env):
    out = tool.execute({"path": "/nonexistent/definitely/missing"}, {"env": env})
    assert not out.success


def test_binary_file_rejected(tool, env, tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 64 + b"payload")
    out = tool.execute({"path": str(f)}, {"env": env})
    assert not out.success
    assert "binary" in out.output
    assert "bash" in out.output


def test_utf8_text_not_flagged_binary(tool, env, tmp_path):
    f = tmp_path / "cn.txt"
    f.write_text("中文内容\nsecond line\n", encoding="utf-8")
    out = tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert "中文内容" in out.output


_PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489") + b"\x00" * 16


def test_media_disabled_by_default(tool, env, tmp_path):
    """Without enable_media, image suffixes hit the binary sniff and are rejected."""
    f = tmp_path / "shot.png"
    f.write_bytes(_PNG)
    out = tool.execute({"path": str(f)}, {"env": env})
    assert not out.success
    assert out.media == []
    assert "binary" in out.output


def test_media_disabled_description_says_nothing_about_media(tool):
    """The default prompt must not advertise media reading at all."""
    desc = tool.description.lower()
    for word in ("media", "image", "audio", "video", "png", "wav", "mp4"):
        assert word not in desc


def test_media_enabled_description_mentions_media(media_tool):
    assert "images (png/jpg/jpeg/gif/webp)" in media_tool.description


def test_image_file_returned_as_image(media_tool, env, tmp_path):
    """png/jpg/... are not rejected as binary; they come back as base64 images."""
    import base64

    f = tmp_path / "shot.png"
    f.write_bytes(_PNG)
    out = media_tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert out.media == [{"kind": "image", "media_type": "image/png", "data": base64.b64encode(_PNG).decode()}]
    assert "image/png" in out.output


def test_audio_file_returned_as_audio(media_tool, env, tmp_path):
    import base64

    wav = b"RIFF" + b"\x00" * 40  # header-shaped bytes are enough for the tool
    f = tmp_path / "clip.wav"
    f.write_bytes(wav)
    out = media_tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert out.media == [
        {"kind": "audio", "media_type": "audio/wav", "data": base64.b64encode(wav).decode(), "format": "wav"}
    ]


def test_video_file_returned_as_video(media_tool, env, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)
    out = media_tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert out.media[0]["kind"] == "video"
    assert out.media[0]["media_type"] == "video/mp4"


def test_media_over_size_limit(env, tmp_path):
    from mimoagent.tools.read import ReadTool

    small_tool = ReadTool({"enable_media": True, "max_media_bytes": 10})
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00" * 64)
    out = small_tool.execute({"path": str(f)}, {"env": env})
    assert not out.success
    assert "byte limit" in out.output


def test_image_case_insensitive_extension(media_tool, env, tmp_path):
    f = tmp_path / "shot.JPG"
    f.write_bytes(_PNG)
    out = media_tool.execute({"path": str(f)}, {"env": env})
    assert out.success
    assert out.media[0]["media_type"] == "image/jpeg"


def test_image_missing_file(media_tool, env, tmp_path):
    out = media_tool.execute({"path": str(tmp_path / "missing.png")}, {"env": env})
    assert not out.success
    assert "does not exist" in out.output


def test_image_over_size_limit(env, tmp_path):
    from mimoagent.tools.read import ReadTool

    small_tool = ReadTool({"enable_media": True, "max_image_bytes": 10})
    f = tmp_path / "big.png"
    f.write_bytes(_PNG)
    out = small_tool.execute({"path": str(f)}, {"env": env})
    assert not out.success
    assert "byte limit" in out.output
