"""Tests for the write tool (copy_to-based transfer)."""

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.base import ToolException
from mimoagent.tools.write import WriteTool


@pytest.fixture
def env():
    return LocalEnvironment()


@pytest.fixture
def tool():
    return WriteTool()


def run_write(tool, env, path, text):
    return tool.execute({"path": str(path), "file_text": text}, {"env": env})


def test_create_new_file(tool, env, tmp_path):
    p = tmp_path / "new.txt"
    out = run_write(tool, env, p, "hello\nworld\n")
    assert out.success, out.output
    assert p.read_text() == "hello\nworld\n"
    assert "created" in out.output


def test_overwrite_existing_file(tool, env, tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    out = run_write(tool, env, p, "new contents\n")
    assert out.success, out.output
    assert p.read_text() == "new contents\n"
    assert "overwritten" in out.output


def test_parent_dirs_created(tool, env, tmp_path):
    p = tmp_path / "a" / "b" / "c.txt"
    out = run_write(tool, env, p, "deep\n")
    assert out.success, out.output
    assert p.read_text() == "deep\n"


def test_large_file(tool, env, tmp_path):
    # Past ARG_MAX: contents must not travel on a command line.
    text = ("x" * 200 + "\n") * 5000  # ~1MB
    p = tmp_path / "big.txt"
    out = run_write(tool, env, p, text)
    assert out.success, out.output[:500]
    assert p.read_text() == text


def test_special_chars_and_unicode(tool, env, tmp_path):
    text = "quotes \" ' ` $VAR $(cmd) \\ % \t 中文 🚀\nno trailing newline"
    p = tmp_path / "special.txt"
    out = run_write(tool, env, p, text)
    assert out.success, out.output
    assert p.read_text() == text


def test_empty_file(tool, env, tmp_path):
    p = tmp_path / "empty.txt"
    out = run_write(tool, env, p, "")
    assert out.success, out.output
    assert p.read_text() == ""


def test_trailing_newlines_exact(tool, env, tmp_path):
    p = tmp_path / "nl.txt"
    out = run_write(tool, env, p, "a\n\n\n")
    assert out.success, out.output
    assert p.read_text() == "a\n\n\n"


def test_relative_path_rejected(tool, env):
    with pytest.raises(ToolException):
        tool.execute({"path": "rel.txt", "file_text": "x"}, {"env": env})


def test_missing_file_text_rejected(tool, env, tmp_path):
    with pytest.raises(ToolException):
        tool.execute({"path": str(tmp_path / "x.txt")}, {"env": env})
