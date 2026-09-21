"""Comprehensive correctness/robustness tests for the edit tool.

The edit tool must behave like an exact byte-level ``str.replace(old, new, 1)``
on the file contents, with uniqueness enforcement — regardless of what shell
machinery implements it underneath. All tests run against LocalEnvironment so
they execute the same bash the container would run.
"""

import glob
import time

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.base import ToolException
from mimoagent.tools.edit import EditTool


@pytest.fixture
def env():
    return LocalEnvironment()


@pytest.fixture
def tool():
    return EditTool()


@pytest.fixture
def workfile(tmp_path):
    return tmp_path / "target.txt"


def run_edit(tool, env, path, old_str, new_str=None):
    params = {"path": str(path), "old_str": old_str}
    if new_str is not None:
        params["new_str"] = new_str
    return tool.execute(params, {"env": env})


def assert_no_residue(tmp_path):
    """The tool must not leave backups or temp scratch files behind."""
    leftovers = [p for p in glob.glob(str(tmp_path / "*")) if not p.endswith("target.txt")]
    assert leftovers == [], f"residue next to target: {leftovers}"
    # Sibling tests running in parallel (pytest-xdist) create the same
    # /tmp/mimo_edit_* scratch dirs transiently; a real leak persists, a
    # neighbour's in-flight edit disappears within milliseconds.
    deadline = time.monotonic() + 3.0
    while True:
        tmp_residue = glob.glob("/tmp/miniswe_*") + glob.glob("/tmp/mimo_edit_*")
        if not tmp_residue or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert tmp_residue == [], f"residue in /tmp: {tmp_residue}"


# --- basic semantics ---------------------------------------------------------


def test_basic_replace(tool, env, workfile, tmp_path):
    workfile.write_text("hello world\nsecond line\n")
    out = run_edit(tool, env, workfile, "hello world", "goodbye world")
    assert out.success, out.output
    assert workfile.read_text() == "goodbye world\nsecond line\n"
    assert_no_residue(tmp_path)


def test_multiline_replace(tool, env, workfile):
    original = "def foo():\n    a = 1\n    b = 2\n    return a + b\n"
    workfile.write_text(original)
    out = run_edit(tool, env, workfile, "    a = 1\n    b = 2", "    a = 10\n    b = 20\n    c = 30")
    assert out.success, out.output
    assert workfile.read_text() == "def foo():\n    a = 10\n    b = 20\n    c = 30\n    return a + b\n"


def test_delete_via_empty_new_str(tool, env, workfile):
    workfile.write_text("keep\nDELETE ME\nkeep too\n")
    out = run_edit(tool, env, workfile, "DELETE ME\n", "")
    assert out.success, out.output
    assert workfile.read_text() == "keep\nkeep too\n"


def test_delete_via_omitted_new_str(tool, env, workfile):
    workfile.write_text("aXb")
    out = run_edit(tool, env, workfile, "X")
    assert out.success, out.output
    assert workfile.read_text() == "ab"


def test_replace_at_file_start(tool, env, workfile):
    workfile.write_text("first line\nrest\n")
    out = run_edit(tool, env, workfile, "first", "FIRST")
    assert out.success, out.output
    assert workfile.read_text() == "FIRST line\nrest\n"


def test_replace_at_file_end_no_trailing_newline(tool, env, workfile):
    workfile.write_text("alpha\nomega")
    out = run_edit(tool, env, workfile, "omega", "OMEGA")
    assert out.success, out.output
    assert workfile.read_text() == "alpha\nOMEGA"


def test_whole_file_replace(tool, env, workfile):
    workfile.write_text("entire content")
    out = run_edit(tool, env, workfile, "entire content", "new content")
    assert out.success, out.output
    assert workfile.read_text() == "new content"


# --- byte-exactness: newlines -----------------------------------------------


def test_file_trailing_newline_preserved(tool, env, workfile):
    workfile.write_text("a\nb\nc\n")
    out = run_edit(tool, env, workfile, "b", "B")
    assert out.success, out.output
    assert workfile.read_text() == "a\nB\nc\n"


def test_file_without_trailing_newline_not_given_one(tool, env, workfile):
    workfile.write_text("a\nb")
    out = run_edit(tool, env, workfile, "a", "A")
    assert out.success, out.output
    assert workfile.read_text() == "A\nb"


def test_old_str_with_trailing_newline_matched_exactly(tool, env, workfile):
    # "x\n" (with newline) must consume the newline; the lines join afterwards.
    workfile.write_text("one\nx\ntwo\n")
    out = run_edit(tool, env, workfile, "x\n", "")
    assert out.success, out.output
    assert workfile.read_text() == "one\ntwo\n"


def test_new_str_trailing_newline_preserved(tool, env, workfile):
    workfile.write_text("AB")
    out = run_edit(tool, env, workfile, "A", "A\n")
    assert out.success, out.output
    assert workfile.read_text() == "A\nB"


def test_old_str_with_multiple_trailing_newlines(tool, env, workfile):
    workfile.write_text("head\n\n\ntail\n")
    out = run_edit(tool, env, workfile, "head\n\n\n", "head\n")
    assert out.success, out.output
    assert workfile.read_text() == "head\ntail\n"


def test_blank_line_in_middle_of_old_str(tool, env, workfile):
    workfile.write_text("def a():\n    pass\n\n\ndef b():\n    pass\n")
    out = run_edit(tool, env, workfile, "    pass\n\n\ndef b():", "    pass\n\n\ndef c():")
    assert out.success, out.output
    assert workfile.read_text() == "def a():\n    pass\n\n\ndef c():\n    pass\n"


def test_crlf_content_preserved(tool, env, workfile):
    workfile.write_bytes(b"line1\r\nline2\r\nline3\r\n")
    out = run_edit(tool, env, workfile, "line2", "LINE2")
    assert out.success, out.output
    assert workfile.read_bytes() == b"line1\r\nLINE2\r\nline3\r\n"


# --- uniqueness / not-found -------------------------------------------------


def test_not_found_fails_and_file_unchanged(tool, env, workfile, tmp_path):
    original = "nothing to see here\n"
    workfile.write_text(original)
    out = run_edit(tool, env, workfile, "ABSENT", "x")
    assert not out.success
    assert "not found" in out.output.lower()
    assert workfile.read_text() == original
    assert_no_residue(tmp_path)


def test_multiple_occurrences_fails_and_file_unchanged(tool, env, workfile, tmp_path):
    original = "dup\nmiddle\ndup\n"
    workfile.write_text(original)
    out = run_edit(tool, env, workfile, "dup", "x")
    assert not out.success
    assert workfile.read_text() == original
    assert_no_residue(tmp_path)


def test_overlapping_occurrences_counted(tool, env, workfile):
    # "aa" occurs twice in "aaa" (non-overlapping count = 1, overlapping = 2).
    # Either count > 1 rejection or a consistent first-match replace is
    # acceptable; what is NOT acceptable is corrupting the file. We pin the
    # non-overlapping str.count semantics (1 match -> replace first).
    workfile.write_text("aaa")
    out = run_edit(tool, env, workfile, "aa", "b")
    assert out.success, out.output
    assert workfile.read_text() == "ba"


def test_multiline_duplicate_blocks_rejected(tool, env, workfile):
    block = "if x:\n    return 1\n"
    workfile.write_text(block + "else:\n    pass\n" + block)
    out = run_edit(tool, env, workfile, block, "CHANGED\n")
    assert not out.success


# --- special characters ------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        'quote " double',
        "quote ' single",
        "back\\slash",
        "dollar $VAR ${VAR} $(cmd)",
        "backtick `cmd`",
        "percent %s %d",
        "glob * ? [a-z]",
        "regex .* ^ $ ( ) | + chars",
        "ampersand & here",
        "semicolon ; pipe | redirect > <",
        "tab\tinside",
        "exclam !history",
        "tilde ~user",
        "hash # comment",
    ],
)
def test_special_chars_roundtrip(tool, env, workfile, needle):
    content = f"before\n{needle}\nafter\n"
    workfile.write_text(content)
    out = run_edit(tool, env, workfile, needle, "REPLACED")
    assert out.success, f"{needle!r}: {out.output}"
    assert workfile.read_text() == "before\nREPLACED\nafter\n"


def test_awk_ampersand_not_expanded_in_replacement(tool, env, workfile):
    # awk's sub()/gsub() expand "&" to the matched text; the tool must insert
    # new_str literally.
    workfile.write_text("X marks the spot\n")
    out = run_edit(tool, env, workfile, "X", "A & B")
    assert out.success, out.output
    assert workfile.read_text() == "A & B marks the spot\n"


def test_backslash_sequences_literal(tool, env, workfile):
    workfile.write_text("path = OLD\n")
    out = run_edit(tool, env, workfile, "OLD", r"C:\new\table\0end")
    assert out.success, out.output
    assert workfile.read_text() == "path = C:\\new\\table\\0end\n"


def test_unicode_content(tool, env, workfile):
    workfile.write_text("# 旧的注释 🚀\nvalue = 1\n", encoding="utf-8")
    out = run_edit(tool, env, workfile, "# 旧的注释 🚀", "# 新的注释 ✨")
    assert out.success, out.output
    assert workfile.read_text(encoding="utf-8") == "# 新的注释 ✨\nvalue = 1\n"


def test_single_quotes_heavy_code(tool, env, workfile):
    old = "print('it''s \"quoted\"')"
    workfile.write_text(f"{old}\n")
    out = run_edit(tool, env, workfile, old, 'print("done")')
    assert out.success, out.output
    assert workfile.read_text() == 'print("done")\n'


# --- scale -------------------------------------------------------------------


def test_large_old_and_new_strings(tool, env, workfile):
    # Way past MAX_ARG_STRLEN (128KiB per argv element on Linux): the tool may
    # not pass file contents through a command line.
    old = "\n".join(f"line {i}: " + "x" * 100 for i in range(3000))  # ~320KB
    new = old.replace("x", "y")
    workfile.write_text(f"HEADER\n{old}\nFOOTER\n")
    out = run_edit(tool, env, workfile, old, new)
    assert out.success, out.output[:500]
    assert workfile.read_text() == f"HEADER\n{new}\nFOOTER\n"


def test_small_edit_in_large_file(tool, env, workfile):
    filler = "\n".join(f"filler {i}" for i in range(50000))  # ~600KB
    workfile.write_text(f"{filler}\nNEEDLE\n{filler}2\n")
    out = run_edit(tool, env, workfile, "NEEDLE", "THREAD")
    assert out.success, out.output[:500]
    assert "THREAD" in workfile.read_text()
    assert "NEEDLE" not in workfile.read_text()


# --- argument validation -----------------------------------------------------


def test_nonexistent_file(tool, env, tmp_path):
    out = run_edit(tool, env, tmp_path / "ghost.txt", "a", "b")
    assert not out.success
    assert "not found" in out.output.lower()


def test_relative_path_rejected(tool, env):
    with pytest.raises(ToolException):
        tool.execute({"path": "relative.txt", "old_str": "a"}, {"env": env})


def test_missing_old_str_rejected(tool, env, workfile):
    workfile.write_text("x")
    with pytest.raises(ToolException):
        tool.execute({"path": str(workfile)}, {"env": env})


def test_empty_old_str_rejected(tool, env, workfile):
    workfile.write_text("x")
    out_or_exc = None
    try:
        out_or_exc = run_edit(tool, env, workfile, "", "y")
    except ToolException:
        return  # also acceptable
    assert not out_or_exc.success
    assert workfile.read_text() == "x"


def test_old_equals_new_rejected(tool, env, workfile):
    workfile.write_text("same\n")
    out_or_exc = None
    try:
        out_or_exc = run_edit(tool, env, workfile, "same", "same")
    except ToolException:
        return  # also acceptable
    assert not out_or_exc.success
    assert workfile.read_text() == "same\n"


def test_no_env_rejected(tool):
    with pytest.raises(ToolException):
        tool.execute({"path": "/x", "old_str": "a"}, {})


# --- output quality ----------------------------------------------------------


def test_success_output_mentions_change(tool, env, workfile):
    workfile.write_text("old_value = 1\n")
    out = run_edit(tool, env, workfile, "old_value", "new_value")
    assert out.success
    # The observation should show evidence of the change (a diff or snippet).
    assert "new_value" in out.output
