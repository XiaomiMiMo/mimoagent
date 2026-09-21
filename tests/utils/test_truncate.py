"""Unified char-based middle truncation (utils.truncate)."""

from mimoagent.utils.truncate import truncate_middle


def test_short_text_passthrough():
    assert truncate_middle("hello", 20000) == "hello"


def test_exact_limit_passthrough():
    text = "x" * 100
    assert truncate_middle(text, 100) == text


def test_zero_or_negative_budget_disables():
    text = "x" * 1000
    assert truncate_middle(text, 0) == text
    assert truncate_middle(text, -1) == text


def test_keeps_head_and_tail():
    text = "A" * 6000 + "B" * 6000 + "C" * 6000
    out = truncate_middle(text, 10000)
    assert out.startswith("[TRUNCATION WARNING]")
    # Head: first 5000 chars are all A; tail: last 5000 chars are all C.
    assert "A" * 5000 in out
    assert "C" * 5000 in out
    # The all-B middle must be gone entirely (5000 B kept on neither side).
    assert "B" * 5001 not in out


def test_notice_and_marker_present():
    text = "x" * 30000
    out = truncate_middle(text, 20000)
    assert out.startswith("[TRUNCATION WARNING]")
    assert "[...OUTPUT OMITTED" in out
    # Notice states the accounting: total, omitted, kept head/tail.
    assert "30000" in out
    assert "10000" in out


def test_omitted_count_correct():
    text = "x" * 25000
    out = truncate_middle(text, 20000)
    assert "5000 characters" in out  # 25000 - 10000 - 10000


def test_hints_partial_read_without_naming_a_catalogue_tool():
    out = truncate_middle("x" * 30000, 20000)
    assert "narrower request" in out
    # The banner serves every catalogue: bashonly has no read tool, and the
    # range parameter is view_range in one catalogue and offset/limit in another.
    for tool_specific in ("view_range", "grep", "sed", "offset"):
        assert tool_specific not in out


def test_content_budget_respected():
    # Actual content kept (head + tail) is exactly max_chars.
    text = "".join(str(i % 10) for i in range(50000))
    out = truncate_middle(text, 20000)
    assert text[:10000] in out
    assert text[-10000:] in out
