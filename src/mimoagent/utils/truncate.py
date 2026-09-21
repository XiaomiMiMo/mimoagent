"""Character-based middle truncation for tool observations.

This is the truncation point for the default and CC catalogues: individual
tools return their full output and ``DefaultAgent._emit_outcome`` runs it
through :func:`truncate_middle` before the tool message is appended. Those
tools must not pre-truncate — a partial output silently clipped by a tool
would then look complete to the model.

A catalogue that budgets its own output is the exception, not a second cut on
top of this one. The Codex tools cap in tokens the way codex-rs does, so
``CodexAgentConfig`` sets ``max_observation_length`` to 0 to disable this
function outright. Stacking both was worse than either alone: whichever cut
ran second kept the head and tail of an already-clipped string, deleting the
first cut's marker from the middle and then reporting the clipped length as if
it were the original.

Character-based on purpose: tokenizer-based truncation encodes/decodes the
whole observation per tool call, which is far too expensive for a hot path
that runs on every tool response. Chars are a good-enough proxy and free.

The truncation must be unmissable. A model that doesn't notice the cut treats
the spliced text as contiguous and hallucinates about the missing middle, so
the output gets a loud banner up front (with re-read advice) AND an inline
marker at the exact cut point. The advice names no tool or parameter: this
function serves every catalogue (bashonly has no read tool at all, and the
range parameter is ``view_range`` in one catalogue and ``offset``/``limit`` in
another), so a concrete hint is wrong for most of them.
"""

from __future__ import annotations

_NOTICE_TEMPLATE = """\
[TRUNCATION WARNING] The tool output below is INCOMPLETE: it was {total} characters, exceeding the {max_chars}-character limit, so the MIDDLE was cut out. Only the first {head} and last {tail} characters are kept; {omitted} characters are missing at the point marked [...OUTPUT OMITTED...]. Do NOT assume the text is contiguous across that marker. If you need the omitted part, make a narrower request instead (read only the relevant line range, or filter / paginate the command output) rather than repeating the same call.

"""

_CUT_MARKER_TEMPLATE = """

[...OUTPUT OMITTED: {omitted} characters truncated from the middle ({total} total, kept first {head} + last {tail}). The text below resumes near the END of the output...]

"""


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep the first and last ``max_chars // 2`` characters of ``text``.

    Returns ``text`` unchanged when it fits. When it doesn't, the result is a
    warning banner + head + inline cut marker + tail (the banner/marker are on
    top of the ``max_chars`` content budget).
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - head - tail
    fields = {
        "total": len(text),
        "max_chars": max_chars,
        "head": head,
        "tail": tail,
        "omitted": omitted,
    }
    return _NOTICE_TEMPLATE.format(**fields) + text[:head] + _CUT_MARKER_TEMPLATE.format(**fields) + text[-tail:]
