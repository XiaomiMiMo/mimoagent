"""Output truncation for the Codex catalogue, layered the way codex-rs does it.

codex-rs (``ac192cd7``) bounds a command's output four times, each layer with
its own budget and its own marker, and the model is told about only two of
them. This module carries all four so the whitebox harness produces the same
text the checkpoint was trained on:

1. **Collection** (``unified_exec/head_tail_buffer.rs``): the process reader
   keeps at most ``UNIFIED_EXEC_OUTPUT_MAX_BYTES`` (1 MiB), half head and half
   tail, and joins them with ``... N bytes omitted ...``. Nothing above this
   layer ever sees more than 1 MiB.
2. **Nested result** (``tools/context.rs`` ``code_mode_result``): what
   ``tools.exec_command(...)`` resolves to inside JavaScript. If the call
   passed ``max_output_tokens`` the raw text is cut to that many tokens under
   the ``Warning: truncated output`` header; if it did not, JavaScript gets the
   collected text untouched. There is no ceiling on the request.
3. **Cell result** (``tools/code_mode/mod.rs`` ``truncate_code_mode_result``):
   everything the cell ``text()``-ed, cut to the ``// @exec:`` pragma's
   ``max_output_tokens`` (default ``DEFAULT_MAX_OUTPUT_TOKENS``), same header.
   Again no ceiling.
4. **History** (``context_manager/history.rs``): when the tool output is
   recorded into the conversation it is cut once more to the model's
   truncation policy times 1.2 (10000 × 1.2 = 12000 tokens for the gpt-5.x
   entries in ``models.json``), with the bare marker and no header. This is
   the only hard bound and the model is never told about it. In mimoagent it
   is ``DefaultAgentConfig.max_observation_length`` (characters), which
   ``CodexAgent`` defaults to 48000 = 12000 tokens x 4 bytes, applied by the
   agent with mimoagent's own ``truncate_middle`` banner.

Token estimates are byte-based (four bytes per token, ``APPROX_BYTES_PER_TOKEN``)
exactly as in ``codex-rs/utils/string/src/truncate.rs``. The marker text and
the removed-token arithmetic are upstream's: a cut reports
``total_bytes - budget_bytes`` in tokens, not the bytes actually removed, so a
second cut over already-cut text under-reports just as upstream does.
"""

from __future__ import annotations

import math

DEFAULT_MAX_OUTPUT_TOKENS = 10_000
APPROX_BYTES_PER_TOKEN = 4
UNIFIED_EXEC_OUTPUT_MAX_BYTES = 1024 * 1024


def approx_token_count(text: str) -> int:
    """``codex_utils_string::approx_token_count``: ceil(bytes / 4)."""

    return (len(text.encode("utf-8")) + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN


def resolve_max_tokens(requested: object) -> int:
    """``unified_exec::resolve_max_tokens``: the request, or the default when absent/unusable."""

    if (
        isinstance(requested, bool)
        or not isinstance(requested, int | float)
        or (isinstance(requested, float) and not math.isfinite(requested))
        or requested < 0
    ):
        return DEFAULT_MAX_OUTPUT_TOKENS
    return int(requested)


def format_output_omission_marker(omitted_bytes: int) -> str:
    """``unified_exec::format_output_omission_marker``."""

    return f"... {omitted_bytes} bytes omitted ..."


def collect_head_tail(text: str, max_bytes: int = UNIFIED_EXEC_OUTPUT_MAX_BYTES) -> tuple[str, int]:
    """Layer 1: keep ``max_bytes`` of ``text`` as head + tail, upstream's ``HeadTailBuffer``.

    Returns the retained text (with the omission marker between head and tail
    when anything was dropped, as ``to_bytes_with_omission_marker`` renders it)
    and the number of omitted bytes.
    """

    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, 0
    head_budget = max_bytes // 2
    tail_budget = max_bytes - head_budget
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail = encoded[len(encoded) - tail_budget :].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    return f"{head}\n{format_output_omission_marker(omitted)}\n{tail}", omitted


def _split_string(text: str, beginning_bytes: int, end_bytes: int) -> tuple[str, str]:
    """``truncate::split_string``: a prefix within ``beginning_bytes`` and a suffix within ``end_bytes``."""

    encoded = text.encode("utf-8")
    total = len(encoded)
    tail_start_target = max(0, total - end_bytes)
    prefix_end = 0
    suffix_start = total
    suffix_started = False
    index = 0
    for char in text:
        char_end = index + len(char.encode("utf-8"))
        if char_end <= beginning_bytes:
            prefix_end = char_end
        elif index >= tail_start_target and not suffix_started:
            suffix_start = index
            suffix_started = True
        index = char_end
    if suffix_start < prefix_end:
        suffix_start = prefix_end
    return encoded[:prefix_end].decode("utf-8"), encoded[suffix_start:].decode("utf-8")


def truncate_middle_tokens(text: str, max_tokens: int) -> str:
    """``truncate::truncate_middle_with_token_budget``: keep both ends within ``max_tokens``.

    The marker is upstream's ``…N tokens truncated…`` with N estimated from
    the bytes over budget. A zero budget yields the marker alone.
    """

    if not text:
        return ""
    max_bytes = max_tokens * APPROX_BYTES_PER_TOKEN
    total_bytes = len(text.encode("utf-8"))
    if max_tokens > 0 and total_bytes <= max_bytes:
        return text
    if max_bytes == 0:
        return f"…{approx_token_count(text)} tokens truncated…"
    if total_bytes <= max_bytes:
        return text
    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    prefix, suffix = _split_string(text, left_budget, right_budget)
    removed_tokens = (total_bytes - max_bytes + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN
    return f"{prefix}…{removed_tokens} tokens truncated…{suffix}"


def formatted_truncate_text(text: str, max_tokens: int) -> str:
    """``output_truncation::formatted_truncate_text``: the cut under its header.

    Unchanged text comes back as is; a cut is preceded by the original token
    count and line count so the model can size a re-read.
    """

    if len(text.encode("utf-8")) <= max_tokens * APPROX_BYTES_PER_TOKEN:
        return text
    original_token_count = approx_token_count(text)
    total_lines = len(text.splitlines())
    result = truncate_middle_tokens(text, max_tokens)
    return f"Warning: truncated output (original token count: {original_token_count})\nTotal output lines: {total_lines}\n\n{result}"


def truncate_exec_output(text: str, omitted_bytes: int, max_tokens: int | None) -> tuple[str, int | None]:
    """Layer 2 / direct policy: ``ExecCommandToolOutput::truncated_output_with_policy``.

    ``text`` is the collected output (already carrying the omission marker when
    ``omitted_bytes`` is non-zero). With no budget the text is returned as is.
    Otherwise the cut mirrors upstream: under the collection marker the header
    drops the line count and restores the marker if the cut removed it.
    Returns the text and the original token count when a cut happened.
    """

    if max_tokens is None:
        return text, None
    if omitted_bytes == 0:
        result = formatted_truncate_text(text, max_tokens)
        return result, (approx_token_count(text) if result != text else None)
    marker = format_output_omission_marker(omitted_bytes)
    if len(text.encode("utf-8")) <= max_tokens * APPROX_BYTES_PER_TOKEN:
        return (text if marker in text else f"{marker}\n{text}"), None
    original_token_count = approx_token_count(text)
    truncated = truncate_middle_tokens(text, max_tokens)
    omission_notice = "" if marker in truncated else f"{marker}\n"
    return (
        f"Warning: truncated output (original token count: {original_token_count})\n{omission_notice}\n{truncated}",
        original_token_count,
    )


__all__ = [
    "APPROX_BYTES_PER_TOKEN",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "UNIFIED_EXEC_OUTPUT_MAX_BYTES",
    "approx_token_count",
    "collect_head_tail",
    "format_output_omission_marker",
    "formatted_truncate_text",
    "resolve_max_tokens",
    "truncate_exec_output",
    "truncate_middle_tokens",
]
