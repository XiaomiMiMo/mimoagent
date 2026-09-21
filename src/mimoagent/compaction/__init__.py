"""Model-decided conversation compaction.

A ``Compact`` tool (``mimoagent.tools.compact``) lets the model summarize its own
conversation history to free context. This package holds the summary prompt
(``prompt``) and the message-surgery helpers (``surgery``) that build the slice
to summarize and the rebuilt ``[system…, boundary, original issue anchor,
summary]`` history.
"""

from mimoagent.compaction.prompt import build_codex_compact_prompt, build_compact_prompt, format_compact_summary
from mimoagent.compaction.surgery import (
    CODEX_SUMMARY_PREFIX,
    COMPACT_BOUNDARY_PREFIX,
    COMPACT_ORIGINAL_USER_ISSUE_PREFIX,
    COMPACT_SUMMARY_PREFIX,
    build_local_compacted_history,
    build_summary_request_message,
    create_compact_boundary_message,
    create_compact_summary_message,
    create_original_user_issue_anchor_message,
    get_messages_after_last_compact_boundary,
    is_codex_summary_message,
    is_compact_boundary_message,
    is_compact_summary_message,
    is_original_user_issue_anchor_message,
    messages_for_compaction,
    original_user_issue_anchor_message,
    split_leading_system_messages,
)

__all__ = [
    "COMPACT_BOUNDARY_PREFIX",
    "COMPACT_ORIGINAL_USER_ISSUE_PREFIX",
    "COMPACT_SUMMARY_PREFIX",
    "CODEX_SUMMARY_PREFIX",
    "build_compact_prompt",
    "build_codex_compact_prompt",
    "build_summary_request_message",
    "build_local_compacted_history",
    "create_compact_boundary_message",
    "create_original_user_issue_anchor_message",
    "create_compact_summary_message",
    "format_compact_summary",
    "get_messages_after_last_compact_boundary",
    "is_compact_boundary_message",
    "is_compact_summary_message",
    "is_original_user_issue_anchor_message",
    "messages_for_compaction",
    "original_user_issue_anchor_message",
    "is_codex_summary_message",
    "split_leading_system_messages",
]
