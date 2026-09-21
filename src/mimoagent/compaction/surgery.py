"""Message-surgery helpers for model-decided conversation compaction.

These build the slice of history to summarize and the rebuilt history that
replaces it. The rebuilt history keeps a verbatim initial user issue anchor so
the resumed agent does not depend on the summary to preserve the task. The
trigger is a model tool call (see ``tools/compact.py``); there is no automatic
token-threshold or reactive-on-error compaction here.
"""

from __future__ import annotations

import time

from mimoagent.compaction.prompt import build_compact_prompt
from mimoagent.utils.cal_token import rough_token_count_estimation_for_message

COMPACT_BOUNDARY_PREFIX = "[compact boundary]"
COMPACT_SUMMARY_PREFIX = "[compact summary]"
COMPACT_ORIGINAL_USER_ISSUE_PREFIX = "[original user issue]"
CODEX_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking process. "
    "You also have access to the state of the tools that were used by that language model. Use this "
    "to build on the work that has already been done and avoid duplicating work. Here is the summary produced "
    "by the other language model, use the information in this summary to assist with your own analysis:"
)


def is_codex_summary_message(message: dict) -> bool:
    """Return whether a user message is an installed Codex handoff summary."""
    return message.get("role") == "user" and str(message.get("content") or "").startswith(CODEX_SUMMARY_PREFIX)


def is_compact_boundary_message(message: dict) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith(COMPACT_BOUNDARY_PREFIX)


def is_compact_summary_message(message: dict) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith(COMPACT_SUMMARY_PREFIX)


def is_original_user_issue_anchor_message(message: dict) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith(
        COMPACT_ORIGINAL_USER_ISSUE_PREFIX
    )


def create_original_user_issue_anchor_message(message: dict) -> dict:
    """Return a user message that preserves the initial issue verbatim."""
    if is_original_user_issue_anchor_message(message):
        return dict(message)

    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)

    return {
        "role": "user",
        "content": (
            f"{COMPACT_ORIGINAL_USER_ISSUE_PREFIX}\n"
            "The initial user issue is preserved verbatim below as a stable task anchor across compaction.\n\n"
            "<original_user_issue>\n"
            f"{content.strip()}\n"
            "</original_user_issue>"
        ),
    }


def original_user_issue_anchor_message(messages: list[dict]) -> dict | None:
    """Find the initial user issue and return it as a compaction anchor."""
    _, conversation = split_leading_system_messages(messages)
    for message in conversation:
        if message.get("role") != "user":
            continue
        if is_compact_boundary_message(message) or is_compact_summary_message(message):
            continue
        return create_original_user_issue_anchor_message(message)
    return None


def get_messages_after_last_compact_boundary(messages: list[dict]) -> list[dict]:
    """Return messages after the newest compact boundary.

    This excludes the boundary itself and keeps the previous compact summary plus
    post-compact conversation, which makes repeated compaction summarize the last
    summary and new work instead of resurrecting already discarded history.
    """
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return messages[index + 1 :]
    return list(messages)


def split_leading_system_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    split_index = 0
    while split_index < len(messages) and messages[split_index].get("role") == "system":
        split_index += 1
    return list(messages[:split_index]), list(messages[split_index:])


def messages_for_compaction(messages: list[dict]) -> list[dict]:
    """Build the slice handed to the summarizer: system messages + the
    conversation since the last compact boundary."""
    system_messages, conversation = split_leading_system_messages(messages)
    visible_conversation = get_messages_after_last_compact_boundary(conversation)
    return [*system_messages, *visible_conversation]


def create_compact_boundary_message(*, compact_count: int, pre_tokens: int) -> dict:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "role": "user",
        "content": (
            f"{COMPACT_BOUNDARY_PREFIX}\n"
            f"trigger: model\n"
            f"compact_count: {compact_count}\n"
            f"pre_compact_estimated_tokens: {pre_tokens}\n"
            f"created_at: {timestamp}\n"
            "Conversation history before this point was compacted into the following summary."
        ),
    }


def create_compact_summary_message(summary: str) -> dict:
    content = (
        f"{COMPACT_SUMMARY_PREFIX}\n"
        "<compact_summary>\n"
        f"{summary.strip()}\n"
        "</compact_summary>\n\n"
        "Continue the task from this summary. If details from before compaction are missing, "
        "state the gap before relying on them."
    )
    return {"role": "user", "content": content}


def build_summary_request_message(custom_instructions: str | None = None) -> dict:
    return {"role": "user", "content": build_compact_prompt(custom_instructions)}


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _truncate_message_to_tokens(message: dict, tokens: int) -> dict:
    """Approximate token truncation while preserving the chat message shape."""
    text = _message_text(message)
    if tokens <= 0:
        return {"role": "user", "content": ""}
    # The project intentionally uses a tokenizer-free 4 chars/token estimate.
    text = text[: max(1, tokens * 4)]
    return {"role": "user", "content": text}


def build_local_compacted_history(
    messages: list[dict],
    summary: str,
    *,
    max_user_tokens: int = 20_000,
    summary_prefix: str = CODEX_SUMMARY_PREFIX,
) -> list[dict]:
    """Build the Codex local-compaction replacement history.

    Tool/assistant messages are deliberately omitted. Real user messages are
    selected from newest to oldest up to ``max_user_tokens`` and restored in
    chronological order, followed by the summary as a special user message.
    Leading system messages remain instructions for chat-compatible adapters.
    """
    system, _ = split_leading_system_messages(messages)
    selected: list[dict] = []
    remaining = max(0, int(max_user_tokens))
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = _message_text(message)
        if (
            not content
            or is_compact_boundary_message(message)
            or is_compact_summary_message(message)
            or is_codex_summary_message(message)
        ):
            continue
        estimate = rough_token_count_estimation_for_message(message)
        if remaining <= 0:
            break
        if estimate > remaining:
            selected.append(_truncate_message_to_tokens(message, remaining))
            remaining = 0
            break
        selected.append(dict(message))
        remaining -= estimate
    selected.reverse()
    summary_content = f"{summary_prefix}\n{summary.strip() or '(no summary available)'}"
    return [*system, *selected, {"role": "user", "content": summary_content}]


__all__ = [
    "COMPACT_BOUNDARY_PREFIX",
    "COMPACT_ORIGINAL_USER_ISSUE_PREFIX",
    "COMPACT_SUMMARY_PREFIX",
    "CODEX_SUMMARY_PREFIX",
    "is_codex_summary_message",
    "build_summary_request_message",
    "build_local_compacted_history",
    "create_compact_boundary_message",
    "create_original_user_issue_anchor_message",
    "create_compact_summary_message",
    "get_messages_after_last_compact_boundary",
    "is_compact_boundary_message",
    "is_compact_summary_message",
    "is_original_user_issue_anchor_message",
    "messages_for_compaction",
    "original_user_issue_anchor_message",
    "split_leading_system_messages",
]
