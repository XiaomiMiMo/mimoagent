"""Rough, tokenizer-free message size estimation.

Used to drive the context-usage signal shown to the model (see the
``<context_usage>`` footer in ``CCAgent._format_observation``) and the
pre-compaction token count recorded on the compact boundary. It is a cheap
~4-bytes/token heuristic — deliberately not a real tokenizer — because it runs
on the whole history every step and only needs to be good enough to tell the
model "the window is getting full".
"""

import json
from collections.abc import Mapping, Sequence
from math import floor
from typing import Any

MEDIA_BLOCK_TOKEN_ESTIMATE = 2000


def rough_token_count_estimation(content: str, bytes_per_token: float = 4) -> int:
    """Estimate token count using a simple characters-per-token heuristic."""
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be greater than 0")
    return floor(len(content) / bytes_per_token + 0.5)


def rough_token_count_estimation_for_messages(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate the token footprint of a whole ``messages`` list."""
    return sum(rough_token_count_estimation_for_message(message) for message in messages)


def rough_token_count_estimation_for_message(message: Mapping[str, Any]) -> int:
    message_type = message.get("type") or message.get("role")
    total_tokens = 0

    if message_type in {"assistant", "user", "system", "tool", "message"} and message.get("content"):
        total_tokens += rough_token_count_estimation_for_content(message.get("content"))

    # Native Responses items do not carry chat ``role``/``content`` fields for
    # function calls and their outputs. Count their serialized arguments so
    # Codex native compaction still triggers on tool-heavy histories.
    if message_type in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
        total_tokens += rough_token_count_estimation(_json_stringify(message))

    if message.get("tool_calls"):
        total_tokens += rough_token_count_estimation(_json_stringify(message.get("tool_calls")))

    if message.get("reasoning_content"):
        total_tokens += rough_token_count_estimation_for_content(message.get("reasoning_content"))

    if message.get("thinking_blocks"):
        total_tokens += rough_token_count_estimation_for_content(message.get("thinking_blocks"))

    # Native Responses reasoning items may carry the replayable reasoning seal
    # directly on the item. It is resent on every later turn, so include its
    # serialized size when deciding whether Codex history needs compaction.
    if message.get("encrypted_content"):
        total_tokens += rough_token_count_estimation(str(message.get("encrypted_content")))

    return total_tokens


def rough_token_count_estimation_for_content(content: Any) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return rough_token_count_estimation(content)
    if _is_sequence_but_not_text(content):
        return sum(rough_token_count_estimation_for_block(block) for block in content)
    return rough_token_count_estimation_for_block(content)


def rough_token_count_estimation_for_block(block: Any) -> int:
    if not block:
        return 0
    if isinstance(block, str):
        return rough_token_count_estimation(block)
    if not isinstance(block, Mapping):
        return rough_token_count_estimation(_json_stringify(block))

    block_type = block.get("type")
    if block_type in {"text", "input_text", "output_text"}:
        return rough_token_count_estimation(str(block.get("text") or ""))
    if block_type in {"image", "image_url", "input_image", "document"}:
        return MEDIA_BLOCK_TOKEN_ESTIMATE
    if block_type == "tool_result":
        return rough_token_count_estimation_for_content(block.get("content"))
    if block_type == "tool_use":
        return rough_token_count_estimation(str(block.get("name") or "") + _json_stringify(block.get("input") or {}))
    if block_type == "thinking":
        return rough_token_count_estimation(str(block.get("thinking") or ""))
    if block_type == "redacted_thinking":
        return rough_token_count_estimation(str(block.get("data") or ""))

    return rough_token_count_estimation(_json_stringify(block))


def _is_sequence_but_not_text(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _json_stringify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
