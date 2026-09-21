"""Anthropic Messages API model (native SDK).

Agents speak chat-completions format (``messages`` with ``tool_calls`` /
``role="tool"``); this model translates both directions:

* request:  chat messages -> ``system`` blocks + alternating user/assistant
  turns with text / image / tool_use / tool_result content blocks
* response: content blocks -> ``{content, tool_calls, reasoning_content,
  thinking_blocks}``

The raw content blocks of each assistant turn are stashed on the assistant
message under ``anthropic_blocks`` and replayed verbatim on later requests, so
block order (interleaved thinking), multiple text blocks and block types this
module doesn't parse all survive multi-turn conversations without lossy
reconstruction. Assistant messages lacking that key (e.g. replayed
conversations produced elsewhere) are reconstructed from ``thinking_blocks`` +
``content`` + ``tool_calls``. ``thinking_blocks`` is still emitted alongside —
token accounting (``cal_token``) and existing trajectories consume it.

Multimodal: image parts in user messages become ``image`` blocks; image parts
in tool messages ride inside the ``tool_result`` block, which the Messages API
supports natively. Audio/video are NOT supported on this protocol (the gateway
silently drops them) — such parts degrade to an explicit text placeholder; use
``protocol: chat`` for audio/video.

Streaming: ``stream: true`` runs the same request over SSE and lets the SDK
accumulate events back into the identical final Message — same payload, same
stats. Useful for long generations (proxies/LBs that kill idle connections;
the SDK itself refuses non-streaming requests it estimates at >10 min).

Prompt caching: ``cache_control`` breakpoints are placed on the last content
block of the two most recent user turns (tool results convert to user turns),
so the whole conversation prefix — system prompt and tools included — is
covered by the earlier breakpoint.

Additionally (only when ``config.set_user_id=True``): some gateways load-balance
across several upstream accounts. Even with ``cache_control`` set, requests of
one conversation then land on different accounts unless ``metadata.user_id`` is
pinned, and the prompt cache never hits. With this on, every model instance
(= one rollout of one instance) gets a fixed ``user_id`` reused across all its
turns, so the gateway keeps the conversation on one account.
"""

import inspect
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mimoagent.models import GlobalModelStats, TokenStats
from mimoagent.models.utils.content import media_part_kind, split_data_url
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs

logger = logging.getLogger("anthropic_model")

# Assistant messages carry their raw Messages-API content blocks under this
# key (mirrors ``responses_items`` on the responses protocol): the chat-format
# fields are a projection for logs/training, the wire replays the raw blocks.
ANTHROPIC_BLOCKS_KEY = "anthropic_blocks"

# Keys in model_kwargs that configure the client, not a single call. Read (not
# popped): blackbox harnesses read ``model.config.model_kwargs["base_url"]`` /
# ``["api_key"]`` after construction, so config must stay intact.
_CLIENT_KWARGS = ("api_key", "base_url", "api_base", "timeout")

# Params the installed SDK accepts on ``messages.create``; anything else is
# tunnelled through ``extra_body``.
_CREATE_PARAMS = frozenset(inspect.signature(anthropic.resources.messages.Messages.create).parameters) - {"self"}

_NO_RETRY_EXCEPTIONS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
    anthropic.UnprocessableEntityError,
    KeyboardInterrupt,
)


@dataclass
class AnthropicModelConfig:
    model_name: str
    # Per-call kwargs forwarded to ``messages.create`` (max_tokens, thinking,
    # temperature, ...). ``api_key`` / ``base_url`` / ``api_base`` / ``timeout``
    # found here configure the client instead.
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    protocol: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 600.0
    # ``max_tokens`` is mandatory on the Messages API; used when model_kwargs
    # don't set it.
    default_max_tokens: int = 8192
    # See module docstring.
    set_user_id: bool = False


class AnthropicModel:
    def __init__(self, *, shared_stats: GlobalModelStats | None = None, **kwargs):
        # Unknown yaml keys (e.g. legacy ``calculate_cost``) are warned about
        # and dropped instead of raising TypeError.
        self.config = AnthropicModelConfig(**filter_dataclass_kwargs(AnthropicModelConfig, kwargs))
        # Caller-scoped aggregator (e.g. one per batch run). Deliberately not a
        # module-level singleton: multiple agents may share one process, and a
        # global counter would couple their stats and its call limit.
        self.shared_stats = shared_stats
        self.n_calls = 0
        self.token_stats = TokenStats()

        model_kwargs = self.config.model_kwargs
        api_key = model_kwargs.get("api_key") or self.config.api_key or os.getenv("ANTHROPIC_API_KEY") or "EMPTY"
        base_url = model_kwargs.get("base_url") or model_kwargs.get("api_base") or self.config.base_url
        if base_url:
            # The SDK appends /v1/messages itself; configs routinely share an
            # openai-style ".../v1" base_url, so strip the suffix.
            base_url = base_url.rstrip("/").removesuffix("/v1")
        timeout = model_kwargs.get("timeout") or self.config.timeout
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)

        self._user_metadata: dict[str, str] = {}
        if self.config.set_user_id:
            self._user_metadata = {"user_id": "mswea-cache-" + uuid.uuid4().hex}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(_NO_RETRY_EXCEPTIONS),
    )
    def _query(self, messages: list[dict], **kwargs) -> tuple[TokenStats, dict]:
        call_kwargs = {k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in _CLIENT_KWARGS}
        stream = bool(call_kwargs.pop("stream", False))
        call_kwargs.setdefault("max_tokens", self.config.default_max_tokens)
        if "tools" in call_kwargs:
            call_kwargs["tools"] = [_convert_tool_definition(t) for t in (call_kwargs["tools"] or [])]
        if "tool_choice" in call_kwargs:
            tool_choice = _convert_tool_choice(call_kwargs.pop("tool_choice"))
            if tool_choice is not None:
                call_kwargs["tool_choice"] = tool_choice
        if self._user_metadata:
            call_kwargs["metadata"] = self._user_metadata | dict(call_kwargs.get("metadata") or {})

        system_blocks, converted = _convert_messages(messages)
        _apply_cache_control(system_blocks, converted)
        if system_blocks:
            call_kwargs.setdefault("system", system_blocks)
        call_kwargs = _route_unknown_kwargs(call_kwargs)

        if stream:
            # SSE + SDK-side accumulation: ``get_final_message()`` rebuilds the
            # exact non-streaming Message (thinking signatures, tool_use inputs,
            # usage included), so everything downstream is shared. The helper's
            # signature matches ``create`` minus ``stream`` itself.
            with self.client.messages.stream(
                model=self.config.model_name, messages=converted, **call_kwargs
            ) as event_stream:
                response = event_stream.get_final_message()
        else:
            response = self.client.messages.create(model=self.config.model_name, messages=converted, **call_kwargs)
        payload = _build_assistant_payload(response)
        if not (payload.get("content") or payload.get("reasoning_content") or payload.get("tool_calls")):
            raise ValueError("Empty assistant response: content, reasoning_content and tool_calls are all empty.")
        return _extract_token_stats(response), payload

    def query(self, messages: list[dict], **kwargs) -> dict:
        token_count, payload = self._query(messages, **kwargs)
        self._record_call(token_count)
        return payload

    def _record_call(self, token_count: TokenStats) -> None:
        self.n_calls += 1
        self.token_stats.input_tokens += token_count.input_tokens
        self.token_stats.output_tokens += token_count.output_tokens
        self.token_stats.cache_read_tokens += token_count.cache_read_tokens
        self.token_stats.cache_creation_tokens += token_count.cache_creation_tokens
        if self.shared_stats is not None:
            self.shared_stats.add(token_count)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "input_tokens": self.token_stats.input_tokens,
            "output_tokens": self.token_stats.output_tokens,
            "cache_read_tokens": self.token_stats.cache_read_tokens,
            "cache_creation_tokens": self.token_stats.cache_creation_tokens,
            "total_tokens": self.token_stats.total_tokens,
        }


def _route_unknown_kwargs(call_kwargs: dict) -> dict:
    """Tunnel kwargs the installed SDK doesn't know through ``extra_body``.

    Explicit ``extra_body`` entries win over tunnelled ones on key conflict.
    """
    known = {k: v for k, v in call_kwargs.items() if k in _CREATE_PARAMS}
    unknown = {k: v for k, v in call_kwargs.items() if k not in _CREATE_PARAMS}
    if unknown:
        known["extra_body"] = unknown | dict(known.get("extra_body") or {})
    return known


def _convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert chat-completions messages to (system blocks, Messages API turns)."""
    system_blocks: list[dict] = []
    turns: list[dict] = []

    def emit(role: str, blocks: list[dict]) -> None:
        # The Messages API wants alternating roles; consecutive same-role
        # messages (e.g. parallel tool results, or a tool result followed by a
        # user nudge) merge into one turn.
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_blocks.extend(_content_to_blocks(msg.get("content")))
        elif role == "user":
            blocks = _content_to_blocks(msg.get("content"))
            if blocks:
                emit("user", blocks)
        elif role == "assistant":
            if raw_blocks := msg.get(ANTHROPIC_BLOCKS_KEY):
                # Verbatim replay (thinking blocks included) — order, multiple
                # text blocks and unparsed block types are all preserved.
                emit("assistant", [dict(b) for b in raw_blocks])
                continue
            blocks = [dict(b) for b in msg.get("thinking_blocks") or []]
            blocks.extend(_content_to_blocks(msg.get("content")))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": fn.get("name"),
                        "input": _parse_tool_arguments(fn.get("arguments")),
                    }
                )
            if blocks:
                emit("assistant", blocks)
        elif role == "tool":
            blocks = _content_to_blocks(msg.get("content"))
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": msg.get("tool_call_id")}
            if blocks:
                result["content"] = blocks
            emit("user", [result])
        else:
            raise ValueError(f"Unsupported message role for Anthropic Messages API: {role!r}")
    return system_blocks, turns


def _content_to_blocks(content: Any) -> list[dict]:
    """Chat-completions content (str or parts) -> Messages API content blocks.

    Images convert to ``image`` blocks. Audio/video have no Messages API block
    type — gateways tend to silently drop unknown block shapes (the
    model never sees them), so they degrade to an explicit text placeholder
    instead: the model knows an attachment exists that it cannot perceive.
    Use ``protocol: chat`` for audio/video input.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks: list[dict] = []
    for piece in content:
        if not isinstance(piece, dict):
            blocks.append({"type": "text", "text": str(piece)})
            continue
        piece_type = piece.get("type")
        if piece_type == "image_url":
            image_url = piece.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not url:
                continue
            if data := split_data_url(url):
                media_type, b64 = data
                source = {"type": "base64", "media_type": media_type, "data": b64}
            else:
                source = {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
        elif media_part_kind(piece) in ("audio", "video"):
            kind = media_part_kind(piece)
            blocks.append(
                {
                    "type": "text",
                    "text": f"[{kind} attachment omitted: the anthropic protocol endpoint does not "
                    f'accept {kind} input; use protocol "chat" to consume it]',
                }
            )
        elif piece_type == "text":
            if piece.get("text"):
                blocks.append({k: v for k, v in piece.items() if k in ("type", "text", "cache_control")})
        else:
            # Already a Messages API block (image, tool_result replay, ...).
            blocks.append(dict(piece))
    return blocks


def _parse_tool_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _apply_cache_control(system_blocks: list[dict], turns: list[dict], breakpoints: int = 2) -> None:
    """Place ephemeral breakpoints on the last block of the last N user turns."""
    for blocks in [system_blocks] + [t["content"] for t in turns]:
        for block in blocks:
            block.pop("cache_control", None)
    n_tagged = 0
    for turn in reversed(turns):
        if n_tagged >= breakpoints:
            break
        if turn["role"] != "user" or not turn["content"]:
            continue
        turn["content"][-1]["cache_control"] = {"type": "ephemeral"}
        n_tagged += 1


def _convert_tool_definition(tool: dict) -> dict:
    """Chat-completions function tool -> Messages API tool."""
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"] or {}
        return {
            "name": fn.get("name"),
            "description": fn.get("description"),
            "input_schema": fn.get("parameters"),
        }
    return tool


def _convert_tool_choice(tool_choice: Any) -> dict | None:
    """Chat-completions tool_choice -> Messages API tool_choice (None = drop)."""
    if isinstance(tool_choice, str):
        return {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}.get(tool_choice)
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return {"type": "tool", "name": (tool_choice.get("function") or {}).get("name")}
        return tool_choice
    return None


def _build_assistant_payload(response: Any) -> dict[str, Any]:
    texts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_parts: list[str] = []
    thinking_blocks: list[dict] = []
    raw_blocks: list[dict] = []

    for block in response.content or []:
        raw_blocks.append(block.model_dump(exclude_none=True))
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input or {}, ensure_ascii=False)},
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            thinking_blocks.append(block.model_dump(exclude_none=True))
            if block_type == "thinking" and getattr(block, "thinking", None):
                reasoning_parts.append(block.thinking)

    payload: dict[str, Any] = {"content": "".join(texts)}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if reasoning_parts:
        payload["reasoning_content"] = "\n".join(reasoning_parts)
    if thinking_blocks:
        payload["thinking_blocks"] = thinking_blocks
    if raw_blocks:
        payload[ANTHROPIC_BLOCKS_KEY] = raw_blocks
    return payload


def _extract_token_stats(response: Any) -> TokenStats:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenStats()
        return TokenStats(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
    except Exception as e:  # pragma: no cover - defensive logging only
        logger.debug(f"Failed to extract token stats: {e}")
        return TokenStats()
