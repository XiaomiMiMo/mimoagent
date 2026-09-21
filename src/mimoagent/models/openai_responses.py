"""OpenAI Responses API model (https://developers.openai.com/api/reference/responses/overview).

Agents speak chat-completions format (``messages`` with ``tool_calls`` /
``role="tool"``); this model translates both directions:

* request:  chat messages -> ``instructions`` + ``input`` items
* response: output items  -> ``{content, tool_calls, reasoning_content}``

The raw output items of each assistant turn are stashed on the assistant
message under ``responses_items`` and replayed verbatim on later turns, so
reasoning items (including ``encrypted_content``) survive multi-turn function
calling without lossy reconstruction. Assistant messages lacking that key
(e.g. replayed conversations produced elsewhere) are reconstructed from
``content`` + ``tool_calls``.

Note (some gateway / Codex backends): if a request carries no ``instructions``,
the backend may inject its own default system prompt. We therefore map the
first system message to ``instructions`` instead of an input item.

Streaming: ``stream: true`` consumes the SSE event stream and takes the full
``Response`` object off the terminal ``response.completed`` / ``incomplete``
event — same payload and stats as non-streaming, no delta accumulation.

Multimodal: image parts (``{"type": "image_url", ...}``) in user messages
become ``input_image`` items; image parts in tool messages ride inside the
``function_call_output`` item's content list. Audio/video are NOT supported by
the gateway on this protocol (explicit 400) — such parts degrade to a text
placeholder; use ``protocol: chat`` for audio/video.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mimoagent.models import GlobalModelStats, TokenStats
from mimoagent.models.utils.content import content_media_parts, content_text, media_part_kind
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs

logger = logging.getLogger("openai_responses_model")

RESPONSES_ITEMS_KEY = "responses_items"


@dataclass
class OpenAIResponsesModelConfig:
    model_name: str
    # Per-call kwargs forwarded to ``client.responses.create`` (temperature,
    # max_output_tokens, reasoning, include, ...). ``api_key`` / ``api_base`` /
    # ``base_url`` found here configure the client instead, so existing
    # chat-style config blobs keep working.
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    protocol: str = "responses"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 3600.0
    max_retries: int = 0


# Keys in model_kwargs that configure the client, not a single call. Read (not
# popped) so ``model.config.model_kwargs`` stays intact for callers that
# inspect it after construction (e.g. blackbox harnesses).
_CLIENT_KWARGS = ("api_key", "base_url", "api_base", "timeout", "max_retries")


class OpenAIResponsesModel:
    def __init__(self, *, shared_stats: GlobalModelStats | None = None, **kwargs):
        self.config = OpenAIResponsesModelConfig(**filter_dataclass_kwargs(OpenAIResponsesModelConfig, kwargs))
        # Caller-scoped aggregator (e.g. one per batch run). Deliberately not a
        # module-level singleton: multiple agents may share one process, and a
        # global counter would couple their stats and its call limit.
        self.shared_stats = shared_stats
        self.n_calls = 0
        self.token_stats = TokenStats()

        model_kwargs = self.config.model_kwargs
        api_key = model_kwargs.get("api_key") or self.config.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base_url = model_kwargs.get("base_url") or model_kwargs.get("api_base") or self.config.base_url
        timeout = model_kwargs.get("timeout") or self.config.timeout
        max_retries = model_kwargs.get("max_retries")
        if max_retries is None:
            max_retries = self.config.max_retries
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=int(max_retries))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
                openai.BadRequestError,
                openai.UnprocessableEntityError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        call_kwargs = {k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in _CLIENT_KWARGS}
        stream = bool(call_kwargs.pop("stream", False))
        if "tools" in call_kwargs:
            call_kwargs["tools"] = [_convert_tool_definition(t) for t in (call_kwargs["tools"] or [])]
        instructions, input_items = _convert_messages(messages)
        if instructions is not None:
            call_kwargs.setdefault("instructions", instructions)
        call_kwargs.setdefault("store", False)
        if stream:
            response = self._stream_final_response(input_items, call_kwargs)
        else:
            response = self.client.responses.create(model=self.config.model_name, input=input_items, **call_kwargs)
        payload = _build_assistant_payload(response)
        if not (payload.get("content") or payload.get("reasoning_content") or payload.get("tool_calls")):
            if _is_length_exhausted(response):
                return response, payload
            raise ValueError("Empty assistant response: content, reasoning_content and tool_calls are all empty.")
        return response, payload

    def _stream_final_response(self, input_items: list[dict], call_kwargs: dict) -> Any:
        """SSE streaming: terminal events carry the complete ``Response`` object
        (output items + usage), so no client-side delta accumulation is needed.
        ``incomplete`` (e.g. max_output_tokens hit) still has usable output;
        ``failed``/``error`` raises so the retry decorator takes over."""
        events = self.client.responses.create(
            model=self.config.model_name, input=input_items, stream=True, **call_kwargs
        )
        final = None
        for event in events:
            event_type = getattr(event, "type", None)
            if event_type in ("response.completed", "response.incomplete"):
                final = event.response
            elif event_type in ("response.failed", "error"):
                raise ValueError(f"Responses stream reported failure: {event}")
        if final is None:
            raise ValueError("Responses stream ended without a terminal response event.")
        return final

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response, payload = self._query(messages, **kwargs)
        self._record_call(_extract_token_stats(response))
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


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert chat-completions messages to (instructions, Responses input items)."""
    instructions: str | None = None
    items: list[dict] = []
    custom_call_ids: set[str] = set()
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            if instructions is None:
                instructions = content_text(msg.get("content"))
            else:
                items.append({"role": "system", "content": content_text(msg.get("content"))})
        elif role == "user":
            items.append({"role": "user", "content": _convert_input_content(msg.get("content"))})
        elif role == "assistant":
            raw_items = msg.get(RESPONSES_ITEMS_KEY)
            if raw_items:
                items.extend(raw_items)
                custom_call_ids.update(
                    item.get("call_id")
                    for item in raw_items
                    if item.get("type") == "custom_tool_call" and item.get("call_id")
                )
            else:
                content = content_text(msg.get("content"))
                if content:
                    items.append({"role": "assistant", "content": content})
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    call_id = call.get("id")
                    if call.get("type") == "custom":
                        items.append(
                            {
                                "type": "custom_tool_call",
                                "call_id": call_id,
                                "name": fn.get("name"),
                                "input": fn.get("arguments") or "",
                            }
                        )
                        if call_id:
                            custom_call_ids.add(call_id)
                    else:
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn.get("name"),
                                "arguments": fn.get("arguments") or "{}",
                            }
                        )
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            item = {
                "type": "custom_tool_call_output" if call_id in custom_call_ids else "function_call_output",
                "call_id": call_id,
                "output": _convert_tool_output(msg.get("content")),
            }
            if msg.get("code_mode_notification") and item["type"] == "custom_tool_call_output":
                item["name"] = msg.get("name") or "exec"
            items.append(item)
        else:
            raise ValueError(f"Unsupported message role for Responses API: {role!r}")
    return instructions, items


def _convert_input_content(content: Any) -> Any:
    """Chat-completions content -> Responses input content (str or part list).

    Images convert to ``input_image``. The gateway rejects audio/video input
    items outright (400 ``responses_feature_not_supported``), so those parts
    degrade to an explicit text placeholder; use ``protocol: chat`` for them.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[dict] = []
    for piece in content:
        if not isinstance(piece, dict):
            parts.append({"type": "input_text", "text": str(piece)})
            continue
        piece_type = piece.get("type")
        if piece_type == "image_url":
            image_url = piece.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if url:
                part = {"type": "input_image", "image_url": url}
                if isinstance(image_url, dict) and image_url.get("detail"):
                    part["detail"] = image_url["detail"]
                parts.append(part)
        elif media_part_kind(piece) in ("audio", "video"):
            kind = media_part_kind(piece)
            parts.append(
                {
                    "type": "input_text",
                    "text": f"[{kind} attachment omitted: the responses protocol endpoint does not "
                    f'accept {kind} input; use protocol "chat" to consume it]',
                }
            )
        elif piece_type == "text":
            parts.append({"type": "input_text", "text": piece.get("text") or ""})
        else:
            # Already a Responses input part (input_text, input_image, ...).
            parts.append(dict(piece))
    return parts


def _convert_tool_output(content: Any) -> Any:
    """Tool-result content -> ``function_call_output.output`` (str, or parts when media ride along)."""
    if content_media_parts(content):
        return _convert_input_content(content)
    return content_text(content)


def _convert_tool_definition(tool: dict) -> dict:
    """Chat-completions function tool -> Responses function tool (flat)."""
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"] or {}
        return {
            "type": "function",
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters"),
        }
    return tool


def _is_length_exhausted(response: Any) -> bool:
    """True when the Responses object reports it stopped for lack of room."""
    if getattr(response, "status", None) != "incomplete":
        return False
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return reason == "max_output_tokens"


def _build_assistant_payload(response: Any) -> dict[str, Any]:
    texts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_parts: list[str] = []
    raw_items: list[dict] = []

    for item in response.output or []:
        raw_items.append(item.model_dump(exclude_none=True))
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text":
                    texts.append(part.text)
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )
        elif item_type == "custom_tool_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "custom",
                    "function": {"name": item.name, "arguments": item.input},
                }
            )
        elif item_type == "reasoning":
            # Reasoning text may arrive as ``summary`` (OpenAI) or as
            # ``content`` with ``reasoning_text`` parts (vLLM and some gateways).
            for summary in getattr(item, "summary", None) or []:
                if getattr(summary, "text", None):
                    reasoning_parts.append(summary.text)
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "reasoning_text" and getattr(part, "text", None):
                    reasoning_parts.append(part.text)

    payload: dict[str, Any] = {"content": "".join(texts)}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if reasoning_parts:
        payload["reasoning_content"] = "\n".join(reasoning_parts)
    if raw_items:
        payload[RESPONSES_ITEMS_KEY] = raw_items
    return payload


def _extract_token_stats(response: Any) -> TokenStats:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenStats()
        input_details = getattr(usage, "input_tokens_details", None)
        cache_read = getattr(input_details, "cached_tokens", 0) or 0
        # Non-standard field returned by some gateways.
        cache_creation = 0
        extra = getattr(input_details, "model_extra", None) or {}
        if isinstance(extra, dict):
            cache_creation = extra.get("cache_write_tokens", 0) or 0
        return TokenStats(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )
    except Exception as e:  # pragma: no cover - defensive logging only
        logger.debug(f"Failed to extract token stats: {e}")
        return TokenStats()
