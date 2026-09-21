"""OpenAI chat-completions model (native SDK, ``protocol: chat``).

Requests go straight through the official ``openai`` client, so what hits the
gateway is exactly what the config says. ``model_name`` is the serving name,
sent verbatim. ``model_kwargs.base_url`` / ``api_key`` / ``timeout`` build the
client; everything else in ``model_kwargs`` is forwarded to
``chat.completions.create`` per call. Kwargs the installed SDK version doesn't
know are routed into ``extra_body`` instead of raising, so gateway-specific
params keep working across SDK versions.

Multimodal: messages may carry image (``image_url``), audio
(``input_audio``) and video (``video_url``) parts — the MiMo dialect of the
chat-completions format. Message structure is sent faithfully as-is, tool
messages included: no client-side rewriting. Note that some backends
(some gateways, as of 2026-08) reject media parts under ``role="tool"`` with a
400 — that limit is the backend's to lift, not this client's to paper over.
"""

import inspect
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
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs

logger = logging.getLogger("openai_chat_model")

# Keys in model_kwargs that configure the client, not a single call. Read (not
# popped): blackbox harnesses read ``model.config.model_kwargs["base_url"]`` /
# ``["api_key"]`` after construction, so config must stay intact.
_CLIENT_KWARGS = ("api_key", "base_url", "api_base", "timeout", "max_retries")

# Params the installed SDK accepts on ``chat.completions.create``; anything
# else is tunnelled through ``extra_body``.
_CREATE_PARAMS = frozenset(inspect.signature(openai.resources.chat.completions.Completions.create).parameters) - {
    "self"
}

_NO_RETRY_EXCEPTIONS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
    openai.UnprocessableEntityError,
    KeyboardInterrupt,
)


@dataclass
class OpenAIChatModelConfig:
    model_name: str
    # Per-call kwargs forwarded to ``chat.completions.create`` (temperature,
    # max_tokens, extra_body, ...). ``api_key`` / ``base_url`` / ``api_base`` /
    # ``timeout`` found here configure the client instead.
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    protocol: str = "chat"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 3600.0
    # SDK-internal retries re-send the same prefix while the first generation
    # is still running, forking the session. Keep them off; _query already
    # retries with logging via tenacity. model_kwargs.max_retries overrides.
    max_retries: int = 0


class OpenAIChatModel:
    def __init__(self, *, shared_stats: GlobalModelStats | None = None, **kwargs):
        # Unknown yaml keys (e.g. legacy ``calculate_cost``/``api_key_pool``
        # left in old configs or override-config blobs) are warned about and
        # dropped instead of raising TypeError.
        self.config = OpenAIChatModelConfig(**filter_dataclass_kwargs(OpenAIChatModelConfig, kwargs))
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
        retry=retry_if_not_exception_type(_NO_RETRY_EXCEPTIONS),
    )
    def _query(self, messages: list[dict], **kwargs) -> tuple[TokenStats, dict]:
        call_kwargs = {k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in _CLIENT_KWARGS}
        stream = bool(call_kwargs.pop("stream", False))
        call_kwargs = _route_unknown_kwargs(call_kwargs)
        if stream:
            token_count, payload, finish_reason = self._query_streaming(messages, call_kwargs)
        else:
            response = self.client.chat.completions.create(
                model=self.config.model_name, messages=messages, **call_kwargs
            )
            payload = _build_assistant_payload(response.choices[0].message)
            token_count = _extract_token_stats(getattr(response, "usage", None))
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        if not (payload.get("content") or payload.get("reasoning_content") or payload.get("tool_calls")):
            if _is_length_exhausted(finish_reason):
                return token_count, payload
            raise ValueError("Empty assistant response: content, reasoning_content and tool_calls are all empty.")
        return token_count, payload

    def _query_streaming(self, messages: list[dict], call_kwargs: dict) -> tuple[TokenStats, dict, str | None]:
        # Streaming: aggregate every chunk into a complete payload so the rest of
        # the pipeline (token accounting, message parsing) is identical to the
        # non-streaming path. include_usage makes the final chunk carry usage.
        stream_options = dict(call_kwargs.pop("stream_options", None) or {})
        stream_options.setdefault("include_usage", True)
        stream = self.client.chat.completions.create(
            model=self.config.model_name, messages=messages, stream=True, stream_options=stream_options, **call_kwargs
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        signature_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = None
        finish_reason = None
        for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            if not getattr(chunk, "choices", None):
                continue
            finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            # Some gateways (e.g. deepseek channels) withhold plaintext
            # reasoning and stream an opaque ``reasoning_signature`` reference
            # instead; it must be replayed on the assistant message so the
            # gateway can splice the reasoning back server-side. Dropping it
            # silently severs the model's chain-of-thought every tool round.
            sig = getattr(delta, "reasoning_signature", None)
            if sig:
                signature_parts.append(sig)
            for tc in delta.tool_calls or []:
                slot = tool_calls.setdefault(
                    tc.index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.type:
                    slot["type"] = tc.type
                if tc.function:
                    slot["function"]["name"] += tc.function.name or ""
                    slot["function"]["arguments"] += tc.function.arguments or ""
        payload: dict[str, Any] = {"content": "".join(content_parts) or None}
        if tool_calls:
            payload["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        if reasoning_parts:
            payload["reasoning_content"] = "".join(reasoning_parts)
        if signature_parts:
            payload["reasoning_signature"] = "".join(signature_parts)
        return _extract_token_stats(usage), payload, finish_reason

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


def _is_length_exhausted(finish_reason: Any) -> bool:
    """True when the server stopped for lack of room, not because it was done."""
    return finish_reason == "length"


def _build_assistant_payload(message: Any) -> dict[str, Any]:
    assistant_payload: dict[str, Any] = {"content": message.content}

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        assistant_payload["tool_calls"] = [_normalize_tool_call(call) for call in tool_calls]

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        assistant_payload["reasoning_content"] = reasoning_content

    reasoning_signature = getattr(message, "reasoning_signature", None)
    if reasoning_signature:
        assistant_payload["reasoning_signature"] = reasoning_signature

    return assistant_payload


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    """Convert SDK tool call objects to plain dictionaries."""

    if isinstance(call, dict):
        function_payload = call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {
                "name": getattr(function_payload, "name", None),
                "arguments": getattr(function_payload, "arguments", None),
            }
        return {
            "id": call.get("id"),
            "type": call.get("type"),
            "function": {
                "name": function_payload.get("name"),
                "arguments": function_payload.get("arguments"),
            },
        }

    function_payload = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "type": getattr(call, "type", None),
        "function": {
            "name": getattr(function_payload, "name", None) if function_payload else None,
            "arguments": getattr(function_payload, "arguments", None) if function_payload else None,
        },
    }


def _extract_token_stats(usage: Any) -> TokenStats:
    """Extract token usage statistics from a chat-completions ``usage`` object."""
    try:
        if usage is None:
            return TokenStats()

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cache_read_tokens = 0
        cache_creation_tokens = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            if isinstance(details, dict):
                cache_read_tokens = details.get("cached_tokens", 0) or 0
                cache_creation_tokens = details.get("cached_tokens_created", 0) or 0
            else:
                cache_read_tokens = getattr(details, "cached_tokens", 0) or 0

        # Anthropic-style field names surfaced by some gateways.
        if getattr(usage, "cache_read_input_tokens", None):
            cache_read_tokens = usage.cache_read_input_tokens
        if getattr(usage, "cache_creation_input_tokens", None):
            cache_creation_tokens = usage.cache_creation_input_tokens

        return TokenStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    except Exception as e:  # pragma: no cover - defensive logging only
        logger.debug(f"Failed to extract token stats: {e}")
        return TokenStats()
