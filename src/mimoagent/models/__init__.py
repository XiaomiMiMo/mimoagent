"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import logging
import os
import threading
from dataclasses import dataclass

from mimoagent import Model

# The openai / anthropic SDKs log one httpx INFO line per request ("HTTP
# Request: POST ... 200 OK"), which floods batch logs when the driver
# configures the root logger at INFO. Every protocol module imports this
# package, so silencing here covers chat/anthropic/responses alike.
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class TokenStats:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_creation_tokens


class GlobalModelStats:
    """Aggregated model statistics tracker with optional call limit.

    Not a process singleton: the caller decides the aggregation scope by
    creating an instance and injecting it into models via ``shared_stats``
    (e.g. one per batch run). A model without an injected instance only
    tracks its own ``token_stats`` and is subject to no call limit.
    """

    def __init__(self):
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.call_limit = int(os.getenv("MIMOAGENT_GLOBAL_CALL_LIMIT", "0"))
        if self.call_limit > 0 and not os.getenv("MIMOAGENT_SILENT_STARTUP"):
            print(f"Global call limit: {self.call_limit}")

    def add(self, token_stats) -> None:
        """Add a model call with its token stats, checking limits."""
        with self._lock:
            self._input_tokens += token_stats.input_tokens
            self._output_tokens += token_stats.output_tokens
            self._cache_read_tokens += token_stats.cache_read_tokens
            self._cache_creation_tokens += token_stats.cache_creation_tokens
            self._n_calls += 1
        if 0 < self.call_limit < self._n_calls + 1:
            raise RuntimeError(f"Global call limit exceeded: {self._n_calls + 1}")

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self._cache_read_tokens

    @property
    def cache_creation_tokens(self) -> int:
        return self._cache_creation_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens + self._cache_read_tokens + self._cache_creation_tokens

    @property
    def n_calls(self) -> int:
        return self._n_calls


def get_model(
    input_model_name: str | None = None,
    config: dict | None = None,
    shared_stats: GlobalModelStats | None = None,
) -> Model:
    """Get an initialized model object from any kind of user input or settings.

    ``shared_stats``: optional aggregator the model reports every call into,
    in addition to its own per-instance ``token_stats``.
    """
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    # API key resolution (from env -> config -> None)
    if "model_kwargs" not in config:
        config["model_kwargs"] = {}

    model_class = get_model_class(resolved_model_name, config)
    if shared_stats is not None:
        return model_class(shared_stats=shared_stats, **config)
    return model_class(**config)


def get_model_name(input_model_name: str | None = None, config: dict | None = None) -> str:
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    if from_env := os.getenv("MIMOAGENT_MODEL_NAME"):
        return from_env
    raise ValueError(
        "No model configured: set model.model_name in the yaml config, pass --model, or export MIMOAGENT_MODEL_NAME."
    )


def get_model_class(model_name: str, config: dict | None = None) -> type:
    """Select the model class from the ``protocol`` config field.

    ``protocol`` is one of:

    * ``chat`` (default) — OpenAI chat completions (``/v1/chat/completions``)
    * ``anthropic`` — Anthropic Messages API (``/v1/messages``)
    * ``responses`` — OpenAI Responses API (``/v1/responses``)

    ``model_name`` is the serving name sent to the backend verbatim; it plays
    no role in routing. Blackbox scaffolds never query through the model
    object (it only carries gateway config), so their configs simply omit
    ``protocol``.
    """
    protocol = (config or {}).get("protocol") or "chat"
    if protocol == "chat":
        from mimoagent.models.openai_chat import OpenAIChatModel

        return OpenAIChatModel
    if protocol == "anthropic":
        from mimoagent.models.anthropic import AnthropicModel

        return AnthropicModel
    if protocol == "responses":
        from mimoagent.models.openai_responses import OpenAIResponsesModel

        return OpenAIResponsesModel
    raise ValueError(
        f"model.protocol must be one of 'chat', 'anthropic', 'responses'; got {protocol!r} (model_name={model_name!r})"
    )
