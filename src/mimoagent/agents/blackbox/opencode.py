"""Upstream opencode CLI as a MiMo Agent blackbox harness.

The upstream of MiMo-Code (see :mod:`mimocode` for the fork's adapter). The
mechanics are near-identical — ``OPENCODE_CONFIG_CONTENT`` inline provider
config, ``opencode run --format json`` NDJSON events, stdin prompt — with
three upstream-only quirks this adapter handles:

* ``max_tokens`` is silently capped at 32000 unless
  ``OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`` raises the ceiling;
* every session fires one extra LLM call to generate a title unless
  ``--title`` is passed;
* an unreachable gateway is retried forever (no exit), so ``run_timeout``
  is the only backstop.

The provider always speaks ``@ai-sdk/anthropic`` (``/v1/messages``): thinking
signatures replay across turns and cache_control breakpoints (system + last
two messages) apply to custom providers, verified against 1.18.18. xhigh/max
effort lands as a handwritten model variant selected via ``--variant``
(opencode only auto-generates variants for models.dev catalog entries).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gateway"


@dataclass
class OpencodeAgentConfig(HarnessAgentConfig):
    version: str = "1.18.18"
    # Model context window / max output tokens (config ``limit`` block).
    # ``context`` drives auto-compaction; ``output`` above 32000 is unlocked
    # via OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX automatically.
    context_limit: int | None = None
    output_limit: int | None = None
    # Extended thinking, budget style: model options.thinking.budgetTokens.
    thinking_budget: int | None = None
    # Adaptive thinking with a fixed effort ("xhigh" | "max" | ...): lands as
    # a handwritten model variant selected via --variant. Mutually exclusive
    # with thinking_budget.
    thinking_effort: str | None = None
    # Subscription-style Claude gateway channels reject any request whose tools
    # include the name "todowrite" (400 "Third-party apps..."), so it is
    # stripped from every built-in agent by default.
    todo_tool: bool = False


class OpencodeAgent(HarnessAgent):
    """Runs upstream opencode inside the pod under the MiMo Agent protocol."""

    NAME = "opencode"
    ERROR_STATUS = "OpencodeError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-opencode-logs/opencode.txt"]

    def __init__(self, model, env, *, config_class=OpencodeAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    def _provider_config(self) -> str:
        if self.config.thinking_budget and self.config.thinking_effort:
            raise RuntimeError(
                "[opencode] thinking_budget and thinking_effort are mutually "
                "exclusive (budget-based vs adaptive-effort thinking)"
            )
        model = self._resolved_model
        model_entry: dict[str, Any] = {"name": model}
        limit = {
            k: v
            for k, v in (
                ("context", self.config.context_limit),
                ("output", self.config.output_limit),
            )
            if v
        }
        if limit:
            model_entry["limit"] = limit
        if self.config.thinking_budget:
            model_entry["options"] = {
                "thinking": {
                    "type": "enabled",
                    "budgetTokens": self.config.thinking_budget,
                }
            }
        if self.config.thinking_effort:
            model_entry["variants"] = {
                self.config.thinking_effort: {
                    "thinking": {"type": "adaptive"},
                    "effort": self.config.thinking_effort,
                }
            }
        config = {
            "model": f"{_PROVIDER_ID}/{model}",
            "provider": {
                _PROVIDER_ID: {
                    "npm": "@ai-sdk/anthropic",
                    "name": "Gateway",
                    "options": {
                        "baseURL": self._gateway_base_url(v1=True),
                        "apiKey": self._api_key,
                    },
                    "models": {model: model_entry},
                }
            },
            "permission": "allow",
            "share": "disabled",
            "autoupdate": False,
            # lsp/formatter are coding capability, and snapshot is opencode's
            # own undo history — all three help the model do the task.
            "snapshot": True,
            "lsp": True,
            "formatter": True,
        }
        # Per-agent is the only wire-effective switch (root "tools" is a
        # dead deprecated key); these three are all the built-in agents.
        # Nothing is dropped for being unreachable from a pod (webfetch stays);
        # todowrite is the one exception, and not on capability grounds: it
        # trips subscription-style gateway channels' third-party fingerprint
        # (400) by exact name.
        disabled: dict[str, bool] = {}
        if not self.config.todo_tool:
            disabled["todowrite"] = False
        config["agent"] = {name: {"tools": dict(disabled)} for name in ("build", "plan", "general")}
        return json.dumps(config, ensure_ascii=False)

    def _harness_env(self) -> dict[str, str]:
        env = {
            "OPENCODE_CONFIG_CONTENT": self._provider_config(),
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_SHARE": "1",
            # The repo under test may carry its own opencode.json — never let
            # it override the injected provider/permission config.
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            # Sessions land in a SQLite db under XDG data — keep it hermetic
            # under the (pod-local) logs dir; never a network mount (WAL).
            "XDG_DATA_HOME": f"{self._logs_dir}/xdg/data",
            "XDG_CONFIG_HOME": f"{self._logs_dir}/xdg/config",
            "XDG_CACHE_HOME": f"{self._logs_dir}/xdg/cache",
            "XDG_STATE_HOME": f"{self._logs_dir}/xdg/state",
        }
        if self.config.output_limit and self.config.output_limit > 32000:
            env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(self.config.output_limit)
        return env

    def _command_parts(self, task: str) -> list[str]:
        parts = [
            "opencode",
            "run",
            "--format json",
            "--dangerously-skip-permissions",
            # Suppress the per-session title-generation LLM call.
            "--title mimo-rollout",
            f"-m {shlex.quote(f'{_PROVIDER_ID}/{self._resolved_model}')}",
        ]
        if self._turns > 0:
            parts.append("--continue")
        if self.config.thinking_effort:
            parts.append(f"--variant {shlex.quote(self.config.thinking_effort)}")
        return parts

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """Same event shapes as mimocode: ``text`` parts, ``step_finish``
        token blocks, ``error`` events."""
        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = ""
        error_text = ""
        per_call: list[dict[str, int]] = []
        for ev in events:
            etype = ev.get("type")
            if etype == "text":
                text = ((ev.get("part") or {}).get("text") or "").strip()
                if text:
                    result_text = text
            elif etype == "step_finish":
                tokens = (ev.get("part") or {}).get("tokens") or {}
                if isinstance(tokens, dict):
                    cache = tokens.get("cache") or {}
                    per_call.append(
                        {
                            "input": tokens.get("input", 0),
                            "output": tokens.get("output", 0),
                            "cache_read": cache.get("read", 0),
                            "cache_write": cache.get("write", 0),
                        }
                    )
            elif etype == "error":
                error_text = json.dumps(ev, ensure_ascii=False)
        if error_text:
            # any error event fails the turn, even alongside partial text
            status = self.ERROR_STATUS
            result_text = result_text or error_text
        self._fold_usage(per_call)
        if not result_text:
            # no text event at rc==0 = broken/empty stream, not success
            status = self.ERROR_STATUS
            result_text = f"(no agent message; opencode rc={rc}, {len(events)} events)"
        return result_text, status
