"""OpenClaw CLI as a MiMo Agent blackbox harness.

Node app (payload ships a pinned node 24 + the npm tree with prod deps
installed locally). ``openclaw agent --local`` runs one agent turn fully
in-process — no gateway server, channels, cron or update check ever start on
this path. The custom provider speaks ``api: "anthropic-messages"``
(``baseUrl`` without ``/v1`` → ``POST .../v1/messages``, x-api-key auth).
Verified on 2026.7.1-2:

* historic thinking blocks replay with signatures whenever thinking is on;
* cache_control (ephemeral) lands on the stable system prefix + last user
  message (``params.cacheRetention: "none"`` would disable it);
* exec approvals default to full access (sandbox off), so yolo is the
  natural state; the tool surface is still slimmed to the coding profile;
* stock whitelists predate claude-opus-5 (thinking would clamp to
  budget_tokens ≤ maxTokens-1024) — the payload build patches them so
  ``thinking: xhigh|max`` reaches the wire as adaptive +
  ``output_config.effort``. Two more build patches go with it: the Anthropic
  transport otherwise clamps the declared model ``maxTokens`` to 32000 for
  anything but Claude Sonnet 5 (``OPENCLAW_USE_MODEL_MAX_TOKENS=1`` honours
  it), and ``shouldPreserveThinkingBlocks()`` only recognises new Claude
  families as ``claude-[5-9]`` — ``claude-opus-5`` misses, so every prior
  turn's signed thinking was dropped, costing both replay fidelity and the
  prompt cache. See harness_payloads/build_harness_payloads.sh.

Output is a single JSON envelope on stdout: ``payloads[].text`` is the reply,
``meta.agentMeta.usage`` carries the turn's token totals. Multi-turn reuses
one ``--session-key``. The user message gets a timestamp prefix injected by
openclaw itself — that is its normal message shaping, not ours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gw"

# Denied for a wire reason, not a capability one: with the ``sessions_*`` family
# (or the two memory tools) on the wire, subscription-style gateway channels 400
# the whole request as a third-party app. Verified  by bisecting the
# captured body: coding base + ``subagents`` + goals + plan + cron + web = 16
# tools passes, and adding sessions_spawn or memory_* is what tips it over.
# ``subagents`` — the actual subagent tool — stays.
_DENY_TOOLS = [
    "sessions_spawn",
    "sessions_send",
    "sessions_list",
    "sessions_history",
    "sessions_yield",
]


@dataclass
class OpenclawAgentConfig(HarnessAgentConfig):
    version: str = "2026.7.1-2"
    context_limit: int = 1000000
    # Request max_tokens; also the ceiling thinking budgets are clamped
    # against (upstream default 8192 would strangle thinking).
    output_limit: int = 65536
    # off | minimal | low | medium | high | xhigh | adaptive | max.
    # None uses openclaw's default (medium). xhigh/max need the payload's
    # whitelist patch to reach the wire as adaptive+effort.
    thinking: str | None = None


class OpenclawAgent(HarnessAgent):
    """Runs OpenClaw inside the pod under the MiMo Agent protocol."""

    NAME = "openclaw"
    ERROR_STATUS = "OpenclawError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-openclaw-logs/openclaw.txt"]
    # --json prints one pretty-printed envelope on stdout; keep stderr
    # diagnostics from interleaving into it.
    SPLIT_STDERR = True

    def __init__(self, model, env, *, config_class=OpenclawAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    def _parse_events(self, raw: str) -> list[dict[str, Any]]:
        """Extract the single JSON envelope from the front of the log (the
        appended stderr tail is ignored unless the envelope is missing)."""
        decoder = json.JSONDecoder()
        idx = raw.find("{")
        while idx != -1:
            try:
                obj, _end = decoder.raw_decode(raw[idx:])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and ("payloads" in obj or "meta" in obj):
                return [obj]
            idx = raw.find("{", idx + 1)
        return []

    @property
    def _state_dir(self) -> str:
        # SQLite (WAL) state — must be pod-local disk, never a network mount.
        return f"{self._logs_dir}/state"

    def _openclaw_config(self) -> str:
        model = self._resolved_model
        config: dict[str, Any] = {
            "agents": {
                "defaults": {
                    "workspace": self._cwd,
                    # don't seed AGENTS.md/SOUL.md templates into the graded repo
                    "skipBootstrap": True,
                    "model": {"primary": f"{_PROVIDER_ID}/{model}"},
                    # no timeoutSeconds here: the schema rejects 0, and the
                    # per-run `--timeout 0` (unlimited) overrides it anyway
                    # off for the same fingerprint reason as the sessions_* deny
                    "memorySearch": {"enabled": False},
                    "heartbeat": {"every": "0m"},
                }
            },
            "tools": {
                "profile": "coding",
                "deny": _DENY_TOOLS,
                "exec": {"mode": "full"},
            },
            "models": {
                "mode": "merge",
                "providers": {
                    _PROVIDER_ID: {
                        # endpoint = baseUrl(/v1)? + /v1/messages either way
                        "baseUrl": self._gateway_base_url(v1=False),
                        "apiKey": self._api_key,
                        "api": "anthropic-messages",
                        "models": [
                            {
                                "id": model,
                                "name": model,
                                "reasoning": True,
                                "input": ["text"],
                                "contextWindow": self.config.context_limit,
                                "maxTokens": self.config.output_limit,
                            }
                        ],
                    }
                },
            },
            "update": {"checkOnStart": False},
            "diagnostics": {"otel": {"enabled": False}},
            "discovery": {"mdns": {"mode": "off"}},
        }
        if self.config.thinking:
            config["agents"]["defaults"]["thinkingDefault"] = self.config.thinking
        return json.dumps(config, ensure_ascii=False)

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._state_dir}")
        self._copy_text_to_pod(self._openclaw_config(), f"{self._state_dir}/openclaw.json")

    def _harness_env(self) -> dict[str, str]:
        return {
            "OPENCLAW_STATE_DIR": self._state_dir,
            "OPENCLAW_CONFIG_PATH": f"{self._state_dir}/openclaw.json",
            # payload build patch: the Anthropic transport otherwise clamps the
            # declared model maxTokens to 32000 for everything that is not
            # Claude Sonnet 5, silently capping output_limit.
            "OPENCLAW_USE_MODEL_MAX_TOKENS": "1",
        }

    def _command_parts(self, task: str) -> list[str]:
        # One stable session key per agent instance: re-running with the same
        # key IS the resume mechanism (and sidesteps the daily session reset).
        return [
            "openclaw",
            "agent",
            "--local",
            "--agent main",
            "--session-key mimo-task",
            f"--message-file {self._instruction_file}",
            "--json",
            "--timeout 0",  # run_timeout on env.execute is the backstop
        ]

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """--json prints one envelope: {payloads: [{text}], meta: {...}}.
        stderr diagnostics share the output file, so _parse_ndjson picks the
        envelope out of whatever JSON-looking lines survived."""
        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = ""
        envelope: dict[str, Any] = {}
        for ev in events:
            if "payloads" in ev or "meta" in ev:
                envelope = ev
        payloads = envelope.get("payloads") or []
        texts = [p.get("text") for p in payloads if isinstance(p, dict) and p.get("text")]
        meta = envelope.get("meta") or {}
        result_text = (meta.get("finalAssistantVisibleText") or "\n".join(texts)).strip()
        stop_reason = (meta.get("stopReason") or "").lower()
        if stop_reason in {"timeout", "error", "aborted", "abort", "cancelled", "canceled"}:
            status = self.ERROR_STATUS
        if meta.get("aborted"):
            status = self.ERROR_STATUS

        usage = (meta.get("agentMeta") or {}).get("usage") or {}
        if usage:
            self._fold_usage(
                [
                    {
                        "input": usage.get("input", 0),
                        "output": usage.get("output", 0),
                        "cache_read": usage.get("cacheRead", 0),
                        "cache_write": usage.get("cacheWrite", 0),
                    }
                ]
            )
            # the envelope carries no exact API-call count; tool calls + 1
            # (final reply) approximates it far better than the single
            # aggregate _fold_usage just counted.
            tool_calls = int((meta.get("toolSummary") or {}).get("calls", 0) or 0)
            try:
                self.model.n_calls += max(0, tool_calls)
            except Exception:
                pass
        if not envelope or not result_text:
            # rc==0 without a JSON envelope / final text = broken run
            status = self.ERROR_STATUS
        if not result_text:
            result_text = f"(no agent message; openclaw rc={rc}, {len(events)} events)"
        return result_text, status
