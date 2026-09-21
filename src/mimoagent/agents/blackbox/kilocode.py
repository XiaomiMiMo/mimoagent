"""Kilo Code CLI as a MiMo Agent blackbox harness.

A fork of opencode (see :mod:`opencode`), so the provider block, the
``--format json`` NDJSON event stream and the ``variants``/``--variant`` effort
mechanism are shared verbatim. Payload is upstream's bun-compiled standalone
binary — the ``-baseline`` linux-x64 build, the only variant that runs both on
AVX2 and non-AVX2 hosts (``-musl`` links musl's loader and dies on glibc pods).

Verified on 7.4.22 against an Anthropic-compatible gateway serving claude-opus-5: adaptive
thinking + ``output_config.effort`` reach the wire natively (no whitelist patch,
unlike opencode/openclaw), signed thinking blocks replay across turns,
cache_control gives cache_read hits, and one turn issues exactly as many
requests as it reports ``step_finish`` events.

Three things are load-bearing and easy to lose:

* **the ``permission`` deny map**, not opencode's per-agent ``tools`` block.
  Kilo's default 12-tool set includes the exact lowercase name ``todowrite``,
  which the gateway's Claude-subscription channel rejects outright (400
  "Third-party apps...") — a total failure, not a degradation. Denying drops
  the tool from the wire schema (12 → 6, measured).
* **``KILO_DISABLE_DEFAULT_PLUGINS``** — otherwise every fresh HOME runs a
  live ``registry.npmjs.org`` install of ``@kilocode/plugin``, which in a
  no-egress pod becomes a silent 5/10/20s backoff stall.
* **``--title`` + ``small_model``** — kilo's small-model resolver walks
  haiku/mini/nano family heuristics that a single-model custom provider always
  misses, then hard-falls back to ``kilo/kilo-auto/small`` (its own cloud), so
  the session-title call would leave the gateway entirely.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gateway"

# Nothing is denied on capability grounds. ``question``/``suggest`` used to be
# listed here, but they are not registered in ``kilo run`` at all — with an empty
# deny map the wire still carries the same 11 tools and neither name appears
# (measured against a local mock), so denying them was dead config. Only the
# fingerprint names below are real, and they are added in ``_permission_map``.
_DENY_TOOLS: list[str] = []


@dataclass
class KilocodeAgentConfig(HarnessAgentConfig):
    version: str = "7.4.22"
    # Model context window / max output tokens (config ``limit`` block).
    # ``context`` drives auto-compaction; ``output`` above 32000 is unlocked
    # via KILO_EXPERIMENTAL_OUTPUT_TOKEN_MAX automatically (the cap survived
    # the fork from opencode).
    context_limit: int | None = None
    output_limit: int | None = None
    # Extended thinking, budget style: model options.thinking.budgetTokens.
    thinking_budget: int | None = None
    # Adaptive thinking with a fixed effort ("xhigh" | "max" | ...): lands as
    # a handwritten model variant selected via --variant. Mutually exclusive
    # with thinking_budget.
    thinking_effort: str | None = None
    # Subscription-style Claude gateway channels reject any request whose tools
    # include the name "todowrite" (400 "Third-party apps..."), so todowrite /
    # todoread are denied by default.
    todo_tool: bool = False
    # Any non-empty title suppresses the session-title LLM call (kilo only
    # generates one when the title is still its own "New session - <ISO>" stamp).
    session_title: str = "mimo-rollout"
    # Ignore the graded repo's own ./kilo.json, .kilo/ and AGENTS.md. Hermetic
    # by default so a repo-planted config cannot append to the system prompt.
    # Note kilo's flag is broader than opencode's: it also suppresses
    # AGENTS.md, so a run that wants repo instructions honoured must set this
    # false and accept the config-injection risk that comes with it.
    project_config: bool = False


class KilocodeAgent(HarnessAgent):
    """Runs Kilo Code inside the pod under the MiMo Agent protocol."""

    NAME = "kilocode"
    ERROR_STATUS = "KilocodeError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-kilocode-logs/kilocode.txt"]
    # --format json writes only JSON lines to stdout; the debug log goes to a
    # file under XDG_DATA_HOME. Merging stderr is harmless (_parse_ndjson drops
    # non-{ lines) and keeps a readable tail in the collected log.
    SPLIT_STDERR = False

    def __init__(self, model, env, *, config_class=KilocodeAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    def _permission_map(self) -> dict[str, str]:
        """Root-level permission map. Kilo evaluates it with last-match-wins
        for glob patterns, so ``"*"`` must come first and the specific denies
        after — which this insertion order gives naturally."""
        perm = {"*": "allow"}
        deny = list(_DENY_TOOLS)
        if not self.config.todo_tool:
            deny += ["todowrite", "todoread"]
        for tool in deny:
            perm[tool] = "deny"
        return perm

    def _provider_config(self) -> str:
        if self.config.thinking_budget and self.config.thinking_effort:
            raise RuntimeError(
                "[kilocode] thinking_budget and thinking_effort are mutually "
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
        qualified = f"{_PROVIDER_ID}/{model}"
        config = {
            "model": qualified,
            # Title / summary / branch-name calls resolve through the "small"
            # model; without this they fall back to kilo/kilo-auto/small, i.e.
            # off our gateway entirely.
            "small_model": qualified,
            # No Kilo/Apertis catalog fetch on startup (~293 models, ~2s).
            "disabled_providers": ["kilo", "apertis"],
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
            "permission": self._permission_map(),
            "share": "disabled",
            "autoupdate": False,
            # coding capability + the agent's own undo history
            "snapshot": True,
            "lsp": True,
            "formatter": True,
            "instructions": [],
            "mcp": {},
            # auto-compaction is part of how a long task gets finished
            "compaction": {"auto": True},
        }
        return json.dumps(config, ensure_ascii=False)

    def _harness_env(self) -> dict[str, str]:
        env = {
            "KILO_CONFIG_CONTENT": self._provider_config(),
            "KILO_DISABLE_MODELS_FETCH": "1",
            "KILO_DISABLE_AUTOUPDATE": "1",
            "KILO_DISABLE_LSP_DOWNLOAD": "1",
            "KILO_DISABLE_SHARE": "1",
            # Kilo-only cloud endpoints (ingest.kilosessions.ai,
            # wss://events.kiloapps.io, PostHog).
            "KILO_DISABLE_SESSION_INGEST": "1",
            "KILO_DISABLE_PRESENCE": "1",
            "KILO_TELEMETRY_LEVEL": "off",
            # Without this a fresh HOME npm-installs @kilocode/plugin at
            # startup — the one remaining runtime network dependency.
            "KILO_DISABLE_DEFAULT_PLUGINS": "1",
            "KILO_DISABLE_EMBEDDED_WEB_UI": "1",
            "KILO_DISABLE_EXTERNAL_SKILLS": "1",
            # Sessions land in a SQLite db (WAL) — keep it on pod-local disk
            # under the logs dir, never a network mount. The explicit filename
            # stops a channel rename from relocating it.
            "KILO_DISABLE_CHANNEL_DB": "1",
            "KILO_DB": f"{self._logs_dir}/kilo.db",
            # The npm launcher normally sets this; we exec the binary directly.
            "KILO_TREE_SITTER_WASM_DIR": f"{self._bin_dir}/tree-sitter",
            "XDG_DATA_HOME": f"{self._logs_dir}/xdg/data",
            "XDG_CONFIG_HOME": f"{self._logs_dir}/xdg/config",
            "XDG_CACHE_HOME": f"{self._logs_dir}/xdg/cache",
            "XDG_STATE_HOME": f"{self._logs_dir}/xdg/state",
        }
        if not self.config.project_config:
            env["KILO_DISABLE_PROJECT_CONFIG"] = "1"
        if self.config.output_limit and self.config.output_limit > 32000:
            env["KILO_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(self.config.output_limit)
        return env

    def _command_parts(self, task: str) -> list[str]:
        parts = [
            "kilo",
            "run",
            "--format json",
            # Documented auto-approve flag. Without it the ask handler
            # auto-*rejects* and the turn dies on a silent error event.
            "--auto",
            # Suppress the per-session title-generation LLM call.
            f"--title {shlex.quote(self.config.session_title)}",
            f"-m {shlex.quote(f'{_PROVIDER_ID}/{self._resolved_model}')}",
        ]
        if self._turns > 0:
            parts.append("--continue")
        if self.config.thinking_effort:
            parts.append(f"--variant {shlex.quote(self.config.thinking_effort)}")
        # prompt arrives on stdin (the shared instruction-file redirect); kilo
        # reads it whenever stdout is not a TTY
        return parts

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """Same event shapes as opencode: ``text`` parts, ``step_finish``
        token blocks, ``error`` events — except kilo puts the error payload at
        the top level rather than under ``part``."""
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
                err = ev.get("error") or {}
                error_text = (
                    ((err.get("data") or {}).get("message")) or err.get("name") or json.dumps(ev, ensure_ascii=False)
                )
        if error_text:
            # any error event fails the turn, even alongside partial text
            status = self.ERROR_STATUS
            result_text = result_text or error_text
        self._fold_usage(per_call)
        if not result_text:
            # no text event at rc==0 = broken/empty stream, not success
            status = self.ERROR_STATUS
            result_text = f"(no agent message; kilocode rc={rc}, {len(events)} events)"
        return result_text, status
