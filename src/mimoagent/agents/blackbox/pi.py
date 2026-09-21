"""Pi CLI as a mimoagent blackbox agent.

Pi runs inside the task pod and talks to the shared model gateway through its
custom ``gateway`` provider.  The wire protocol is selected by ``agent.api``:
the default ``openai-completions`` only needs ``/v1/chat/completions`` on the
gateway, while ``anthropic-messages`` speaks ``/v1/messages`` — required for
Claude models with extended thinking, where the thinking-block signatures must
be replayed verbatim (Pi's Anthropic driver preserves them; the OpenAI
Completions schema has no place to carry them).
"""

from __future__ import annotations

import json
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mimoagent import Environment, Model
from mimoagent.agents.base import AgentConfig, BaseAgent, InfraError
from mimoagent.agents.blackbox.detached import run_detached
from mimoagent.agents.blackbox.install_common import installer_env_prefix, stage_install_common
from mimoagent.environments import TransportError

_RESOURCES = Path(__file__).resolve().parent / "resources"
_INSTALL_SCRIPT = _RESOURCES / "install-pi.sh"

_POD_INSTALL_SCRIPT = "/tmp/mimo-install-pi.sh"
_POD_LOGS_DIR = "/tmp/mimo-pi-logs"
_POD_SESSION_DIR = f"{_POD_LOGS_DIR}/sessions"
_POD_INSTRUCTION_FILE = "/tmp/mimo-pi-instruction.txt"
_POD_OUTPUT_FILE = f"{_POD_LOGS_DIR}/pi.txt"
_POD_MODELS_FILE = f"{_POD_LOGS_DIR}/models.json"
_PI_BIN_DIR = "/opt/mimo-pi/bin"
_PROVIDER_ID = "gateway"
_URL_CREDENTIALS_RE = re.compile(r"(https?://)[^/@\s]+(?::[^/@\s]*)?@")


@dataclass
class PiAgentConfig(AgentConfig):
    """Configuration for the Pi blackbox scaffold."""

    cwd: str | None = None
    version: str = "0.83.0"
    # Pi provider API: "openai-completions" (/v1/chat/completions) or
    # "anthropic-messages" (/v1/messages). Use the latter for Claude models
    # with extended thinking so signatures survive the round trip.
    api: str = "openai-completions"
    thinking: str | None = None
    # Anthropic adaptive thinking (the native shape for Opus 4.6+): sends
    # ``thinking.type: "adaptive"`` plus ``output_config.effort`` instead of a
    # token budget, and unlocks the native ``xhigh``/``max`` effort levels.
    # The model decides per step whether to think; the effort is fixed by
    # ``thinking``. Requires ``api: anthropic-messages``.
    adaptive_thinking: bool = False
    # Overrides Pi's per-level thinking budgets (settings.json thinkingBudgets),
    # e.g. {"high": 32768}. Only meaningful for budget-based thinking (i.e.
    # ``adaptive_thinking: false``); budgets are capped at max_tokens - 1024.
    thinking_budgets: dict[str, int] | None = None
    context_window: int = 262144
    max_tokens: int = 32768
    # Input modalities advertised to Pi, written straight into the provider
    # catalog entry (Pi calls the field ``input``).
    #
    # ⚠ Leaving this out tells Pi the model is text-only: it then drops every
    # image and feeds the agent
    #   "[Current model does not support images. The image will be omitted
    #    from this request.]"
    # No error, no failure — the agent keeps reading renders and sees nothing.
    #
    # Pi only consults its built-in catalog for model ids it ships; a gateway
    # alias (e.g. claude-opus-5 behind a gateway) is unknown to it, so
    # capabilities can *only* come from the entry we generate here.
    #
    # Value mirrors Pi 0.83.0's built-in claude-opus-4-8 entry:
    #   { id: "claude-opus-4-8", …, reasoning: true, input: ["text", "image"], … }
    # Pass ["text"] explicitly for text-only models.
    input_modalities: list[str] | None = None
    install_timeout: int = 1800
    run_timeout: int = 18000
    # Stall watchdog for the detached run: kill pi when its event log saw no
    # write for this many seconds. 0 disables it; keep it above the longest
    # legitimate model-call silence.
    stall_timeout: int = 0
    extra_env: dict[str, str] | None = None
    skip_install: bool = False
    # Extra environment for the install script (mirror overrides such as
    # GITHUB_BASE; see resources/install-common.sh).
    install_env: dict[str, str] | None = None


class PiAgent(BaseAgent):
    """Run Pi in the task pod while satisfying mimoagent's Agent protocol."""

    IDLE_STATUS = "Completed"
    ENV_TRAJECTORY_FILES = [_POD_OUTPUT_FILE]

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = PiAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._installed = False
        self._staged = False
        self._turns = 0

    def get_model_query_kwargs(self) -> dict:
        return {}

    def step(self) -> dict | None:  # pragma: no cover - run() owns Pi's loop
        return None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Install Pi once, run one task turn, and collect its JSON event stream."""
        self.extra_template_vars |= {"task": task, **kwargs}
        try:
            if not self.config.skip_install and not self._installed:
                self._install()
                self._installed = True
            if not self._staged:
                self._stage()
                self._staged = True
            rc = self._run_pi(task)
        except TransportError as e:
            raise InfraError(str(e)) from e

        events = self._collect_events()
        result_text, status = self._collect(rc, events)
        self.add_message("user", task)
        self.add_message("assistant", result_text)
        self._turns += 1
        return status, result_text

    @property
    def _cwd(self) -> str:
        if self.config.cwd:
            return self.config.cwd
        return getattr(self.env.config, "cwd", "/testbed")

    @property
    def _resolved_model(self) -> str:
        # ``model_name`` is the gateway serving name, sent verbatim — it may
        # itself contain slashes (channel-prefixed models like ``aws/claude-x``).
        return self.model.config.model_name

    @property
    def _model_kwargs(self) -> dict:
        return getattr(self.model.config, "model_kwargs", {}) or {}

    def _install(self) -> None:
        version = self.config.version
        if not version:
            raise ValueError("pi: agent.version must be pinned")
        self.logger.info(f"[pi] installing Pi {version} into pod (standalone binary)...")
        started_at = time.monotonic()
        self.env.copy_to(str(_INSTALL_SCRIPT), _POD_INSTALL_SCRIPT)
        path_prefix = stage_install_common(self.env) + installer_env_prefix(self.config.install_env)
        res = self.env.execute(
            f"{path_prefix}PI_VERSION={shlex.quote(version)} bash {_POD_INSTALL_SCRIPT}",
            timeout=self.config.install_timeout,
        )
        out = res.get("output", "") or ""
        if "PI_INSTALL_OK" not in out:
            safe_tail = _URL_CREDENTIALS_RE.sub(r"\1[redacted]@", out[-3000:])
            raise RuntimeError(f"[pi] install failed (rc={res.get('returncode')}). Output tail:\n{safe_tail}")
        elapsed = time.monotonic() - started_at
        self.logger.info(f"[pi] install OK in {elapsed:.1f}s")

    def _copy_text_to_pod(self, text: str, remote_path: str) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(text)
            local_path = f.name
        try:
            self.env.copy_to(local_path, remote_path)
        finally:
            os.unlink(local_path)

    # Wire protocols this adapter has been validated against. Pi supports more
    # (openai-responses, google-generative-ai, ...) — extend deliberately.
    _SUPPORTED_APIS = ("openai-completions", "anthropic-messages")

    def _api_base_url(self) -> str:
        base_url = (self._model_kwargs.get("base_url") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("[pi] model.model_kwargs.base_url is required")
        if self.config.api == "anthropic-messages":
            # Pi's Anthropic driver wraps @anthropic-ai/sdk, which appends
            # /v1/messages to the base URL itself.
            return base_url.removesuffix("/v1")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return base_url

    def _provider_config(self) -> str:
        """Build Pi's custom provider config for the selected wire protocol.

        For Chat Completions, BlackBoxRouter reads ``max_tokens`` but not
        ``max_completion_tokens``. The compatibility block therefore selects
        the former and also avoids optional OpenAI fields that a minimal
        compatible gateway may reject. The Anthropic driver has its own wire
        format, so it gets no compat block.
        """
        if self.config.api not in self._SUPPORTED_APIS:
            raise RuntimeError(
                f"[pi] unknown api: {self.config.api!r} (expected one of {sorted(self._SUPPORTED_APIS)})"
            )
        if self.config.adaptive_thinking and self.config.api != "anthropic-messages":
            raise RuntimeError("[pi] adaptive_thinking requires api: anthropic-messages")
        model = self._resolved_model
        model_entry: dict[str, Any] = {
            "id": model,
            "reasoning": True,
            # Must be declared: Pi treats an entry without ``input`` as
            # text-only and silently strips every image (see
            # PiAgentConfig.input_modalities). Gateway aliases are absent from
            # Pi's built-in catalog, so this entry is the only source of truth
            # for model capabilities.
            "input": list(self.config.input_modalities or ["text", "image"]),
            "contextWindow": self.config.context_window,
            "maxTokens": self.config.max_tokens,
        }
        if self.config.adaptive_thinking:
            # Mirror Pi's built-in claude-opus-4-8 catalog entry: adaptive
            # thinking payload with the native xhigh/max effort levels.
            model_entry["thinkingLevelMap"] = {"xhigh": "xhigh", "max": "max"}
            model_entry["compat"] = {"forceAdaptiveThinking": True}
        provider: dict[str, Any] = {
            "baseUrl": self._api_base_url(),
            "api": self.config.api,
            "apiKey": self._model_kwargs.get("api_key") or "dummy-key",
            "models": [model_entry],
        }
        if self.config.api == "openai-completions":
            provider["compat"] = {
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
                "supportsStore": False,
                "maxTokensField": "max_tokens",
            }
        else:
            # Multi-replica gateways only hit the prompt cache when consecutive
            # calls land on the same replica. x-session-affinity (from Pi's
            # session id) is the routing hook for that; harmless if the
            # gateway ignores it.
            provider["compat"] = {"sendSessionAffinityHeaders": True}
        return json.dumps({"providers": {_PROVIDER_ID: provider}}, ensure_ascii=False)

    def _stage(self) -> None:
        self.env.execute(f"mkdir -p {_POD_LOGS_DIR} {_POD_SESSION_DIR}")
        self._copy_text_to_pod(self._provider_config(), _POD_MODELS_FILE)
        if self.config.thinking_budgets:
            self._copy_text_to_pod(
                json.dumps({"thinkingBudgets": self.config.thinking_budgets}),
                f"{_POD_LOGS_DIR}/settings.json",
            )

    def _build_env(self) -> dict[str, str]:
        env = {
            "PI_CODING_AGENT_DIR": _POD_LOGS_DIR,
            "PI_CODING_AGENT_SESSION_DIR": _POD_SESSION_DIR,
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
        router_host = urlparse(self._model_kwargs.get("base_url") or "").hostname
        if router_host:
            env["no_proxy"] = f"{router_host},localhost,127.0.0.1"
            env["NO_PROXY"] = env["no_proxy"]
        if self.config.extra_env:
            env.update(self.config.extra_env)
        return env

    def _run_pi(self, task: str) -> int:
        self._copy_text_to_pod(task, _POD_INSTRUCTION_FILE)
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self._build_env().items())
        parts = [
            "pi",
            "--mode json",
            f"--provider {_PROVIDER_ID}",
            f"--model {shlex.quote(self._resolved_model)}",
            f"--session-dir {_POD_SESSION_DIR}",
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
        if self._turns > 0:
            parts.append("--continue")
        if self.config.thinking:
            parts.append(f"--thinking {shlex.quote(self.config.thinking)}")

        cmd = (
            f'export PATH={_PI_BIN_DIR}:"$PATH"; '
            f"{env_prefix} {' '.join(parts)} "
            f"< {_POD_INSTRUCTION_FILE} > {_POD_OUTPUT_FILE} 2>&1"
        )
        self.logger.info(f"[pi] running (turn {self._turns + 1}) in {self._cwd}")
        rc, _ = run_detached(
            self.env,
            cmd,
            cwd=self._cwd,
            timeout=self.config.run_timeout,
            tag="pi",
            log_hint="event log: agent_msgs/pi.txt",
            logger=self.logger,
            idle_files=[_POD_OUTPUT_FILE],
            idle_timeout=self.config.stall_timeout,
        )
        return rc

    def _collect_events(self) -> list[dict[str, Any]]:
        raw = ""
        if self.msg_path and hasattr(self.env, "copy_out"):
            dest = self.msg_path.parent / "pi.txt"
            try:
                self.env.copy_out(_POD_OUTPUT_FILE, str(dest))
                self.logger.info(f"[pi] copied event log to {dest}")
                raw = dest.read_text(errors="replace")
            except Exception as e:
                self.logger.warning(f"[pi] could not copy pi.txt: {e}")
        if not raw:
            try:
                res = self.env.execute(f"tail -c 2000000 {_POD_OUTPUT_FILE} 2>/dev/null || true")
                raw = res.get("output", "") or ""
            except Exception as e:
                self.logger.warning(f"[pi] could not read pi.txt: {e}")
        return self._parse_ndjson(raw)

    @staticmethod
    def _parse_ndjson(raw: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                continue
        return events

    @staticmethod
    def _assistant_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = []
        for event in events:
            if event.get("type") != "message_end":
                continue
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                messages.append(message)
        return messages

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content") or []
        if isinstance(content, str):
            return content.strip()
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

    def _collect(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        messages = self._assistant_messages(events)
        self._fold_usage(messages)

        status = self.IDLE_STATUS
        last = messages[-1] if messages else {}
        stop_reason = last.get("stopReason")
        retry_failed = any(event.get("type") == "auto_retry_end" and event.get("success") is False for event in events)
        if rc != 0 or not messages or stop_reason in {"error", "aborted"} or retry_failed:
            status = "PiError"

        result_text = self._message_text(last)
        if not result_text:
            result_text = last.get("errorMessage") or ""
        if not result_text:
            result_text = f"(no agent message; pi rc={rc}, {len(events)} events)"
        return result_text, status

    def _fold_usage(self, messages: list[dict[str, Any]]) -> None:
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        per_call: list[dict[str, int]] = []
        for message in messages:
            usage = message.get("usage") or {}
            call = {
                "input": int(usage.get("input", 0) or 0),
                "output": int(usage.get("output", 0) or 0),
                "cache_read": int(usage.get("cacheRead", 0) or 0),
                "cache_write": int(usage.get("cacheWrite", 0) or 0),
            }
            per_call.append(call)
            for key, value in call.items():
                totals[key] += value

        stats = getattr(self.model, "token_stats", None)
        if stats is not None and messages:
            stats.input_tokens += totals["input"]
            stats.output_tokens += totals["output"]
            stats.cache_read_tokens += totals["cache_read"]
            stats.cache_creation_tokens += totals["cache_write"]
        if messages:
            self.extra_template_vars["pi_usage"] = totals
            try:
                self.model.n_calls += len(messages)
            except Exception:
                pass
            # Report into the batch-level aggregator too, one add() per pi
            # assistant message, so pi sessions count toward the batch token
            # progress and MIMOAGENT_GLOBAL_CALL_LIMIT like every other scaffold.
            shared = getattr(self.model, "shared_stats", None)
            if shared is not None:
                from mimoagent.models import TokenStats

                for call in per_call:
                    shared.add(
                        TokenStats(
                            input_tokens=call["input"],
                            output_tokens=call["output"],
                            cache_read_tokens=call["cache_read"],
                            cache_creation_tokens=call["cache_write"],
                        )
                    )
