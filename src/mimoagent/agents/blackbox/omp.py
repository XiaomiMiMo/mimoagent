"""Oh My Pi (omp) as a MiMo Agent blackbox harness.

Community fork of Pi (see :mod:`pi` for the upstream adapter) compiled to a
single bun binary at payload build time; the externalized ``pi_natives``
``.node`` files ship alongside and are staged into ``$HOME/.omp/natives/`` at
install. The ``--mode json`` event stream, ``message_end`` usage shape and
``--continue`` semantics match Pi, but the fork diverges everywhere else —
verified on 17.3.4:

* config moved to ``models.yml`` / ``config.yml`` under ``PI_CODING_AGENT_DIR``
  (both written as JSON here — YAML is a superset);
* a custom ``anthropic-messages`` provider MUST set ``auth: apiKey``:
  without it omp impersonates Claude Code (claude-cli UA, oauth betas,
  underscored tool names, max_tokens clamped to 64000);
* ``thinking.mode: anthropic-adaptive`` + ``--thinking xhigh|max`` sends
  ``thinking: {type: adaptive}`` + ``output_config.effort``; signatures
  replay; cache_control lands on system + last messages (≤4 breakpoints);
* stdin prompts are silently dropped under bun (isTTY bug), so the task
  travels as an ``@file`` argv reference;
* ``--no-approve`` is gone (``--approval-mode yolo``), and PI_OFFLINE /
  PI_SKIP_VERSION_CHECK / PI_TELEMETRY are no longer read (update checks are
  interactive-only; OTLP export needs explicit OTEL_* env);
* every thinking request also carries the Anthropic context-management beta
  (``clear_thinking_20251015``), and qualifying tool definitions carry
  ``strict: true`` — neither has an upstream opt-out, and a Bedrock-backed
  route 400s the request over either one. The payload build puts both behind
  env vars; see ``context_management`` / ``tool_strict`` below.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gateway"
# omp's full documented 17.3.4 tool set. Nothing is dropped: there is no
# ask-the-user tool here, and web_search failing inside a pod is acceptable.
# Unknown names make omp exit 2, so keep this to the documented ids.
_DEFAULT_TOOLS = "read,bash,edit,write,grep,glob,todo,task,web_search"


@dataclass
class OmpAgentConfig(HarnessAgentConfig):
    version: str = "17.3.4"
    context_limit: int = 1000000
    output_limit: int = 65536
    # Thinking level for --thinking: off | minimal | low | medium | high |
    # xhigh | max | auto. None omits the flag (model default).
    thinking: str | None = None
    # Wire shape for thinking levels: "anthropic-adaptive" sends
    # thinking:{type:adaptive} + output_config.effort (Opus 4.6+ native);
    # "budget" sends budget_tokens from thinking_budgets.
    thinking_mode: str = "anthropic-adaptive"
    # Only meaningful with thinking_mode: budget (config.yml thinkingBudgets);
    # budgets are clamped to maxTokens - 4000.
    thinking_budgets: dict[str, int] | None = None
    # Tool whitelist (--tools). None runs omp's default set (which includes
    # web_search).
    tools: str | None = _DEFAULT_TOOLS
    # Anthropic context-management beta (clear_thinking_20251015). Upstream
    # attaches it to every thinking request with no way to opt out; the payload
    # build gates it on $OMP_CONTEXT_MANAGEMENT instead. Off by default: an
    # Anthropic-native route accepts it, but a Bedrock-backed one 400s the
    # whole request ("context_management: Extra inputs are not permitted"), and
    # a gateway may route either way per channel.
    context_management: bool = False
    # Same story for `strict: true` on qualifying tool definitions (`edit`),
    # gated on $OMP_TOOL_STRICT by the payload build. Dropping it only relaxes
    # schema adherence for that one tool; keeping it 400s the same routes.
    tool_strict: bool = False


class OmpAgent(HarnessAgent):
    """Runs Oh My Pi inside the pod under the MiMo Agent protocol."""

    NAME = "omp"
    ERROR_STATUS = "OmpError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-omp-logs/omp.txt"]

    def __init__(self, model, env, *, config_class=OmpAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    @property
    def _agent_dir(self) -> str:
        return f"{self._logs_dir}/agent"

    @property
    def _session_dir(self) -> str:
        return f"{self._logs_dir}/sessions"

    def _models_yml(self) -> str:
        model_entry: dict[str, Any] = {
            "id": self._resolved_model,
            "name": self._resolved_model,
            "reasoning": True,
            "contextWindow": self.config.context_limit,
            "maxTokens": self.config.output_limit,
            "thinking": {
                "mode": self.config.thinking_mode,
                "efforts": ["minimal", "low", "medium", "high", "xhigh", "max"],
            },
        }
        models = {
            "providers": {
                _PROVIDER_ID: {
                    # omp's own anthropic client appends /v1/messages(?beta=true)
                    "baseUrl": self._gateway_base_url(v1=False),
                    "apiKey": self._api_key,
                    "api": "anthropic-messages",
                    # without this omp impersonates Claude Code on custom
                    # anthropic-messages providers (isOAuth heuristics)
                    "auth": "apiKey",
                    "models": [model_entry],
                }
            }
        }
        return json.dumps(models, ensure_ascii=False)

    def _config_yml(self) -> str:
        config: dict[str, Any] = {
            "modelRoles": {"default": f"{_PROVIDER_ID}/{self._resolved_model}"},
            "startup": {"checkUpdate": False},
            "tools": {"approvalMode": "yolo"},
            # default 10 retries × exponential backoff hangs for minutes on a
            # dead gateway; the batch layer owns retry policy.
            "retry": {"maxRetries": 3},
        }
        if self.config.thinking_budgets:
            config["thinkingBudgets"] = self.config.thinking_budgets
        return json.dumps(config, ensure_ascii=False)

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._agent_dir} {self._session_dir}")
        # YAML is a JSON superset, and omp only honours the .yml files (json
        # variants are one-shot migration inputs).
        self._copy_text_to_pod(self._models_yml(), f"{self._agent_dir}/models.yml")
        self._copy_text_to_pod(self._config_yml(), f"{self._agent_dir}/config.yml")
        # The compiled binary expects the externalized pi_natives modules
        # under $HOME/.omp/natives/<version>/ (the official release
        # self-extracts them there; our payload stages them explicitly).
        self.env.execute(
            f'mkdir -p "$HOME/.omp/natives/{self.config.version}" && '
            f"cp -n {self._install_dir}/natives/*.node "
            f'"$HOME/.omp/natives/{self.config.version}/" 2>/dev/null || true'
        )

    def _harness_env(self) -> dict[str, str]:
        env = {"PI_CODING_AGENT_DIR": self._agent_dir}
        # both re-enable request fields the payload build patched out
        if self.config.context_management:
            env["OMP_CONTEXT_MANAGEMENT"] = "1"
        if self.config.tool_strict:
            env["OMP_TOOL_STRICT"] = "1"
        return env

    def _command_parts(self, task: str) -> list[str]:
        parts = [
            "omp",
            "--mode json",
            f"--model {shlex.quote(f'{_PROVIDER_ID}/{self._resolved_model}')}",
            f"--session-dir {self._session_dir}",
            "--approval-mode yolo",
            "--no-extensions",
            "--no-skills",
            "--no-rules",
            "--no-title",
        ]
        if self.config.tools:
            parts.append(f"--tools {shlex.quote(self.config.tools)}")
        if self.config.thinking:
            parts.append(f"--thinking {shlex.quote(self.config.thinking)}")
        if self._turns > 0:
            parts.append("--continue")
        # bun's isTTY quirk silently drops piped stdin — pass the task as an
        # @file argument instead.
        parts.append(f"@{self._instruction_file}")
        return parts

    @staticmethod
    def _assistant_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            ev.get("message") or {}
            for ev in events
            if ev.get("type") == "message_end" and (ev.get("message") or {}).get("role") == "assistant"
        ]

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """Pi-compatible stream: message_end carries the authoritative
        message + usage; json mode may exit 0 on model errors, so stopReason
        and auto_retry_end are checked explicitly (same as pi.py)."""
        messages = self._assistant_messages(events)
        per_call = []
        for message in messages:
            usage = message.get("usage") or {}
            per_call.append(
                {
                    "input": usage.get("input", 0),
                    "output": usage.get("output", 0),
                    "cache_read": usage.get("cacheRead", 0),
                    "cache_write": usage.get("cacheWrite", 0),
                }
            )
        self._fold_usage(per_call)

        status = self.IDLE_STATUS
        last = messages[-1] if messages else {}
        retry_failed = any(ev.get("type") == "auto_retry_end" and ev.get("success") is False for ev in events)
        if rc != 0 or not messages or last.get("stopReason") in {"error", "aborted"} or retry_failed:
            status = self.ERROR_STATUS

        content = last.get("content") or []
        if isinstance(content, str):
            result_text = content.strip()
        else:
            result_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        if not result_text:
            result_text = last.get("errorMessage") or ""
        if not result_text:
            result_text = f"(no agent message; omp rc={rc}, {len(events)} events)"
        return result_text, status
