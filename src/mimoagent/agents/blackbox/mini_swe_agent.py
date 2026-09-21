"""Upstream mini-swe-agent as a MiMo Agent blackbox harness.

``--agent-class default`` is required for non-interactive runs; ``-y`` alone
leaves upstream on InteractiveAgent whose ``confirm_exit`` reads stdin and
dies with EOFError.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

# Derive the staged config from the installed mini.yaml so a version bump
# picks up upstream's new prompt instead of a stale copy.
_MKCONFIG = """
import sys, pathlib, yaml
src, dest, max_fmt_errors = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(pathlib.Path(src).read_text())
agent = cfg.setdefault("agent", {})
# mode: confirm is an InteractiveAgent knob; dead under DefaultAgent.
agent.pop("mode", None)
agent["max_consecutive_format_errors"] = max_fmt_errors
pathlib.Path(dest).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("MKCONFIG_OK")
"""

# litellm-free model backend, staged into the payload venv and selected via
# ``model.model_class``. Why: the bundled litellm (drop_params=true) silently
# drops reasoning_effort for openai-provider models and rewrites params per
# provider — params must hit the wire exactly as configured. This subclass
# keeps all of LitellmModel's tool-call/retry/cost plumbing (the OpenAI SDK
# response object is shape-compatible with litellm's) and only replaces the
# transport: OpenAI SDK straight to the gateway, serving model name sent
# verbatim (provider prefix stripped), reasoning_effort ALWAYS in the body via
# extra_body plus deepseek's native ``thinking`` switch (official SDK usage:
# reasoning_effort=... + extra_body={"thinking": {"type": "enabled"}}), and
# reasoning_content round-trips through history replay untouched.
_NATIVE_MODEL = """
import inspect
import os

import openai

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

_CLIENT_KEYS = ("api_key", "base_url", "api_base", "timeout")
# litellm-only knobs that must never reach the wire or the SDK.
_LITELLM_ONLY = ("drop_params", "num_retries", "litellm_model_registry")
# Params the installed SDK accepts on create(); anything else tunnels
# through extra_body (mirrors the deepswe-fix native adapter).
_CREATE_PARAMS = frozenset(
    inspect.signature(openai.resources.chat.completions.Completions.create).parameters
) - {"self"}


class NativeOpenAIModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mk = self.config.model_kwargs
        base_url = mk.get("base_url") or mk.get("api_base") or os.getenv("OPENAI_BASE_URL")
        api_key = mk.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        timeout = float(mk.get("timeout") or 3600)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        # litellm provider prefix (openai/, deepseek/, ...) is routing info,
        # not part of the serving name; send the bare name verbatim.
        self._serving_model = self.config.model_name.split("/", 1)[-1]
        # Fail fast instead of retrying hopeless requests.
        self.abort_exceptions = list(self.abort_exceptions) + [
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        ]

    def _query(self, messages, **kwargs):
        call = {
            k: v
            for k, v in {**self.config.model_kwargs, **kwargs}.items()
            if k not in _CLIENT_KEYS and k not in _LITELLM_ONLY
        }
        extra = dict(call.pop("extra_body", None) or {})
        effort = call.pop("reasoning_effort", None) or "max"
        extra.setdefault("reasoning_effort", effort)
        extra.setdefault("thinking", {"type": "enabled"})
        # Unknown-to-SDK kwargs ride in extra_body (explicit extra_body wins).
        unknown = {k: call.pop(k) for k in [k for k in call if k not in _CREATE_PARAMS]}
        extra = {**unknown, **extra}
        return self._client.chat.completions.create(
            model=self._serving_model,
            messages=messages,
            tools=[BASH_TOOL],
            extra_body=extra,
            **call,
        )
"""
_NATIVE_MODEL_MODULE = "mimo_native_openai_model"


@dataclass
class MiniSweAgentConfig(HarnessAgentConfig):
    version: str = "2.4.6"
    provider_prefix: str = "anthropic/"
    # low|medium|high|xhigh|max
    reasoning_effort: str | None = "xhigh"
    # None = do not send max_tokens; the model behind the gateway decides (upstream mini.yaml does not set it either)
    max_tokens: int | None = None
    # 0 = unlimited
    step_limit: int = 100
    # USD; 0 disables. Upstream default $3 silently truncates.
    cost_limit: float = 0.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    # Upstream default 30s kills any build/test.
    command_timeout: int = 300
    # Upstream default 10 can eat the whole per-turn budget.
    retry_attempts: int = 3


class MiniSweAgent(HarnessAgent):
    """Runs upstream mini-swe-agent inside the pod under the MiMo Agent protocol."""

    NAME = "mini-swe-agent"
    ERROR_STATUS = "MiniSweAgentError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-mini-swe-agent-logs/mini-swe-agent.txt"]
    SPLIT_STDERR = False

    def __init__(self, model, env, *, config_class=MiniSweAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    @property
    def _config_file(self) -> str:
        return f"{self._logs_dir}/mini-harness.yaml"

    @property
    def _traj_file(self) -> str:
        return f"{self._logs_dir}/traj-{self._turns}.json"

    @property
    def _mswea_home(self) -> str:
        # minisweagent mkdirs this at import time; must be pod-local writable.
        return f"{self._logs_dir}/mswea"

    def _stage_files(self) -> None:
        # Stage the litellm-free model backend into the installed venv (importable
        # location); mini selects it through model.model_class.
        self._copy_text_to_pod(
            _NATIVE_MODEL,
            f"{self._install_dir}/venv/lib/python3.12/site-packages/{_NATIVE_MODEL_MODULE}.py",
        )
        script = f"{self._logs_dir}/mkconfig.py"
        self._copy_text_to_pod(_MKCONFIG, script)
        src = f"{self._install_dir}/venv/lib/python3.12/site-packages/minisweagent/config/mini.yaml"
        res = self.env.execute(
            f"{self._install_dir}/venv/bin/python {shlex.quote(script)} "
            f"{shlex.quote(src)} {shlex.quote(self._config_file)} "
            f"{self.config.max_consecutive_format_errors} 2>&1"
        )
        out = res.get("output", "") or ""
        if "MKCONFIG_OK" not in out:
            raise RuntimeError(
                f"[{self.NAME}] could not derive the staged config from {src} "
                f"(rc={res.get('returncode')}):\n{out[-2000:]}"
            )

    def _harness_env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_API_BASE": self._gateway_base_url(v1=False),
            "ANTHROPIC_API_KEY": self._api_key,
            # openai/-prefixed models route to litellm's openai provider, which
            # reads OPENAI_*; with only ANTHROPIC_* exported every non-claude
            # model died on "Missing credentials ... set OPENAI_API_KEY".
            "OPENAI_API_KEY": self._api_key,
            "OPENAI_BASE_URL": self._gateway_base_url(v1=True),
            # deepseek/-prefixed models (litellm deepseek provider — the exact
            # provider the official site trials used; it also translates
            # reasoning_effort into deepseek's native thinking param).
            "DEEPSEEK_API_KEY": self._api_key,
            "DEEPSEEK_API_BASE": self._gateway_base_url(v1=True),
            "MSWEA_CONFIGURED": "1",
            "MSWEA_GLOBAL_CONFIG_DIR": self._mswea_home,
            "MSWEA_SILENT_STARTUP": "1",
            # Missing price entry must not abort a rollout.
            "MSWEA_COST_TRACKING": "ignore_errors",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(self.config.retry_attempts),
            "MSWEA_GLOBAL_COST_LIMIT": "0",
            "MSWEA_GLOBAL_CALL_LIMIT": "0",
            # Otherwise litellm fetches the cost map from raw.githubusercontent
            # at import and blocks on the TLS handshake.
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }

    def _command_parts(self, task: str) -> list[str]:
        specs = [
            f"model.model_name={self.config.provider_prefix}{self._resolved_model}",
            # litellm-free backend (see _NATIVE_MODEL): params reach the wire verbatim, effort is never dropped.
            f"model.model_class={_NATIVE_MODEL_MODULE}.NativeOpenAIModel",
            "environment.environment_class=local",
            f"environment.cwd={self._cwd}",
            f"environment.timeout={self.config.command_timeout}",
            f"agent.step_limit={self.config.step_limit}",
            f"agent.cost_limit={self.config.cost_limit}",
            f"agent.wall_time_limit_seconds={self.config.wall_time_limit_seconds}",
        ]
        if self.config.max_tokens:
            specs.append(f"model.model_kwargs.max_tokens={self.config.max_tokens}")
        if self.config.reasoning_effort:
            specs.append(f"model.model_kwargs.reasoning_effort={self.config.reasoning_effort}")
        parts = [
            "mini",
            "--agent-class default",
            "-y",
            "--exit-immediately",
            # -c replaces the default config list; nothing leaks from bundled mini.yaml.
            f"-c {shlex.quote(self._config_file)}",
        ]
        parts += [f"-c {shlex.quote(s)}" for s in specs]
        parts.append(f"-o {shlex.quote(self._traj_file)}")
        parts.append(f'-t "$(cat {self._instruction_file})"')
        return parts

    def _parse_events(self, raw: str) -> list[dict[str, Any]]:
        return []

    def _read_traj(self) -> dict[str, Any]:
        raw = ""
        if self.msg_path and hasattr(self.env, "copy_out"):
            dest = self.msg_path.parent / f"{self.NAME}-traj-{self._turns}.json"
            try:
                self.env.copy_out(self._traj_file, str(dest))
                raw = dest.read_text(errors="replace")
            except Exception as e:
                self.logger.warning(f"[{self.NAME}] could not copy trajectory: {e}")
        if not raw:
            raw = self._read_pod_text(self._traj_file, max_bytes=20_000_000)
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self.logger.warning(f"[{self.NAME}] trajectory is not valid JSON: {e}")
            return {}

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        # Step/cost limit stops exit 0; must classify from info.exit_status.
        traj = self._read_traj()
        info = traj.get("info") or {}
        messages = traj.get("messages") or []
        exit_status = info.get("exit_status") or ""

        result_text = ""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text") or "" for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
            if isinstance(content, str) and content.strip():
                result_text = content.strip()
                break

        self._fold_usage(self._message_usage(messages))

        status = self.IDLE_STATUS if exit_status == "Submitted" else self.ERROR_STATUS
        if not result_text:
            status = self.ERROR_STATUS
            result_text = (
                f"(no agent message; mini-swe-agent rc={rc}, "
                f"exit_status={exit_status or 'killed'}, {len(messages)} messages)"
            )
        elif status == self.ERROR_STATUS:
            self.logger.warning(f"[{self.NAME}] turn ended as {exit_status or 'killed'} (rc={rc})")
        return result_text, status

    @staticmethod
    def _message_usage(messages: list[dict[str, Any]]) -> list[dict[str, int]]:
        # litellm's prompt_tokens is fresh+cache_read+cache_write; subtract to
        # avoid double-counting on cached runs.
        per_call: list[dict[str, int]] = []
        for msg in messages:
            usage = ((msg.get("extra") or {}).get("response") or {}).get("usage")
            if not isinstance(usage, dict):
                continue
            read = int(usage.get("cache_read_input_tokens") or 0)
            write = int(usage.get("cache_creation_input_tokens") or 0)
            prompt = int(usage.get("prompt_tokens") or 0)
            per_call.append(
                {
                    "input": max(prompt - read - write, 0),
                    "output": int(usage.get("completion_tokens") or 0),
                    "cache_read": read,
                    "cache_write": write,
                }
            )
        return per_call
