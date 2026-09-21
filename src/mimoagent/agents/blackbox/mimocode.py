"""MiMo-Code (Xiaomi's OpenCode fork) as a mimoagent blackbox agent.

Like :class:`CodexAgent`, this agent does not drive the mimoagent ``step()``
loop. It runs the MiMo-Code CLI (https://github.com/XiaomiMiMo/MiMo-Code)
inside the test pod, then hands control back. It satisfies the
:class:`mimoagent.Agent` protocol (``run(task) -> (status, message)``), so
environment setup, ``DatasetEnvironment.calculate_reward()`` (which grades
``git diff HEAD``) and ``save_traj`` all work without modification.

Lifecycle of ``run(task)``:

1. **install** — upload + run ``install-mimocode.sh`` in the pod (idempotent;
   downloads the MiMo-Code GitHub release selected by ``version`` — a single
   Bun-compiled binary, glibc or musl variant auto-detected — plus a static
   ripgrep; ``version`` defaults to the pinned release).
2. **stage** — create the logs dir. Provider routing needs no staged files:
   the whole provider config (base_url/api_key/model) is injected inline via
   the ``MIMOCODE_CONFIG_CONTENT`` env var each run.
3. **run** — ``mimo run --format json --dangerously-skip-permissions`` with
   the instruction on stdin (mimo reads the prompt from stdin when no
   positional message is given; a file redirect avoids argv length limits on
   large problem statements). The NDJSON event stream is redirected to
   ``mimocode.txt`` in the pod. MiMo-Code edits the repo in ``cwd``.
4. **collect** — copy ``mimocode.txt`` out next to the msg file, parse it for
   the final assistant message + per-step token usage, fold usage into
   ``model.token_stats``. ``self.messages`` stays minimal (task + final
   reply); the full event stream lives in the copied-out ``mimocode.txt``.

Auth/model routing reuses the ``model:`` config block (``model_name``,
``model_kwargs.base_url`` / ``api_key``) — one config drives both the native
and the MiMo-Code agent. The provider prefix is always stripped from
``model_name`` (``openai/mimo-v2.5-pro`` → ``mimo-v2.5-pro``) and the model is
addressed as ``gateway/<name>`` under a synthetic OpenAI-compatible provider.
"""

from __future__ import annotations

import json
import shlex
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

# Resources shipped alongside this module (uploaded into the pod at run time).
_RESOURCES = Path(__file__).resolve().parent / "resources"
_INSTALL_SCRIPT = _RESOURCES / "install-mimocode.sh"

# Paths inside the pod.
_POD_INSTALL_SCRIPT = "/tmp/mimo-install-mimocode.sh"
_POD_LOGS_DIR = "/tmp/mimo-mimocode-logs"  # doubles as the XDG_* home
_POD_INSTRUCTION_FILE = "/tmp/mimo-mimocode-instruction.txt"
_POD_OUTPUT_FILE = f"{_POD_LOGS_DIR}/mimocode.txt"
# Stable location the install script lays down the `mimo` binary.
_MIMOCODE_BIN_DIR = "/opt/mimo-mimocode/bin"

# Synthetic provider id in MIMOCODE_CONFIG_CONTENT; models are addressed as
# ``gateway/<model>``.
_PROVIDER_ID = "gateway"


@dataclass
class MimoCodeAgentConfig(AgentConfig):
    """Config for the MiMo-Code blackbox agent.

    Templates are unused (MiMo-Code carries its own system prompt) but kept on
    the base so unknown-key filtering and the shared dataclass surface still
    work.
    """

    # Where MiMo-Code runs / how long. ``cwd`` defaults to the env's working dir.
    cwd: str | None = None
    # MiMo-Code version to install: downloads the ``v<ver>`` GitHub release
    # tarball for this platform and verifies ``mimo --version``. There is no
    # "latest" fallback: every evaluation records an exact build.
    version: str = "0.1.12"
    # Protocol mimo speaks to the gateway:
    #   "anthropic"   → /v1/messages          (@ai-sdk/anthropic)
    #   "completions" → /v1/chat/completions  (@ai-sdk/openai-compatible)
    #   "responses"   → /v1/responses         (@ai-sdk/openai)
    protocol: str = "anthropic"
    # Provider-specific reasoning effort (mimo's ``--variant``: e.g. low |
    # medium | high | max | minimal). None omits the flag (provider default).
    variant: str | None = None
    # Model context window / max output tokens, written into the provider
    # config's model ``limit`` block. mimo uses ``context`` to decide when to
    # auto-compact the session. None leaves it to mimo's catalog default.
    context_limit: int | None = None
    output_limit: int | None = None
    # Extended thinking budget (Claude models): written into the model entry's
    # ``options.thinking`` and passed through as providerOptions. ``variant``
    # does not apply to custom gateway providers, so this is the only way to
    # enable thinking on them. None sends no thinking config.
    thinking_budget: int | None = None
    # Adaptive thinking with a fixed effort (Claude Opus 4.6+ native shape):
    # sends ``thinking.type: "adaptive"`` plus ``output_config.effort``. Lands
    # as a handwritten model variant that the run selects via ``--variant``
    # (mimo only auto-generates variants for catalog models). Mutually
    # exclusive with ``thinking_budget``. e.g. "max" | "high".
    thinking_effort: str | None = None
    # Max agentic iterations before mimo forces a text-only reply, written to
    # agent.<primary>.steps. None keeps upstream's default: unlimited.
    max_turns: int | None = None
    install_timeout: int = 1800
    run_timeout: int = 18000
    # Stall watchdog for the detached run: kill mimo when its event log saw no
    # write for this many seconds. 0 disables it; keep it above the longest
    # legitimate model-call silence.
    stall_timeout: int = 0
    # Extra env overrides layered on top of the derived MIMOCODE_*/XDG_* set.
    # Named ``extra_env`` (not ``env``) to avoid colliding with the Environment
    # positional arg threaded through make_agent / the constructor.
    extra_env: dict[str, str] | None = None
    # Skip the install step for images that already carry the binary.
    skip_install: bool = False
    # Extra environment for the install script (mirror overrides such as
    # GITHUB_BASE; see resources/install-common.sh).
    install_env: dict[str, str] | None = None


class MimoCodeAgent(BaseAgent):
    """Runs the MiMo-Code CLI inside the pod. Implements the mimoagent Agent protocol."""

    IDLE_STATUS = "Completed"
    # NDJSON event stream in the pod. Overwritten per run (each `mimo run` call
    # streams only its own turn), so it holds only the latest turn.
    ENV_TRAJECTORY_FILES = [_POD_OUTPUT_FILE]

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = MimoCodeAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._installed = False
        self._staged = False
        self._turns = 0

    # ----- Agent protocol ---------------------------------------------------

    def get_model_query_kwargs(self) -> dict:
        # No tool schema is fed to mimoagent's model layer — MiMo-Code owns its
        # own tools. Returning {} keeps save_traj's _agent_traj() shape valid.
        return {}

    def step(self) -> dict | None:  # pragma: no cover - not used by this agent
        """Unused: this agent overrides ``run`` rather than stepping a loop."""
        return None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Install (once), run MiMo-Code on ``task`` in the pod, collect the result.

        Returns ``(status, message)``:
        * ``"Completed"`` — mimo exited cleanly.
        * ``"InfraError"`` — pod transport failure (escalated like the native agent).
        * ``"MimoCodeError"`` — mimo exited non-zero / the event stream reported an error.

        The first call starts a fresh session; subsequent calls resume it via
        ``mimo run --continue`` (sessions are keyed on the working directory)
        so follow-up queries see the prior conversation, not just the edited
        workspace.
        """
        self.extra_template_vars |= {"task": task, **kwargs}

        try:
            if not self.config.skip_install and not self._installed:
                self._install()
                self._installed = True
            if not self._staged:
                self._stage()
                self._staged = True
            rc = self._run_mimocode(task)
        except TransportError as e:
            # Mirror DefaultAgent.execute_action: pod transport failures terminate.
            raise InfraError(str(e)) from e

        events = self._collect_events()
        result_text, status = self._collect(rc, events)
        # Keep self.messages minimal: the prompt + final reply. The full event
        # stream lives in the copied-out mimocode.txt.
        self.add_message("user", task)
        self.add_message("assistant", result_text)
        self._turns += 1
        return status, result_text

    # ----- internals --------------------------------------------------------

    @property
    def _cwd(self) -> str:
        if self.config.cwd:
            return self.config.cwd
        return getattr(self.env.config, "cwd", "/testbed")

    @property
    def _resolved_model(self) -> str:
        # ``model_name`` is the gateway serving name, sent verbatim — it may
        # itself contain slashes (channel-prefixed models like ``aws/claude-x``);
        # the synthetic provider re-namespaces it.
        return self.model.config.model_name

    @property
    def _model_kwargs(self) -> dict:
        return getattr(self.model.config, "model_kwargs", {}) or {}

    def _install(self) -> None:
        # No "latest" fallback: an empty version is a config error, not a
        # cue to grab a moving asset.
        version = self.config.version
        if not version:
            raise ValueError("mimocode: agent.version must be set (e.g. 0.1.12)")
        self.logger.info(f"[mimocode] installing mimocode scaffold ({version}) into pod...")
        self.env.copy_to(str(_INSTALL_SCRIPT), _POD_INSTALL_SCRIPT)
        path_prefix = stage_install_common(self.env) + installer_env_prefix(self.config.install_env)
        res = self.env.execute(
            f"{path_prefix}MIMOCODE_VERSION={shlex.quote(version)} bash {_POD_INSTALL_SCRIPT}",
            timeout=self.config.install_timeout,
        )
        out = res.get("output", "") or ""
        if "MIMOCODE_INSTALL_OK" not in out:
            raise RuntimeError(f"[mimocode] install failed (rc={res.get('returncode')}). Output tail:\n{out[-3000:]}")
        self.logger.info("[mimocode] install OK")

    def _copy_text_to_pod(self, text: str, remote_path: str) -> None:
        """Write ``text`` to ``remote_path`` in the pod via a local tempfile +
        ``copy_to`` (tar stream). A heredoc/argv would choke on large or
        non-utf8 content — same rationale as batch.py's patch upload."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(text)
            local_path = f.name
        try:
            self.env.copy_to(local_path, remote_path)
        finally:
            os.unlink(local_path)

    def _stage(self) -> None:
        self.env.execute(f"mkdir -p {_POD_LOGS_DIR}")

    # Maps the ``protocol`` config to the AI-SDK provider package mimo loads.
    # Verified request paths: anthropic → /v1/messages, completions →
    # /v1/chat/completions, responses → /v1/responses.
    _PROTOCOL_NPM = {
        "anthropic": "@ai-sdk/anthropic",
        "completions": "@ai-sdk/openai-compatible",
        "responses": "@ai-sdk/openai",
    }

    def _provider_config(self) -> str:
        """Inline mimocode config JSON: one provider carrying the shared
        ``model:`` block's base_url/api_key + the resolved model, speaking the
        wire protocol selected by ``config.protocol``."""
        base_url = (self._model_kwargs.get("base_url") or "").rstrip("/")
        if base_url and not base_url.endswith("/v1"):
            # All three AI-SDK providers expect the /v1 prefix in baseURL
            # (they append /messages, /chat/completions, /responses).
            base_url += "/v1"
        npm = self._PROTOCOL_NPM.get(self.config.protocol)
        if npm is None:
            raise RuntimeError(
                f"[mimocode] unknown protocol: {self.config.protocol!r} (expected one of {sorted(self._PROTOCOL_NPM)})"
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
        if len(limit) == 1:
            # mimo's config schema rejects a partial limit block ("expected
            # number, received undefined"), which would kill every session at
            # startup. Fail here with the actionable message instead.
            raise ValueError(
                f"mimocode: context_limit and output_limit must be set together (got only {next(iter(limit))})"
            )
        if limit:
            model_entry["limit"] = limit
        if self.config.thinking_budget and self.config.thinking_effort:
            raise RuntimeError(
                "[mimocode] thinking_budget and thinking_effort are mutually "
                "exclusive (budget-based vs adaptive-effort thinking)"
            )
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
        config: dict[str, Any] = {
            "provider": {
                _PROVIDER_ID: {
                    "npm": npm,
                    "name": "Gateway",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": self._model_kwargs.get("api_key") or "",
                    },
                    "models": {model: model_entry},
                }
            }
        }
        if self.config.max_turns:
            # Both primaries are reachable in one run (plan_enter/plan_exit
            # switches between them), so cap them together.
            config["agent"] = {name: {"steps": self.config.max_turns} for name in ("build", "plan")}
        return json.dumps(config, ensure_ascii=False)

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "MIMOCODE_CONFIG_CONTENT": self._provider_config(),
            "MIMOCODE_DISABLE_AUTOUPDATE": "1",
            # Evaluation runs drive the plain agent loop; keep the codex-mode
            # session behaviour off explicitly (overridable via extra_env).
            "MIMOCODE_CODEX_MODE": "false",
            # mimo fetches a models.dev catalog on startup. Pods have no public
            # egress and the SYN just blackholes, hanging the CLI forever.
            # Point the fetch at a local closed port so it fails instantly and
            # mimo falls back to its bundled catalog snapshot.
            "MIMOCODE_MODELS_URL": "http://127.0.0.1:9/api.json",
            # Keep session DB / config / cache hermetic under the logs dir
            # (survives across turns; wiped with the pod).
            "XDG_DATA_HOME": f"{_POD_LOGS_DIR}/xdg/data",
            "XDG_CONFIG_HOME": f"{_POD_LOGS_DIR}/xdg/config",
            "XDG_CACHE_HOME": f"{_POD_LOGS_DIR}/xdg/cache",
            "XDG_STATE_HOME": f"{_POD_LOGS_DIR}/xdg/state",
        }
        # Router lives on a raw pod IP; task-pod proxy env (no_proxy often
        # overridden by the agent config) would send API calls to the proxy →
        # ConnectionRefused / 0 turns. Exempt the router host for this process.
        _host = urlparse(self._model_kwargs.get("base_url") or "").hostname
        if _host:
            env["no_proxy"] = f"{_host},localhost,127.0.0.1"
            env["NO_PROXY"] = env["no_proxy"]
        if self.config.extra_env:
            env.update(self.config.extra_env)
        return env

    def _run_mimocode(self, task: str) -> int:
        # mimo reads the prompt from stdin when no positional message is given;
        # a file redirect avoids argv length / quoting limits on large problem
        # statements.
        self._copy_text_to_pod(task, _POD_INSTRUCTION_FILE)

        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self._build_env().items())

        parts = [
            "mimo",
            "run",
            "--format json",
            "--dangerously-skip-permissions",
            f"-m {shlex.quote(f'{_PROVIDER_ID}/{self._resolved_model}')}",
        ]
        if self._turns > 0:
            # Resume the last session in this working dir so the follow-up
            # query continues the same conversation, not a fresh one.
            parts.append("--continue")
        variant = self.config.variant or self.config.thinking_effort
        if variant:
            parts.append(f"--variant {shlex.quote(variant)}")

        cmd = (
            f'export PATH={_MIMOCODE_BIN_DIR}:"$PATH"; '
            f"{env_prefix} {' '.join(parts)} "
            f"< {_POD_INSTRUCTION_FILE} > {_POD_OUTPUT_FILE} 2>&1"
        )
        self.logger.info(f"[mimocode] running (turn {self._turns + 1}) in {self._cwd}")
        rc, _ = run_detached(
            self.env,
            cmd,
            cwd=self._cwd,
            timeout=self.config.run_timeout,
            tag="mimocode",
            log_hint="event log: agent_msgs/mimocode.txt",
            logger=self.logger,
            idle_files=[_POD_OUTPUT_FILE],
            idle_timeout=self.config.stall_timeout,
        )
        return rc

    # ----- event collection & parsing ----------------------------------------

    def _collect_events(self) -> list[dict[str, Any]]:
        """Copy mimocode.txt out of the pod (next to the msg file, for
        inspection) and parse its NDJSON events. Falls back to catting a tail
        of the file through the exec transport when ``copy_out`` is
        unavailable/fails."""
        raw = ""
        if self.msg_path and hasattr(self.env, "copy_out"):
            dest = self.msg_path.parent / "mimocode.txt"
            try:
                self.env.copy_out(_POD_OUTPUT_FILE, str(dest))
                self.logger.info(f"[mimocode] copied event log to {dest}")
                raw = dest.read_text(errors="replace")
            except Exception as e:
                self.logger.warning(f"[mimocode] could not copy mimocode.txt: {e}")
        if not raw:
            try:
                res = self.env.execute(f"tail -c 2000000 {_POD_OUTPUT_FILE} 2>/dev/null || true")
                raw = res.get("output", "") or ""
            except Exception as e:
                self.logger.warning(f"[mimocode] could not read mimocode.txt: {e}")
        return self._parse_ndjson(raw)

    @staticmethod
    def _parse_ndjson(raw: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or not s.startswith("{"):
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return events

    def _collect(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """Derive (result_text, status) from the event stream + exit code.

        Event shapes (``mimo run --format json``, one JSON object per line):
        ``{"type": "text", "part": {"text": ...}}`` — assistant text;
        ``{"type": "step_finish", "part": {"tokens": {...}}}`` — per-step usage;
        ``{"type": "error", "error": {...}}`` — API/tool failure.
        """
        status = self.IDLE_STATUS if rc == 0 else "MimoCodeError"

        result_text = ""
        error_text = ""
        for ev in events:
            etype = ev.get("type")
            if etype == "text":
                text = ((ev.get("part") or {}).get("text") or "").strip()
                if text:
                    result_text = text
            elif etype == "error":
                error_text = json.dumps(ev, ensure_ascii=False)

        if error_text and not result_text:
            status = "MimoCodeError"
            result_text = error_text
        self._fold_usage(events)

        if not result_text:
            result_text = f"(no agent message; mimo rc={rc}, {len(events)} events)"
        return result_text, status

    def _fold_usage(self, events: list[dict[str, Any]]) -> None:
        """Sum ``step_finish`` token blocks onto ``model.token_stats`` +
        ``n_calls`` so ``save_traj`` reports real numbers instead of zeros."""
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        n_steps = 0
        for ev in events:
            if ev.get("type") != "step_finish":
                continue
            tokens = (ev.get("part") or {}).get("tokens") or {}
            if not isinstance(tokens, dict):
                continue
            n_steps += 1
            totals["input"] += int(tokens.get("input", 0) or 0)
            totals["output"] += int(tokens.get("output", 0) or 0)
            cache = tokens.get("cache") or {}
            totals["cache_read"] += int(cache.get("read", 0) or 0)
            totals["cache_write"] += int(cache.get("write", 0) or 0)
        stats = getattr(self.model, "token_stats", None)
        if stats is not None and n_steps:
            stats.input_tokens += totals["input"]
            stats.output_tokens += totals["output"]
            stats.cache_read_tokens += totals["cache_read"]
            stats.cache_creation_tokens += totals["cache_write"]
        if n_steps:
            self.extra_template_vars["mimocode_usage"] = totals
            # Each step_finish ≈ one model call; advance the model's counter
            # so the trajectory's api_calls reflects the MiMo-Code session.
            try:
                self.model.n_calls += n_steps
            except Exception:
                pass
