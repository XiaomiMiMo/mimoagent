"""Shared skeleton for payload-installed blackbox harness adapters.

The first four blackbox agents (claude-code, codex, mimocode, pi) each carry
their own install/stage/run/collect plumbing. The 2026-08 harness batch (grok,
kimi-code, kimi-cli, openclaw, hermes, opencode, omp) adds seven more adapters
that differ only in configuration and event parsing, so the shared lifecycle
lives here once:

1. **install** — run this harness's own ``resources/install-<name>.sh`` in the
   pod (same shape as install-claude-code.sh / install-codex.sh: idempotent,
   prints ``<NAME>_INSTALL_OK``). By default it installs the pinned version
   from the upstream public source (npm registry, GitHub releases, PyPI)
   under ``/opt/mimo-<name>``; ``payload_url`` / ``payload_path`` are optional
   overrides that install a prebuilt payload tarball instead (for offline or
   mirrored setups — see ``scripts/harness_payloads/build_harness_payloads.sh``).
2. **stage** — write the harness's provider/config files into the pod
   (per-harness hook).
3. **run** — execute one non-interactive task turn with stdin instruction +
   NDJSON/text output redirect (per-harness command; shared transport
   handling).
4. **collect** — copy the raw log next to the msg file, parse events, fold
   token usage into ``model.token_stats`` / ``shared_stats`` (per-harness
   parsing over shared helpers).

Every adapter satisfies the :class:`mimoagent.Agent` protocol
(``run(task) -> (status, message)``) so environment setup,
``DatasetEnvironment.calculate_reward()`` and ``save_traj`` work unchanged.
Auth/model routing reuses the shared ``model:`` config block (``model_name``,
``model_kwargs.base_url`` / ``api_key``); all seven speak Anthropic
``/v1/messages`` to the gateway so Claude extended-thinking signatures are
replayed by the harness's own Anthropic driver.
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

_RESOURCES = Path(__file__).resolve().parent / "resources"


@dataclass
class HarnessAgentConfig(AgentConfig):
    """Config fields shared by every payload-installed harness adapter.

    Templates are unused (each harness carries its own system prompt) but kept
    on the base so unknown-key filtering and the shared dataclass surface still
    work.
    """

    # Where the harness runs / how long. ``cwd`` defaults to the env's cwd.
    cwd: str | None = None
    # Pinned harness version. No "latest": every rollout records an exact
    # build, and the install verifies the binary reports exactly this.
    version: str = ""
    # Optional offline override: a local payload tarball (built by
    # scripts/harness_payloads/build_harness_payloads.sh), uploaded into the
    # pod via copy_to. Normal configs leave this unset so the install script
    # installs from the upstream public source.
    payload_path: str | None = None
    # Optional offline override: the pod downloads this payload URL instead of
    # receiving an upload (an internal mirror of the built payload).
    payload_url: str | None = None
    # Extra environment for the install script, e.g. mirror overrides such as
    # NPM_REGISTRY / PIP_INDEX_URL / NODE_DIST / GITHUB_BASE (see
    # resources/install-common.sh for the full list).
    install_env: dict[str, str] | None = None
    install_timeout: int = 1800
    run_timeout: int = 18000
    # Stall watchdog for the detached run: kill the harness when its event log
    # (and stderr side file) saw no write for this many seconds. 0 disables it.
    # Keep it well above the longest legitimate silence (a slow model call);
    # harnesses that print one JSON document at exit only write at the end.
    stall_timeout: int = 0
    # Extra env overrides layered on top of the derived set. Named
    # ``extra_env`` (not ``env``) to avoid colliding with the Environment
    # positional arg threaded through make_agent / the constructor.
    extra_env: dict[str, str] | None = None
    # The payload may be pre-baked into the image; skip upload + install.
    skip_install: bool = False


class HarnessAgent(BaseAgent):
    """Shared lifecycle for payload-installed blackbox harnesses.

    Subclasses set the class attributes below and implement the four hooks:
    ``_stage_files()``, ``_command_parts(task)``, ``_harness_env()`` and
    ``_parse_result(rc, events)``.
    """

    # -- subclass contract ----------------------------------------------------
    # Short id, e.g. "grok". Drives the pod paths, the log filenames, the
    # resources/install-<name>.sh script and its <NAME>_VERSION / <NAME>_PAYLOAD
    # / <NAME>_INSTALL_OK env-knob prefix.
    NAME = ""
    ERROR_STATUS = "HarnessError"  # e.g. "GrokError"

    IDLE_STATUS = "Completed"
    RUNS_INSIDE_ENV = True
    # True routes stderr to a side file appended after the run, so stdout
    # stays machine-parseable (harnesses that print one JSON document rather
    # than an NDJSON stream).
    SPLIT_STDERR = False

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = HarnessAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._installed = False
        self._staged = False
        self._turns = 0

    # -- derived pod paths -----------------------------------------------------

    @property
    def _install_dir(self) -> str:
        return f"/opt/mimo-{self.NAME}"

    @property
    def _bin_dir(self) -> str:
        return f"/opt/mimo-{self.NAME}/bin"

    @property
    def _logs_dir(self) -> str:
        return f"/tmp/mimo-{self.NAME}-logs"

    @property
    def _instruction_file(self) -> str:
        return f"/tmp/mimo-{self.NAME}-instruction.txt"

    @property
    def _output_file(self) -> str:
        return f"{self._logs_dir}/{self.NAME}.txt"

    # ENV_TRAJECTORY_FILES must be a class attribute for sidecars that inspect
    # it before instantiation; subclasses set it to [f"/tmp/mimo-<name>-logs/<name>.txt"].

    # -- Agent protocol ---------------------------------------------------------

    def get_model_query_kwargs(self) -> dict:
        # No tool schema is fed to mimoagent's model layer — the harness owns
        # its own tools. Returning {} keeps save_traj's _agent_traj() shape valid.
        return {}

    def step(self) -> dict | None:  # pragma: no cover - run() owns the loop
        return None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Install (once), run one task turn in the pod, collect the result.

        Returns ``(status, message)``: ``"Completed"`` on a clean exit,
        ``ERROR_STATUS`` on a non-zero exit / error event, and raises
        :class:`InfraError` on pod transport failure (mirrors MiMo-Code-Lite).
        """
        self.extra_template_vars |= {"task": task, **kwargs}
        try:
            if not self.config.skip_install and not self._installed:
                self._install()
                self._installed = True
            if not self._staged:
                self.env.execute(f"mkdir -p {self._logs_dir}")
                self._stage_files()
                self._staged = True
            rc = self._run_harness(task)
        except TransportError as e:
            raise InfraError(str(e)) from e

        events = self._collect_events()
        result_text, status = self._parse_result(rc, events)
        # Keep self.messages minimal: the prompt + final reply. The full event
        # stream lives in the copied-out <name>.txt.
        self.add_message("user", task)
        self.add_message("assistant", result_text)
        self._turns += 1
        return status, result_text

    # -- hooks -------------------------------------------------------------------

    def _stage_files(self) -> None:
        """Write per-harness config files into the pod (once per session)."""

    def _command_parts(self, task: str) -> list[str]:  # pragma: no cover - abstract
        """CLI argv fragments for one non-interactive task turn.

        The instruction file is already uploaded; parts that read the prompt
        from stdin get it via redirect in ``_run_harness``."""
        raise NotImplementedError

    def _harness_env(self) -> dict[str, str]:
        """Per-harness env (home dirs, telemetry kills, ...)."""
        return {}

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:  # pragma: no cover - abstract
        """Derive ``(result_text, status)`` and fold token usage."""
        raise NotImplementedError

    # -- shared internals ----------------------------------------------------------

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

    @property
    def _api_key(self) -> str:
        return self._model_kwargs.get("api_key") or "dummy-key"

    def _gateway_base_url(self, *, v1: bool) -> str:
        """The shared ``model:`` block's base_url, with or without ``/v1``.

        ``v1=True`` returns ``http://gw/v1`` (drivers that append
        ``/messages``); ``v1=False`` returns ``http://gw`` (drivers that
        append ``/v1/messages``, e.g. wrappers of the official Anthropic SDK).
        """
        base_url = (self._model_kwargs.get("base_url") or "").rstrip("/")
        if not base_url:
            raise RuntimeError(f"[{self.NAME}] model.model_kwargs.base_url is required")
        if v1:
            return base_url if base_url.endswith("/v1") else base_url + "/v1"
        return base_url.removesuffix("/v1")

    @property
    def _install_script(self) -> Path:
        """Each harness owns an ``install-<name>.sh`` next to the other blackbox
        installers: it pins its own default version, knows how to probe the
        unpacked tree, and carries the harness-specific notes. Same shape as
        install-claude-code.sh / install-codex.sh."""
        script = _RESOURCES / f"install-{self.NAME}.sh"
        if not script.is_file():
            raise NotImplementedError(f"{self.NAME}: missing {script.name}")
        return script

    @property
    def _env_prefix(self) -> str:
        """Env-knob / sentinel prefix used by this harness's install script."""
        return self.NAME.upper().replace("-", "_")

    def _install(self) -> None:
        version = self.config.version
        if not version:
            raise ValueError(f"{self.NAME}: agent.version must be pinned")
        payload_path = self.config.payload_path
        payload_url = self.config.payload_url
        if payload_path and payload_url:
            raise ValueError(
                f"{self.NAME}: set at most one of agent.payload_path / "
                "agent.payload_url (neither = the script's upstream public source)"
            )

        script = self._install_script
        pod_script = f"/tmp/mimo-install-{self.NAME}.sh"
        pod_payload = f"/tmp/mimo-{self.NAME}-payload.tar.gz"
        self.env.copy_to(str(script), pod_script)
        # The shared helper library every installer sources.
        path_prefix = stage_install_common(self.env) + installer_env_prefix(self.config.install_env)

        prefix = self._env_prefix
        env_parts = [f"{prefix}_VERSION={shlex.quote(version)}"]
        if not payload_path and not payload_url:
            self.logger.info(f"[{self.NAME}] installing {version} from its upstream source ({script.name})...")
        elif payload_url:
            self.logger.info(f"[{self.NAME}] installing {version} from {payload_url}...")
            env_parts.append(f"{prefix}_PAYLOAD_URL={shlex.quote(payload_url)}")
        else:
            local = Path(payload_path).expanduser()
            if not local.is_file():
                raise RuntimeError(f"[{self.NAME}] payload not found: {local}")
            # Skip the (potentially large) upload when the pod already holds
            # this exact version — the marker check is what makes multi-turn
            # and pre-warmed pods cheap.
            probe = self.env.execute(f"cat {self._install_dir}/.mimo-payload-version 2>/dev/null || true")
            if (probe.get("output", "") or "").strip() == version:
                self.logger.info(f"[{self.NAME}] payload {version} already installed")
            else:
                size_mb = local.stat().st_size / 1e6
                self.logger.info(f"[{self.NAME}] uploading payload {local.name} ({size_mb:.0f} MB) into pod...")
                self.env.copy_to(str(local), pod_payload)
            env_parts.append(f"{prefix}_PAYLOAD={shlex.quote(pod_payload)}")

        res = self.env.execute(
            f"{path_prefix}{' '.join(env_parts)} bash {pod_script}",
            timeout=self.config.install_timeout,
        )
        out = res.get("output", "") or ""
        if f"{prefix}_INSTALL_OK" not in out:
            raise RuntimeError(
                f"[{self.NAME}] install failed (rc={res.get('returncode')}). Output tail:\n{out[-3000:]}"
            )
        self.logger.info(f"[{self.NAME}] install OK ({version})")

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

    def _build_env(self) -> dict[str, str]:
        env = dict(self._harness_env())
        # Router lives on a raw pod IP; task-pod proxy env (no_proxy often
        # overridden by the agent config) would send API calls to the proxy →
        # ConnectionRefused / 0 turns. Exempt the router host for this process.
        host = urlparse(self._model_kwargs.get("base_url") or "").hostname
        if host:
            no_proxy = f"{host},localhost,127.0.0.1"
            env["no_proxy"] = no_proxy
            env["NO_PROXY"] = no_proxy
        if self.config.extra_env:
            env.update(self.config.extra_env)
        return env

    def _run_harness(self, task: str) -> int:
        self._copy_text_to_pod(task, self._instruction_file)
        # Env goes through a sourced file, not inline in the command string:
        # api keys in argv are world-readable via ps on shared hosts.
        env_file = f"{self._logs_dir}/{self.NAME}.env"
        env_text = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in self._build_env().items())
        self._copy_text_to_pod(env_text, env_file)
        parts = self._command_parts(task)
        if self.SPLIT_STDERR:
            # keep stdout machine-parseable (e.g. one pretty-printed JSON
            # envelope): stderr diagnostics are appended after the run.
            err_file = f"{self._logs_dir}/{self.NAME}.stderr.txt"
            redirect = (
                f"< {self._instruction_file} > {self._output_file} 2> {err_file}; "
                f"rc=$?; cat {err_file} >> {self._output_file}; exit $rc"
            )
        else:
            redirect = f"< {self._instruction_file} > {self._output_file} 2>&1"
        cmd = f'export PATH={self._bin_dir}:"$PATH"; . {env_file}; {" ".join(parts)} {redirect}'
        self.logger.info(f"[{self.NAME}] running (turn {self._turns + 1}) in {self._cwd}")
        idle_files = [self._output_file]
        if self.SPLIT_STDERR:
            idle_files.append(f"{self._logs_dir}/{self.NAME}.stderr.txt")
        rc, _ = run_detached(
            self.env,
            cmd,
            cwd=self._cwd,
            timeout=self.config.run_timeout,
            tag=self.NAME,
            log_hint=f"event log: agent_msgs/{self.NAME}.txt",
            logger=self.logger,
            idle_files=idle_files,
            idle_timeout=self.config.stall_timeout,
        )
        return rc

    # -- event collection & usage ------------------------------------------------

    def _collect_events(self) -> list[dict[str, Any]]:
        """Copy the event log out of the pod (next to the msg file, for
        inspection) and parse its NDJSON lines. Falls back to catting a tail
        through the exec transport when ``copy_out`` is unavailable/fails."""
        raw = ""
        if self.msg_path and hasattr(self.env, "copy_out"):
            dest = self.msg_path.parent / f"{self.NAME}.txt"
            try:
                self.env.copy_out(self._output_file, str(dest))
                self.logger.info(f"[{self.NAME}] copied event log to {dest}")
                raw = dest.read_text(errors="replace")
            except Exception as e:
                self.logger.warning(f"[{self.NAME}] could not copy {self.NAME}.txt: {e}")
        if not raw:
            try:
                res = self.env.execute(f"tail -c 2000000 {self._output_file} 2>/dev/null || true")
                raw = res.get("output", "") or ""
            except Exception as e:
                self.logger.warning(f"[{self.NAME}] could not read {self.NAME}.txt: {e}")
        return self._parse_events(raw)

    def _parse_events(self, raw: str) -> list[dict[str, Any]]:
        """Turn the raw event log into event dicts (default: NDJSON lines).
        Harnesses with a different output shape override this."""
        return self._parse_ndjson(raw)

    @staticmethod
    def _content_text(content: Any) -> str:
        """Assistant message text from either a plain string or a block list
        (``[{"type":"text","text":...}, {"type":"think",...}]``) — the kimi
        harnesses switch shapes depending on whether thinking is on."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                block.get("text") or "" for block in content if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _parse_ndjson(raw: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                events.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return events

    def _read_pod_text(self, path: str, max_bytes: int = 2_000_000) -> str:
        """Best-effort read of pod file(s) through the exec transport (for
        session/usage files the harness writes outside the event log).

        ``path`` is passed to the shell unquoted so adapter-internal constants
        may use globs / substitutions — never feed it user-derived strings."""
        try:
            res = self.env.execute(f"cat {path} 2>/dev/null | tail -c {max_bytes}")
            return res.get("output", "") or ""
        except Exception as e:
            self.logger.warning(f"[{self.NAME}] could not read {path}: {e}")
            return ""

    def _fold_usage(self, per_call: list[dict[str, int]]) -> None:
        """Fold per-model-call usage dicts (keys: input/output/cache_read/
        cache_write) into ``model.token_stats``, ``n_calls`` and the
        batch-level ``shared_stats`` aggregator, so harness sessions count
        toward token progress and MIMOAGENT_GLOBAL_CALL_LIMIT like every other
        scaffold."""
        if not per_call:
            return
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for call in per_call:
            for key in totals:
                totals[key] += int(call.get(key, 0) or 0)
        stats = getattr(self.model, "token_stats", None)
        if stats is not None:
            stats.input_tokens += totals["input"]
            stats.output_tokens += totals["output"]
            stats.cache_read_tokens += totals["cache_read"]
            stats.cache_creation_tokens += totals["cache_write"]
        self.extra_template_vars[f"{self.NAME}_usage"] = totals
        try:
            self.model.n_calls += len(per_call)
        except Exception:
            pass
        shared = getattr(self.model, "shared_stats", None)
        if shared is not None:
            from mimoagent.models import TokenStats

            for call in per_call:
                shared.add(
                    TokenStats(
                        input_tokens=int(call.get("input", 0) or 0),
                        output_tokens=int(call.get("output", 0) or 0),
                        cache_read_tokens=int(call.get("cache_read", 0) or 0),
                        cache_creation_tokens=int(call.get("cache_write", 0) or 0),
                    )
                )
