"""OpenAI Codex CLI (upstream) as a mimoagent blackbox agent.

Like :class:`ClaudeCodeAgent`, this agent does not drive the mimoagent
``step()`` loop. It runs OpenAI's Codex CLI inside the test pod, then hands
control back. It satisfies the :class:`mimoagent.Agent` protocol
(``run(task) -> (status, message)``), so environment setup,
``DatasetEnvironment.calculate_reward()`` (which grades ``git diff HEAD``) and
``save_traj`` all work without modification.

Lifecycle of ``run(task)``:

1. **install** — upload + run ``install-codex.sh`` in the pod (idempotent;
   downloads the upstream ``codex-package`` bundle — static musl ``codex`` plus
   its ``codex-code-mode-host`` sibling, ``rg``, ``bwrap``, ``zsh`` — from the
   ``openai/codex`` GitHub release for the pinned ``version`` and unpacks it
   under ``/opt/mimo-codex``). The Code Mode host is mandatory for models
   whose built-in ``tool_mode`` is ``code_mode_only`` (gpt-6-astra): codex
   spawns it from its own executable's directory and fails closed (no tools at
   all) when it is absent.
2. **stage** — write ``$CODEX_HOME/config.toml`` (custom ``openai_compat``
   model provider — codex >=0.130 ignores ``OPENAI_BASE_URL``), ``auth.json``,
   the task instruction file, and optionally ``instructions.md`` (a system-prompt
   override wired via ``model_instructions_file`` when ``instruction_path`` is
   set; otherwise codex keeps its built-in system prompt).
3. **run** — ``codex exec --json`` with the instruction on stdin (codex reads
   the prompt from stdin when no positional PROMPT is given; a file redirect
   avoids argv length limits on large problem statements). The NDJSON event
   stream is redirected to ``codex.txt`` in the pod. Codex edits the repo in
   ``cwd``.
4. **collect** — copy ``codex.txt`` out next to the msg file, parse it for the
   final assistant message + token usage, fold usage into
   ``model.token_stats``. ``self.messages`` stays minimal (task + final
   reply); the full event stream lives in the copied-out ``codex.txt``.

Auth/model routing reuses the ``model:`` config block (``model_name``,
``model_kwargs.base_url`` / ``api_key``) — one config drives both the native
and the Codex agent. The provider prefix is always stripped from
``model_name`` (``openai/gpt-5.5`` → ``gpt-5.5``), the way the upstream CLI
expects bare model ids.
"""

from __future__ import annotations

import json
import re
import shlex
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimoagent import Environment, Model
from mimoagent.agents.base import AgentConfig, BaseAgent, InfraError
from mimoagent.agents.blackbox.detached import run_detached
from mimoagent.agents.blackbox.install_common import installer_env_prefix, stage_install_common
from mimoagent.environments import TransportError

# Resources shipped alongside this module (uploaded into the pod at run time).
_RESOURCES = Path(__file__).resolve().parent / "resources"
_INSTALL_SCRIPT = _RESOURCES / "install-codex.sh"

# Paths inside the pod.
_POD_INSTALL_SCRIPT = "/tmp/mimo-install-codex.sh"
_POD_LOGS_DIR = "/tmp/mimo-codex-logs"  # doubles as $CODEX_HOME
_POD_INSTRUCTION_FILE = "/tmp/mimo-codex-instruction.txt"
_POD_OUTPUT_FILE = f"{_POD_LOGS_DIR}/codex.txt"
_POD_INSTRUCTIONS_MD = f"{_POD_LOGS_DIR}/instructions.md"
# Bundle layout the install script lays down (mirrors the official installer):
# bin/{codex,codex-code-mode-host}, codex-path/rg, codex-resources/{bwrap,zsh}.
# Default Codex CLI release (see install-codex.sh for the source).
DEFAULT_CODEX_VERSION = "0.154.0"
_CODEX_ROOT = "/opt/mimo-codex"
_CODEX_BIN_DIR = f"{_CODEX_ROOT}/bin"
# Extra tools codex expects on the agent's PATH (ripgrep).
_CODEX_PATH_DIR = f"{_CODEX_ROOT}/codex-path"


@dataclass
class CodexAgentConfig(AgentConfig):
    """Config for the Codex blackbox agent.

    Templates are unused (Codex carries its own system prompt) but kept on the
    base so unknown-key filtering and the shared dataclass surface still work.
    """

    # Where Codex runs / how long. ``cwd`` defaults to the env's working dir.
    cwd: str | None = None
    # Codex CLI version to install: a pinned "x.y.z" downloads the
    # ``rust-v<ver>`` GitHub release bundle and verifies --version; the literal
    # "latest" resolves to the newest release at install time.
    version: str = DEFAULT_CODEX_VERSION
    effort: str | None = "high"  # model_reasoning_effort: low | medium | high
    # Protocol codex speaks to the gateway when a custom base_url is set.
    wire_api: str = "responses"  # responses | chat
    install_timeout: int = 1800
    run_timeout: int = 18000
    # Stall watchdog for the detached run: kill codex when its event log saw no
    # write for this many seconds. 0 disables it; keep it above the longest
    # legitimate model-call silence.
    stall_timeout: int = 0
    # Enable codex's unified_exec experimental feature.
    unified_exec: bool = True
    # Extra env overrides layered on top of the derived OPENAI_*/CODEX_* set.
    # Named ``extra_env`` (not ``env``) to avoid colliding with the Environment
    # positional arg threaded through make_agent / the constructor.
    extra_env: dict[str, str] | None = None
    # Skip the install step for images that already carry the bundle.
    skip_install: bool = False
    # Extra environment for the install script (mirror overrides such as
    # GITHUB_BASE; see resources/install-common.sh).
    install_env: dict[str, str] | None = None
    # Optional path to a system-prompt file. When set, it's uploaded and wired
    # via codex's ``model_instructions_file`` (which *replaces* the built-in
    # instructions). When None (default), codex uses its built-in system prompt
    # and we don't set the key at all.
    instruction_path: str | None = None


class CodexAgent(BaseAgent):
    """Runs the Codex CLI inside the pod. Implements the mimoagent Agent protocol."""

    IDLE_STATUS = "Completed"
    # NDJSON event stream in the pod. Overwritten per run (each Codex call is
    # a fresh session), so it holds only the latest turn.
    ENV_TRAJECTORY_FILES = [_POD_OUTPUT_FILE]

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = CodexAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._installed = False
        self._staged = False
        self._turns = 0

    # ----- Agent protocol ---------------------------------------------------

    def get_model_query_kwargs(self) -> dict:
        # No tool schema is fed to mimoagent's model layer — Codex owns its own
        # tools. Returning {} keeps save_traj's _agent_traj() shape valid.
        return {}

    def step(self) -> dict | None:  # pragma: no cover - not used by this agent
        """Unused: this agent overrides ``run`` rather than stepping a loop."""
        return None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Install (once), run Codex on ``task`` in the pod, collect the result.

        Returns ``(status, message)``:
        * ``"Completed"`` — codex exited cleanly.
        * ``"InfraError"`` — pod transport failure (escalated like the native agent).
        * ``"CodexError"`` — codex exited non-zero / the event stream reported an error.

        The first call starts a fresh Codex session; subsequent calls resume it
        via ``codex exec resume --last`` so follow-up queries see the prior
        conversation + reasoning (not just the edited workspace).
        """
        self.extra_template_vars |= {"task": task, **kwargs}

        try:
            if not self.config.skip_install and not self._installed:
                self._install()
                self._installed = True
            if not self._staged:
                self._stage()
                self._staged = True
            rc = self._run_codex(task)
        except TransportError as e:
            # Mirror DefaultAgent.execute_action: pod transport failures terminate.
            raise InfraError(str(e)) from e

        events = self._collect_events()
        result_text, status = self._collect(rc, events)
        # Keep self.messages minimal: the prompt + final reply. The full event
        # stream lives in the copied-out codex.txt.
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
        # ``model_name`` is the gateway serving name, sent verbatim.
        return self.model.config.model_name

    @property
    def _model_kwargs(self) -> dict:
        return getattr(self.model.config, "model_kwargs", {}) or {}

    def _install(self) -> None:
        version = self.config.version or DEFAULT_CODEX_VERSION
        self.logger.info(f"[codex] installing codex scaffold ({version}) into pod...")
        self.env.copy_to(str(_INSTALL_SCRIPT), _POD_INSTALL_SCRIPT)
        path_prefix = stage_install_common(self.env) + installer_env_prefix(self.config.install_env)
        res = self.env.execute(
            f"{path_prefix}CODEX_VERSION={shlex.quote(version)} bash {_POD_INSTALL_SCRIPT}",
            timeout=self.config.install_timeout,
        )
        out = res.get("output", "") or ""
        if "CODEX_INSTALL_OK" not in out:
            raise RuntimeError(f"[codex] install failed (rc={res.get('returncode')}). Output tail:\n{out[-3000:]}")
        self.logger.info("[codex] install OK")

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
        """Write $CODEX_HOME/{config.toml,auth.json[,instructions.md]} into the pod (once)."""
        self.env.execute(f"mkdir -p {_POD_LOGS_DIR}")

        api_key = self._model_kwargs.get("api_key") or ""
        base_url = (self._model_kwargs.get("base_url") or "").rstrip("/")

        config_toml = 'web_search = "disabled"\n'

        # Optional system-prompt override. When instruction_path is set, ship
        # that file and point codex at it via model_instructions_file (which
        # *replaces* the built-in instructions). When unset, leave the key out
        # so codex uses its built-in system prompt.
        instr = self.config.instruction_path
        if instr:
            local = Path(instr).expanduser()
            if not local.is_file():
                raise RuntimeError(f"[codex] instruction_path not found: {local}")
            self.env.copy_to(str(local), _POD_INSTRUCTIONS_MD)
            # NB: top-level keys must precede any [table] section in TOML.
            config_toml += f'model_instructions_file = "{_POD_INSTRUCTIONS_MD}"\n'

        if base_url:
            # codex >=0.130 ignores OPENAI_BASE_URL and hardcodes the default
            # openai provider; a custom provider in config.toml is the only
            # reliable routing knob.
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            config_toml += (
                "[model_providers.openai_compat]\n"
                'name = "OpenAI Compat"\n'
                f'base_url = "{base_url}"\n'
                'env_key = "OPENAI_API_KEY"\n'
                f'wire_api = "{self.config.wire_api}"\n'
            )
            self._pin_gateway_host(base_url)
        self._copy_text_to_pod(config_toml, f"{_POD_LOGS_DIR}/config.toml")
        # codex refuses to run without a login; the env_key above is what
        # actually authenticates requests.
        self._copy_text_to_pod(json.dumps({"OPENAI_API_KEY": api_key}), f"{_POD_LOGS_DIR}/auth.json")

    def _pin_gateway_host(self, base_url: str) -> None:
        """Pin the gateway hostname in the pod's /etc/hosts.

        Codex is a static musl binary; musl's resolver queries every
        nameserver in resolv.conf in parallel and takes the first answer —
        including an NXDOMAIN. Some node pools (hostNetwork pods in particular)
        list a public nameserver that NXDOMAINs private names, so under
        concurrency codex randomly loses the race and reports "stream
        disconnected before completion" without ever opening a connection.
        An /etc/hosts entry wins over DNS for musl and glibc alike.

        Resolution runs inside the pod via getent (glibc there queries
        nameservers sequentially, so it is immune to the race). /etc/hosts is
        a bind mount — in-place edits (sed -i) fail with EBUSY, so append.
        Best-effort: an unresolvable host just leaves DNS behavior unchanged.
        """
        host = urllib.parse.urlsplit(base_url).hostname
        if not host or re.fullmatch(r"[\d.]+|\[?[0-9a-fA-F:]+\]?", host):
            return  # already an IP literal — nothing to pin
        self.env.execute(
            f'grep -q " {host}$" /etc/hosts 2>/dev/null && exit 0; '
            f"ip=$(getent ahostsv4 {shlex.quote(host)} | awk '{{print $1; exit}}'); "
            f'[ -n "$ip" ] && echo "$ip {host}" >> /etc/hosts || true'
        )

    def _build_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "CODEX_HOME": _POD_LOGS_DIR,
            "OPENAI_API_KEY": self._model_kwargs.get("api_key") or "",
        }
        if self.config.extra_env:
            env.update(self.config.extra_env)
        return env

    def _run_codex(self, task: str) -> int:
        # Codex reads the prompt from stdin when no positional PROMPT is given;
        # a file redirect avoids argv length / quoting limits on large problem
        # statements.
        self._copy_text_to_pod(task, _POD_INSTRUCTION_FILE)

        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self._build_env().items())

        parts = [
            "codex",
            "exec",
        ]
        if self._turns > 0:
            # Resume the session recorded under $CODEX_HOME so the follow-up
            # query continues the same conversation, not a fresh one.
            parts.append("resume")
            parts.append("--last")
        parts += [
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            f"--model {shlex.quote(self._resolved_model)}",
            "--json",
            # Without model_reasoning_summary=auto, reasoning tokens are
            # consumed but never surface in the event stream.
            "-c model_reasoning_summary=auto",
        ]
        if self.config.unified_exec:
            parts.append("--enable unified_exec")
        if self.config.effort:
            parts.append(f"-c model_reasoning_effort={self.config.effort}")
        if self._model_kwargs.get("base_url"):
            parts.append("-c model_provider=openai_compat")

        # bin/ holds codex + its code-mode host sibling; codex-path/ holds the
        # bundled rg the upstream installer also exposes to the agent shell.
        cmd = (
            f'export PATH={_CODEX_BIN_DIR}:{_CODEX_PATH_DIR}:"$PATH"; '
            f"{env_prefix} {' '.join(parts)} "
            f"< {_POD_INSTRUCTION_FILE} > {_POD_OUTPUT_FILE} 2>&1"
        )
        self.logger.info(f"[codex] running (turn {self._turns + 1}) in {self._cwd}")
        rc, _ = run_detached(
            self.env,
            cmd,
            cwd=self._cwd,
            timeout=self.config.run_timeout,
            tag="codex",
            log_hint="event log: agent_msgs/codex.txt",
            logger=self.logger,
            idle_files=[_POD_OUTPUT_FILE],
            idle_timeout=self.config.stall_timeout,
        )
        return rc

    # ----- event collection & parsing ----------------------------------------

    def _collect_events(self) -> list[dict[str, Any]]:
        """Copy codex.txt out of the pod (next to the msg file, for inspection)
        and parse its NDJSON events. Falls back to catting a tail of the file
        through the exec transport when ``copy_out`` is unavailable/fails."""
        raw = ""
        if self.msg_path and hasattr(self.env, "copy_out"):
            dest = self.msg_path.parent / "codex.txt"
            try:
                self.env.copy_out(_POD_OUTPUT_FILE, str(dest))
                self.logger.info(f"[codex] copied event log to {dest}")
                raw = dest.read_text(errors="replace")
            except Exception as e:
                self.logger.warning(f"[codex] could not copy codex.txt: {e}")
        if not raw:
            try:
                res = self.env.execute(f"tail -c 2000000 {_POD_OUTPUT_FILE} 2>/dev/null || true")
                raw = res.get("output", "") or ""
            except Exception as e:
                self.logger.warning(f"[codex] could not read codex.txt: {e}")
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
        """Derive (result_text, status) from the event stream + exit code."""
        status = self.IDLE_STATUS if rc == 0 else "CodexError"

        result_text = ""
        n_messages = 0
        error_text = ""
        for ev in events:
            etype = ev.get("type")
            if etype == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") == "agent_message":
                    n_messages += 1
                    text = (item.get("text") or "").strip()
                    if text:
                        result_text = text
            elif etype in ("error", "turn.failed"):
                error_text = json.dumps(ev, ensure_ascii=False)

        if error_text and not result_text:
            status = "CodexError"
            result_text = error_text
        self._fold_usage(events, n_messages)

        if not result_text:
            result_text = f"(no agent message; codex rc={rc}, {len(events)} events)"
        return result_text, status

    def _fold_usage(self, events: list[dict[str, Any]], n_messages: int) -> None:
        """Map the last ``turn.completed`` usage block onto ``model.token_stats``
        + ``n_calls`` so ``save_traj`` reports real numbers instead of zeros."""
        usage: dict[str, Any] = {}
        for ev in reversed(events):
            if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
                break
        stats = getattr(self.model, "token_stats", None)
        if stats is not None and usage:
            stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
            stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
            stats.cache_read_tokens += int(usage.get("cached_input_tokens", 0) or 0)
        if usage:
            self.extra_template_vars["codex_usage"] = usage
        if n_messages:
            # Each agent_message ≈ one model call; advance the model's counter
            # so the trajectory's api_calls reflects the Codex session.
            try:
                self.model.n_calls += n_messages
            except Exception:
                pass
