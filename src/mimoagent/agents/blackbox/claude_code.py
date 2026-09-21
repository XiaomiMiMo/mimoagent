"""Claude Code as a mimoagent blackbox agent.

Instead of driving the mimoagent ``step()`` loop, this agent runs Anthropic's
Claude Code scaffold *inside the test pod* via the Claude Agent SDK, then hands
control back. It satisfies the :class:`mimoagent.Agent` protocol
(``run(task) -> (status, message)``), so environment setup,
``DatasetEnvironment.calculate_reward()`` (which grades ``git diff HEAD`` in the
repo) and ``save_traj`` all work without modification.

Lifecycle of ``run(task)``:

1. **install** — upload + run ``install-claude-code.sh`` in the pod (idempotent;
   downloads the pinned Claude Code CLI from Anthropic's release bucket, a
   standalone Python 3.12 and ``claude-agent-sdk`` from PyPI). Skipped on
   subsequent ``run()`` calls (multi-turn).
2. **stage** — upload the SDK runner ``run_claude_sdk.py`` and write the task
   instruction to a file in the pod.
3. **run** — exec the runner with ``ANTHROPIC_*`` env derived from the shared
   ``model`` config; Claude edits the repo in ``cwd``.
4. **collect** — read back the SDK log + result, fold them into ``self.messages``
   and ``model.token_stats`` so the trajectory + token stats serialize normally.

Auth/model routing reuses the ``model:`` config block (``model_name``,
``model_kwargs.base_url`` / ``api_key``) — one config drives both the native and
the Claude Code agent.
"""

from __future__ import annotations

import json
import shlex
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
_INSTALL_SCRIPT = _RESOURCES / "install-claude-code.sh"
_SDK_RUNNER = _RESOURCES / "run_claude_sdk.py"

# Paths inside the pod.
_POD_INSTALL_SCRIPT = "/tmp/mimo-install-claude-code.sh"
_POD_SDK_RUNNER = "/tmp/mimo-run-claude-sdk.py"
_POD_LOGS_DIR = "/tmp/mimo-claude-logs"
_POD_INSTRUCTION_FILE = "/tmp/mimo-claude-instruction.txt"
_POD_APPEND_SYSTEM_PROMPT_FILE = "/tmp/mimo-claude-append-system-prompt.txt"
_POD_OPTIONS_FILE = "/tmp/mimo-claude-options.json"
# Standalone CPython the install script lays down; runs the SDK runner.
_PYTHON_BIN = "/opt/mimo-python/python/bin/python3.12"
# Pinned standalone Claude Code CLI the install script downloads.
# Passed to the SDK runner via --cli-path so we don't use the wheel-bundled CLI.
_POD_CLAUDE_BIN = "/opt/mimo-claude/claude"
# Default Claude Code CLI release (see install-claude-code.sh for the source).
DEFAULT_CLAUDE_VERSION = "2.1.269"


@dataclass
class ClaudeCodeAgentConfig(AgentConfig):
    """Config for the Claude Code blackbox agent.

    Templates are unused (Claude Code carries its own system prompt) but kept on
    the base so unknown-key filtering and the shared dataclass surface still work.
    """

    # Where Claude runs / how long. ``cwd`` defaults to the env's working dir.
    cwd: str | None = None
    # Claude Code CLI version to install (verified with ``claude -v``). The
    # literal "latest" resolves to the newest published release at install time.
    version: str = DEFAULT_CLAUDE_VERSION
    max_turns: int | None = None
    effort: str = "high"  # low | medium | high | max
    # Extra instructions appended on top of the built-in claude_code system
    # prompt preset (the preset is preserved, not replaced).
    append_system_prompt: str | None = None
    # Per-instance project CLAUDE.md content, written to ``<cwd>/CLAUDE.md``
    # before the run (batch.py wires it from the instance's ``claudemd`` field;
    # agents that don't declare this field never receive it).
    claudemd: str | None = None
    install_timeout: int = 1800
    run_timeout: int = 18000
    # Stall watchdog for the detached run: kill the session when the raw
    # session log (claude-code.txt) saw no write for this many seconds. 0
    # disables it. Keep it above CLAUDE_STREAM_IDLE_TIMEOUT_MS (15 min): a
    # model call that long is legitimate during training pauses.
    stall_timeout: int = 0
    # Extra ANTHROPIC_*/CLAUDE_* env overrides layered on top of the derived
    # set. Named ``extra_env`` (not ``env``) to avoid colliding with the
    # Environment positional arg threaded through make_agent / the constructor.
    extra_env: dict[str, str] | None = None
    # Extra Claude CLI flags forwarded through ClaudeAgentOptions.extra_args.
    # Keys omit the leading ``--``; values are strings or None for a bare flag.
    # Example: {"tools": "Bash,Read,Edit", "setting-sources": ""}.
    cli_args: dict[str, str | None] | None = None
    # None preserves the historical default (WebSearch disabled). An empty list
    # explicitly removes that default; non-empty lists replace it.
    disallowed_tools: list[str] | None = None
    # Skip the install step for images that already carry the CLI + SDK.
    skip_install: bool = False
    # Extra environment for the install script (mirror overrides such as
    # PIP_INDEX_URL / PBS_BASE / CLAUDE_RELEASES_BASE).
    install_env: dict[str, str] | None = None
    # Enable the Agent SDK's debug stream; the debug log is copied out next to
    # the msg file as claude-code-debug.log.
    debug: bool = False


class ClaudeCodeAgent(BaseAgent):
    """Runs Claude Code inside the pod. Implements the mimoagent Agent protocol."""

    IDLE_STATUS = "Completed"
    # Raw session log the SDK runner writes in the pod. Cumulative across
    # turns (opened in append mode when the session is continued).
    ENV_TRAJECTORY_FILES = [f"{_POD_LOGS_DIR}/claude-code.txt"]

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = ClaudeCodeAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._installed = False
        self._staged = False
        self._turns = 0

    # ----- Agent protocol ---------------------------------------------------

    def get_model_query_kwargs(self) -> dict:
        # No tool schema is fed to mimoagent's model layer — Claude Code owns its
        # own tools. Returning {} keeps save_traj's _agent_traj() shape valid.
        return {}

    def step(self) -> dict | None:  # pragma: no cover - not used by this agent
        """Unused: this agent overrides ``run`` rather than stepping a loop."""
        return None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Install (once), run Claude Code on ``task`` in the pod, collect result.

        Returns ``(status, message)``:
        * ``"Completed"`` — runner exited cleanly.
        * ``"InfraError"`` — pod transport failure (escalated like the native agent).
        * ``"ClaudeCodeError"`` — runner exited non-zero / reported an error.
        """
        self.extra_template_vars |= {"task": task, **kwargs}

        try:
            if not self.config.skip_install and not self._installed:
                self._install()
                self._installed = True
            if not self._staged:
                self._stage_runner()
                self._staged = True
            rc, stdout = self._run_claude(task)
        except TransportError as e:
            # Mirror DefaultAgent.execute_action: pod transport failures terminate.
            raise InfraError(str(e)) from e

        # Copy Claude Code's own raw log out of the pod next to this agent's
        # msg file, so the full session is inspectable afterwards.
        self._copy_log_out()
        result_text, status = self._collect(rc, stdout)
        # Keep self.messages minimal: the prompt + final reply. The full
        # conversation lives in the copied-out claude-code.txt.
        self.add_message("user", task)
        self.add_message("assistant", result_text)
        self._turns += 1
        return status, result_text

    # ----- internals --------------------------------------------------------

    @property
    def _cwd(self) -> str:
        if self.config.cwd:
            return self.config.cwd
        # Fall back to the environment's configured working dir.
        return getattr(self.env.config, "cwd", "/testbed")

    def _install(self) -> None:
        version = self.config.version or DEFAULT_CLAUDE_VERSION
        self.logger.info(f"[claude-code] installing scaffold ({version}) into pod...")
        self.env.copy_to(str(_INSTALL_SCRIPT), _POD_INSTALL_SCRIPT)
        path_prefix = stage_install_common(self.env) + installer_env_prefix(self.config.install_env)
        res = self.env.execute(
            f"{path_prefix}CLAUDE_VERSION={shlex.quote(version)} bash {_POD_INSTALL_SCRIPT}",
            timeout=self.config.install_timeout,
        )
        out = res.get("output", "") or ""
        if "CLAUDE_CODE_INSTALL_OK" not in out:
            raise RuntimeError(
                f"[claude-code] install failed (rc={res.get('returncode')}). Output tail:\n{out[-3000:]}"
            )
        self.logger.info("[claude-code] install OK")

    def _stage_runner(self) -> None:
        self.env.copy_to(str(_SDK_RUNNER), _POD_SDK_RUNNER)
        self.env.execute(f"mkdir -p {_POD_LOGS_DIR}")
        if self.config.claudemd:
            self._write_claudemd()

    def _write_claudemd(self) -> None:
        """Write the per-instance project CLAUDE.md into the workspace.

        Added to .git/info/exclude so the injected file never shows up in the
        graded ``git diff HEAD`` as if the agent had authored it."""
        self.logger.info(f"[claude-code] writing CLAUDE.md to {self._cwd}")
        self._copy_text_to_pod(self.config.claudemd, f"{self._cwd}/CLAUDE.md")
        self.env.execute(
            "grep -qxF CLAUDE.md .git/info/exclude 2>/dev/null || "
            "echo CLAUDE.md >> .git/info/exclude 2>/dev/null || true",
            cwd=self._cwd,
        )

    def _copy_text_to_pod(self, text: str, remote_path: str) -> None:
        """Write ``text`` to ``remote_path`` in the pod via a local tempfile +
        ``copy_to`` (tar stream). The base env exposes only ``copy_to``, and a
        heredoc/argv would choke on large or non-utf8 task statements — same
        rationale as batch.py's patch upload."""
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
        """Derive Claude Code env vars from the shared ``model`` config.

        ``model_name`` may carry a provider prefix (``anthropic/...``,
        ``openai/...``). When a custom base_url is set (a gateway) we keep the
        full name; otherwise we strip the prefix for the official API.
        """
        mcfg = self.model.config
        model_name = mcfg.model_name
        mkwargs = getattr(mcfg, "model_kwargs", {}) or {}
        base_url = mkwargs.get("base_url") or ""
        api_key = mkwargs.get("api_key") or ""

        env: dict[str, str] = {
            "IS_SANDBOX": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "API_TIMEOUT_MS": "1800000",
            # The CLI's default stream idle limit (300s in 2.1.220, 600s in 2.1.233)
            # is shorter than a model-serving pause during training; hitting it
            # aborts the in-flight request and the rollout produces nothing, so
            # raise it explicitly.
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "900000",
            "CLAUDE_CONFIG_DIR": f"{_POD_LOGS_DIR}/sessions",
        }
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
            resolved_model = model_name
        else:
            resolved_model = model_name.split("/")[-1]
        env["ANTHROPIC_MODEL"] = resolved_model
        # Pin every tier to the same model so subagents/haiku calls don't escape
        # to an unrouted default.
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = resolved_model
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = resolved_model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = resolved_model
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = resolved_model

        if self.config.debug:
            # The SDK runner turns this into --debug-to-stderr and writes the
            # stream to claude-code-debug.log in the pod logs dir.
            env["CLAUDE_CODE_DEBUG"] = "1"

        if self.config.extra_env:
            env.update(self.config.extra_env)
        return env

    def _run_claude(self, task: str) -> tuple[int, str]:
        # Write the instruction to a file (avoids argv length / quoting limits on
        # large problem statements) and point the runner at it. The runner reads
        # a JSON list, so the instructions file is a one-element JSON array.
        self._copy_text_to_pod(json.dumps([task]), _POD_INSTRUCTION_FILE)

        if self.config.cli_args is not None or self.config.disallowed_tools is not None:
            options = {
                "cli_args": self.config.cli_args,
                "disallowed_tools": self.config.disallowed_tools,
            }
            self._copy_text_to_pod(json.dumps(options), _POD_OPTIONS_FILE)

        env = self._build_env()
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

        cli_bin = _POD_CLAUDE_BIN
        parts = [
            _PYTHON_BIN,
            _POD_SDK_RUNNER,
            f"--logs-dir={_POD_LOGS_DIR}",
            f"--cwd={shlex.quote(self._cwd)}",
            f"--instructions-file={_POD_INSTRUCTION_FILE}",
            f"--model={shlex.quote(env['ANTHROPIC_MODEL'])}",
            f"--effort={self.config.effort}",
            f"--cli-path={shlex.quote(cli_bin)}",
        ]
        if self.config.cli_args is not None or self.config.disallowed_tools is not None:
            parts.append(f"--options-file={_POD_OPTIONS_FILE}")
        if self.config.max_turns is not None:
            parts.append(f"--max-turns={self.config.max_turns}")
        if self.config.append_system_prompt:
            # Pass via a pod file (not argv) so large/multiline prompts avoid
            # length + quoting limits — same rationale as the instruction file.
            self._copy_text_to_pod(self.config.append_system_prompt, _POD_APPEND_SYSTEM_PROMPT_FILE)
            parts.append(f"--append-system-prompt-file={shlex.quote(_POD_APPEND_SYSTEM_PROMPT_FILE)}")
        if self._turns > 0:
            parts.append("--continue")  # resume the same session for multi-turn

        cmd = f"{env_prefix} {' '.join(parts)} 2>&1"
        self.logger.info(f"[claude-code] running (turn {self._turns + 1}) in {self._cwd}")
        return run_detached(
            self.env,
            cmd,
            timeout=self.config.run_timeout,
            tag="claude-code",
            log_hint="session log: agent_msgs/claude-code.txt",
            logger=self.logger,
            idle_files=[f"{_POD_LOGS_DIR}/claude-code.txt"],
            idle_timeout=self.config.stall_timeout,
        )

    def _copy_log_out(self) -> None:
        """Copy Claude Code's raw session log (``claude-code.txt``) — plus the
        SDK debug log when ``debug`` is on — out of the pod next to this
        agent's msg file, so the full session is inspectable afterwards.
        Best-effort: never raises."""
        if not self.msg_path or not hasattr(self.env, "copy_out"):
            return
        names = ["claude-code.txt"]
        if self.config.debug:
            names.append("claude-code-debug.log")
        for name in names:
            dest = self.msg_path.parent / name
            try:
                self.env.copy_out(f"{_POD_LOGS_DIR}/{name}", str(dest))
                self.logger.info(f"[claude-code] copied {name} to {dest}")
            except Exception as e:
                self.logger.warning(f"[claude-code] could not copy {name}: {e}")

    def _collect(self, rc: int, stdout: str) -> tuple[str, str]:
        """Read the SDK result back from the pod, fold usage into token stats."""
        result_text = ""
        status = self.IDLE_STATUS if rc == 0 else "ClaudeCodeError"

        try:
            res = self.env.execute(f"cat {_POD_LOGS_DIR}/sdk_result.json 2>/dev/null || true")
            raw = (res.get("output", "") or "").strip()
            if raw:
                result_data = json.loads(raw)
                result_text = result_data.get("result") or ""
                if result_data.get("is_error"):
                    status = "ClaudeCodeError"
                self._fold_usage(result_data)
                # Stash the structured result for downstream inspection / save_traj.
                self.extra_template_vars["sdk_result"] = result_data
        except Exception as e:  # never let collection mask the run outcome
            self.logger.warning(f"[claude-code] could not parse sdk_result.json: {e}")

        if not result_text:
            # The runner no longer streams the session to stdout, so read the
            # tail out of the pod (same shape as codex.txt). ``stdout`` is only
            # stderr noise now, kept as the last resort for a dead/timed-out pod
            # where env.execute can no longer reach the file.
            try:
                res = self.env.execute(f"tail -c 5000 {_POD_LOGS_DIR}/claude-code.txt 2>/dev/null || true")
                result_text = (res.get("output", "") or "").strip()
            except Exception as e:
                self.logger.warning(f"[claude-code] could not tail claude-code.txt: {e}")
            if not result_text:
                result_text = stdout[-5000:]
        return result_text, status

    def _fold_usage(self, result_data: dict[str, Any]) -> None:
        """Map the SDK's usage block onto ``model.token_stats`` + ``n_calls`` so
        ``save_traj`` reports real numbers instead of zeros."""
        usage = result_data.get("usage") or {}
        if not isinstance(usage, dict):
            return
        stats = getattr(self.model, "token_stats", None)
        if stats is not None:
            stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
            stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
            stats.cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
            stats.cache_creation_tokens += int(usage.get("cache_creation_input_tokens", 0) or 0)
        num_turns = result_data.get("num_turns")
        if isinstance(num_turns, int):
            # n_calls is a plain int attribute on the model; advance it so the
            # trajectory's api_calls reflects the Claude Code session.
            try:
                self.model.n_calls += num_turns
            except Exception:
                pass
