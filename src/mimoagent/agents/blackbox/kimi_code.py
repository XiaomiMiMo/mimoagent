"""Moonshot Kimi Code CLI as a MiMo Agent blackbox harness.

Node-based CLI (payload ships a pinned node runtime + the self-contained
``dist/main.mjs`` bundle + a static ``rg``). The Anthropic provider wraps the
official ``@anthropic-ai/sdk`` — ``base_url`` must NOT carry ``/v1`` (the SDK
appends ``/v1/messages`` itself). Verified on 0.36.1:

* ``--prompt`` headless mode is inherently fully-auto (approval handler is
  hardcoded to approve) — no yolo flag needed or allowed;
* ``claude-opus-5`` gets adaptive thinking + ``output_config.effort`` with
  the full low..xhigh/max ladder via the ``[thinking]`` config section;
* historic thinking blocks replay with signatures; cache_control lands on
  system + last tool + last message block;
* default ``thinking.keep="all"`` switches to the beta messages API
  (``?beta=true`` + context-management) — ``KIMI_MODEL_THINKING_KEEP=off``
  keeps the wire a plain ``/v1/messages`` for gateways without that beta;
* the request's tool list comes from the compiled-in default agent profile,
  and a staged agent file named ``agent`` with ``override: true`` replaces
  that profile (24 tools on the wire) — see ``_agent_md``. Patching the profile
  literal inside ``dist/main.mjs`` (what 0.28.1 needed) no longer has any
  effect on the wire.

The prompt must travel via argv (no stdin/file path exists in 0.36.1), so the
command substitutes the instruction file: ``--prompt "$(cat ...)"``.

stream-json events carry no usage; per-call usage is folded from the
session's ``agents/main/wire.jsonl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gw"
_MODEL_ALIAS = "main"
# Tool allowlist for the default agent profile (see ``_agent_md``): upstream's
# full 27-entry default. Nothing is dropped, including the tools that cannot
# succeed in a task pod (WebSearch / FetchURL / Cron*) and AskUserQuestion —
# under ``--prompt`` the latter returns instantly (measured 0.059s) with "disabled
# while auto permission mode is active, make a reasonable decision and continue",
# so it never stalls the loop.
_DEFAULT_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "Bash",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "ReadMediaFile",
    "TodoList",
    "Skill",
    "Agent",
    "AgentSwarm",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "CreateGoal",
    "GetGoal",
    "SetGoalBudget",
    "UpdateGoal",
    "WebSearch",
    "FetchURL",
    "CronCreate",
    "CronList",
    "CronDelete",
    "mcp__*",
)


@dataclass
class KimiCodeAgentConfig(HarnessAgentConfig):
    version: str = "0.36.1"
    # → model max_context_size / max_output_size (max_output_size becomes the
    # request max_tokens; unknown model families default to 128000 otherwise).
    context_limit: int = 1000000
    output_limit: int = 65536
    # [thinking] section: enabled + effort. claude-opus-5 resolves to adaptive
    # thinking with efforts low|medium|high|xhigh|max (default high).
    thinking: bool = True
    effort: str | None = None
    # Preserved-thinking mode. "off" keeps requests on the plain
    # /v1/messages path; anything else re-enables kimi's context-management
    # beta (?beta=true) — only for gateways that support it.
    thinking_keep: str = "off"
    # Tool allowlist written into the default profile override (see
    # ``_agent_md``). None leaves upstream's compiled-in profile in place.
    tools: tuple[str, ...] | None = _DEFAULT_TOOLS


class KimiCodeAgent(HarnessAgent):
    """Runs Kimi Code inside the pod under the MiMo Agent protocol."""

    NAME = "kimi-code"
    ERROR_STATUS = "KimiCodeError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-kimi-code-logs/kimi-code.txt"]

    def __init__(self, model, env, *, config_class=KimiCodeAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    @property
    def _home_dir(self) -> str:
        return f"{self._logs_dir}/home"

    def _config_toml(self) -> str:
        lines = [
            f'default_model = "{_MODEL_ALIAS}"',
            "",
            f'[providers."{_PROVIDER_ID}"]',
            'type = "anthropic"',
            # official SDK appends /v1/messages itself
            f'base_url = "{self._gateway_base_url(v1=False)}"',
            f'api_key = "{self._api_key}"',
            "",
            f'[models."{_MODEL_ALIAS}"]',
            f'provider = "{_PROVIDER_ID}"',
            f'model = "{self._resolved_model}"',
            f"max_context_size = {self.config.context_limit}",
            f"max_output_size = {self.config.output_limit}",
            "",
            "[thinking]",
            f"enabled = {'true' if self.config.thinking else 'false'}",
        ]
        if self.config.effort:
            lines.append(f'effort = "{self.config.effort}"')
        lines.append("")
        return "\n".join(lines)

    def _agent_md(self) -> str:
        """Override of the builtin default profile (name ``agent``).

        0.36.1 resolves the request's tool list from the compiled-in profile
        catalog, not from a config key, but a discovered agent file named
        ``agent`` with ``override: true`` replaces that default for every
        session — including ``--continue`` resumes, which re-apply the file
        layer from the session snapshot. ``${base_prompt}`` keeps upstream's
        own system prompt; omitting ``subagents`` keeps the builtin roster so
        the ``Agent`` tool has something to delegate to.
        """
        tools = "\n".join(f"  - {t}" for t in self.config.tools or ())
        return (
            "---\n"
            "name: agent\n"
            "description: MiMo rollout coding agent\n"
            "override: true\n" + (f"tools:\n{tools}\n" if tools else "") + "---\n"
            "${base_prompt}\n"
        )

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._home_dir}/agents")
        self._copy_text_to_pod(self._config_toml(), f"{self._home_dir}/config.toml")
        self._copy_text_to_pod(self._agent_md(), f"{self._home_dir}/agents/agent.md")
        # Suppress the kimi-cli → kimi-code migration prompt.
        self.env.execute(f"touch {self._home_dir}/.skip-migration-from-kimi-cli")
        # Upstream treats effort "max" as session-only and a one-time
        # migration rewrites a persisted max to "high"; a pre-existing marker
        # makes it respect the hand-written value (verified on the wire).
        self._copy_text_to_pod(
            '{"thinking-effort-max-to-high": "2026-01-01T00:00:00.000Z"}\n',
            f"{self._home_dir}/migrations-effort.json",
        )

    def _harness_env(self) -> dict[str, str]:
        return {
            "KIMI_CODE_HOME": self._home_dir,
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
            "KIMI_CODE_MODEL_CATALOG_REFRESH_ON_START": "0",
            "KIMI_CODE_MODEL_CATALOG_REFRESH_INTERVAL_MS": "0",
            "KIMI_MODEL_THINKING_KEEP": self.config.thinking_keep,
            # kimi-code watches its agents dir with chokidar and a failed watch
            # escapes as an unhandled rejection that kills the whole turn
            # (rc=1). Hosts with an exhausted inotify budget hit this — polling
            # two small directories costs nothing and cannot ENOSPC.
            "CHOKIDAR_USEPOLLING": "1",
            "CHOKIDAR_INTERVAL": "1000",
        }

    def _command_parts(self, task: str) -> list[str]:
        parts = ["kimi"]
        if self._turns > 0:
            parts.append("--continue")
        parts += [
            # 0.36.1 reads the prompt from argv only — substitute the staged
            # instruction file (headless mode auto-approves everything).
            f'--prompt "$(cat {self._instruction_file})"',
            "--output-format stream-json",
        ]
        return parts

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        """stream-json lines: ``{"role":"assistant","content":...}`` (also
        tool_calls variants), ``{"role":"tool",...}``, ``{"role":"meta",...}``."""
        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = ""
        session_id = ""
        for ev in events:
            if ev.get("role") == "assistant":
                text = self._content_text(ev.get("content"))
                if text:
                    result_text = text
            elif ev.get("role") == "meta" and ev.get("type") == "session.resume_hint":
                session_id = ev.get("session_id") or ""
        self._fold_usage(self._wire_usage(session_id))
        if not result_text:
            # no assistant content at rc==0 = broken/empty stream, not success
            status = self.ERROR_STATUS
            result_text = f"(no agent message; kimi rc={rc}, {len(events)} events)"
        return result_text, status

    def _wire_usage(self, session_id: str) -> list[dict[str, int]]:
        """Per-call usage lives in the session's wire.jsonl, not in stdout."""
        if not session_id:
            return []
        raw = self._read_pod_text(f"{self._home_dir}/sessions/wd_*/{session_id}/agents/main/wire.jsonl")
        per_call = []
        seen = self.extra_template_vars.setdefault("_kimi_code_usage_seen", 0)
        for entry in self._parse_ndjson(raw):
            usage = entry.get("usage")
            if isinstance(usage, dict):
                per_call.append(
                    {
                        "input": usage.get("inputOther", 0),
                        "output": usage.get("output", 0),
                        "cache_read": usage.get("inputCacheRead", 0),
                        "cache_write": usage.get("inputCacheCreation", 0),
                    }
                )
        # wire.jsonl is cumulative across turns of the same session; only
        # fold the calls added since the previous turn.
        new = per_call[seen:]
        self.extra_template_vars["_kimi_code_usage_seen"] = len(per_call)
        return new
