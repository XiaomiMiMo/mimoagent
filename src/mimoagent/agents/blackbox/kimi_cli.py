"""Moonshot Kimi CLI as a MiMo Agent blackbox harness.

Python app (payload ships standalone CPython 3.12 + a venv built at
``/opt/mimo-kimi-cli`` + a static ``rg``). The ``anthropic`` provider type
wraps ``AsyncAnthropic`` — ``base_url`` must NOT carry ``/v1`` (the SDK
appends ``/v1/messages``). Verified on 1.49.0:

* ``--print`` headless mode reads the whole prompt from stdin and inherently
  auto-approves (afk); exit 75 flags retryable infra failures;
* ``--agent-file`` restates the stock agent profile so the tool list is pinned
  across versions (coding tools, the Agent tool + its subagent roster, plan
  mode, web, AskUserQuestion — afk auto-dismisses the last one instantly);
* thinking replays historic blocks with signatures; cache_control lands on
  system + last message block + last tool (not configurable);
* stock 1.49.0 only ever sends effort ``high`` and its model-name regex
  predates ``claude-opus-5`` (falls back to budget_tokens 32000). The payload
  build patches kosong to (a) treat opus-5/fable ids as adaptive-capable and
  (b) read the effort level from ``$KIMI_CLI_THINKING_EFFORT`` — which is how
  ``effort: xhigh|max`` here reaches the wire as ``output_config.effort``, and
  (c) read max_tokens from ``$KIMI_CLI_MAX_TOKENS`` (upstream pins the
  anthropic provider at 50000, so ``output_limit`` was inert).

stream-json output carries no usage; per-call usage is folded from the
session's ``wire.jsonl`` (``StatusUpdate.token_usage``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gw"
_MODEL_ALIAS = "main"


@dataclass
class KimiCliAgentConfig(HarnessAgentConfig):
    version: str = "1.49.0"
    # → model max_context_size. There is NO max-output knob in 1.49.0: the
    # anthropic path hardcodes max_tokens=50000.
    context_limit: int = 1000000
    # Request max_tokens. Upstream hardcodes 50000 for anthropic providers; the
    # payload patch reads this through $KIMI_CLI_MAX_TOKENS instead.
    output_limit: int = 65536
    thinking: bool = True
    # Effort level (high | xhigh | max ...). Reaches the wire through the
    # payload's KIMI_CLI_THINKING_EFFORT patch; ignored when thinking=false.
    effort: str | None = None
    # Turn cap per print run (upstream default 1000).
    max_steps_per_turn: int | None = None


class KimiCliAgent(HarnessAgent):
    """Runs Kimi CLI inside the pod under the MiMo Agent protocol."""

    NAME = "kimi-cli"
    ERROR_STATUS = "KimiCliError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-kimi-cli-logs/kimi-cli.txt"]

    def __init__(self, model, env, *, config_class=KimiCliAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

    @property
    def _share_dir(self) -> str:
        return f"{self._logs_dir}/share"

    @property
    def _config_file(self) -> str:
        return f"{self._logs_dir}/config.toml"

    @property
    def _agent_file(self) -> str:
        return f"{self._logs_dir}/agent.yaml"

    def _agent_yaml(self) -> str:
        """Custom agent spec (--agent-file): upstream's default profile, nothing
        dropped. ``--print`` sets afk (``ui == "print"``), and afk makes
        AskUserQuestion return immediately with ``{"answers": {}, "note":
        "Running in afk mode. ... Make your own decision."}`` — measured 0.026s,
        not an error — so it cannot stall the loop. SearchWeb/FetchURL stay even
        though they cannot reach anything from a pod. The ``Agent`` tool and its
        subagent roster stay too, so the spec re-declares the three builtin
        subagents by absolute path."""
        agents_dir = f"{self._install_dir}/venv/lib/python3.12/site-packages/kimi_cli/agents/default"
        system_md = f"{agents_dir}/system.md"
        tools = "\n".join(
            f'    - "{t}"'
            for t in (
                "kimi_cli.tools.shell:Shell",
                "kimi_cli.tools.background:TaskList",
                "kimi_cli.tools.background:TaskOutput",
                "kimi_cli.tools.background:TaskStop",
                "kimi_cli.tools.file:ReadFile",
                "kimi_cli.tools.file:ReadMediaFile",
                "kimi_cli.tools.file:Glob",
                "kimi_cli.tools.file:Grep",
                "kimi_cli.tools.file:WriteFile",
                "kimi_cli.tools.file:StrReplaceFile",
                "kimi_cli.tools.todo:SetTodoList",
                "kimi_cli.tools.ask_user:AskUserQuestion",
                "kimi_cli.tools.agent:Agent",
                "kimi_cli.tools.plan.enter:EnterPlanMode",
                "kimi_cli.tools.plan:ExitPlanMode",
                "kimi_cli.tools.web:SearchWeb",
                "kimi_cli.tools.web:FetchURL",
            )
        )
        subagents = "".join(
            f"    {name}:\n      path: {agents_dir}/{name}.yaml\n      description: {desc}\n"
            for name, desc in (
                ("coder", "Good at general software engineering tasks."),
                (
                    "explore",
                    "Fast codebase exploration with prompt-enforced read-only behavior.",
                ),
                (
                    "plan",
                    "Read-only implementation planning and architecture design.",
                ),
            )
        )
        return (
            "version: 1\n"
            "agent:\n"
            '  name: ""\n'
            f"  system_prompt_path: {system_md}\n"
            "  system_prompt_args:\n"
            '    ROLE_ADDITIONAL: ""\n'
            "  tools:\n"
            f"{tools}\n"
            "  subagents:\n"
            f"{subagents}"
        )

    def _config_toml(self) -> str:
        capabilities = '["thinking"]' if self.config.thinking else "[]"
        return "\n".join(
            [
                f'default_model = "{_MODEL_ALIAS}"',
                "default_yolo = true",
                f"default_thinking = {'true' if self.config.thinking else 'false'}",
                # Keep the injected conversation clean of the one-shot afk
                # system-reminder (print mode injects it otherwise).
                "skip_afk_prompt_injection = true",
                "telemetry = false",
                "",
                f"[providers.{_PROVIDER_ID}]",
                'type = "anthropic"',
                # AsyncAnthropic appends /v1/messages itself
                f'base_url = "{self._gateway_base_url(v1=False)}"',
                f'api_key = "{self._api_key}"',
                "",
                f"[models.{_MODEL_ALIAS}]",
                f'provider = "{_PROVIDER_ID}"',
                f'model = "{self._resolved_model}"',
                f"max_context_size = {self.config.context_limit}",
                f"capabilities = {capabilities}",
                "",
            ]
        )

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._share_dir}")
        self._copy_text_to_pod(self._config_toml(), self._config_file)
        self._copy_text_to_pod(self._agent_yaml(), self._agent_file)

    def _harness_env(self) -> dict[str, str]:
        env = {
            "KIMI_SHARE_DIR": self._share_dir,
            "KIMI_DISABLE_TELEMETRY": "1",
            "KIMI_CLI_NO_AUTO_UPDATE": "1",
            # Payload build patch: upstream constructs the anthropic provider
            # with a hardcoded default_max_tokens=50000.
            "KIMI_CLI_MAX_TOKENS": str(self.config.output_limit),
        }
        if self.config.thinking and self.config.effort:
            # Payload build patch: kimi_cli reads the thinking effort level
            # from this env var (upstream hardcodes "high").
            env["KIMI_CLI_THINKING_EFFORT"] = self.config.effort
        return env

    def _command_parts(self, task: str) -> list[str]:
        parts = ["kimi", "--print", "--output-format stream-json"]
        parts.append(f"--config-file {self._config_file}")
        parts.append(f"--agent-file {self._agent_file}")
        if self._turns > 0:
            parts.append("--continue")
        if self.config.max_steps_per_turn:
            parts.append(f"--max-steps-per-turn {self.config.max_steps_per_turn}")
        # prompt arrives on stdin (the shared instruction-file redirect)
        return parts

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        # exit 75 = EX_TEMPFAIL (connection/timeout/429/5xx) — still an error
        # status here; the batch layer decides about retries.
        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = ""
        for ev in events:
            if ev.get("role") == "assistant":
                text = self._content_text(ev.get("content"))
                if text:
                    result_text = text
        self._fold_usage(self._wire_usage())
        if not result_text:
            # no assistant content at rc==0 = broken/empty stream, not success
            status = self.ERROR_STATUS
            result_text = f"(no agent message; kimi-cli rc={rc}, {len(events)} events)"
        return result_text, status

    def _wire_usage(self) -> list[dict[str, int]]:
        """Per-step usage lives in the session's wire.jsonl StatusUpdate
        events (sessions/<md5(workdir)>/<id>/wire.jsonl under the share dir);
        the newest session for this cwd is the one this turn just used."""
        raw = self._read_pod_text(f'"$(ls -td {self._share_dir}/sessions/*/* 2>/dev/null | head -1)/wire.jsonl"')
        per_call = []
        seen = self.extra_template_vars.setdefault("_kimi_cli_usage_seen", 0)
        for entry in self._parse_ndjson(raw):
            message = entry.get("message") or {}
            if message.get("type") != "StatusUpdate":
                continue
            usage = (message.get("payload") or {}).get("token_usage") or {}
            if isinstance(usage, dict) and usage:
                per_call.append(
                    {
                        "input": usage.get("input_other", 0),
                        "output": usage.get("output", 0),
                        "cache_read": usage.get("input_cache_read", 0),
                        "cache_write": usage.get("input_cache_creation", 0),
                    }
                )
        new = per_call[seen:]
        self.extra_template_vars["_kimi_cli_usage_seen"] = len(per_call)
        return new
