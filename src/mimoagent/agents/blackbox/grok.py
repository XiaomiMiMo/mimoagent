"""xAI Grok Build CLI as a MiMo Agent blackbox harness.

Single static binary (no libc requirements at all — the payload is just
``bin/grok``). ``$GROK_HOME/config.toml`` defines a custom model with
``api_backend = "messages"``: requests go to ``{base_url}/messages`` (plain
concatenation, so the config carries the ``/v1`` suffix). Verified on
1.0.4:

* ``--reasoning-effort xhigh`` sends adaptive thinking +
  ``output_config.effort: xhigh``; unset sends no thinking field. **1.0.4
  cannot reach ``max``**: the CLI rejects the value ("unknown effort level
  'max'") and xhigh now reaches the wire verbatim, where 0.2.106 still
  rewrote xhigh into ``max`` (both captured on the real channel). ``effort:
  max`` therefore folds to xhigh with a warning — this arm's ceiling is one
  notch below the others';
* historic thinking blocks replay with signatures (resume and tool loops);
* cache_control lands on system[0] only (tools+system prefix caching);
* every auxiliary model role (session title, web search, ...) fires its own
  request and cannot be switched off; they are pointed at a dead-port
  ``title-blackhole`` model entry so nothing off-trajectory reaches the
  gateway (unpinned, the title request goes out as ``model:"grok-4.5"``).
  The 1.0.4 per-turn dashboard summary is NOT one of those roles — it uses
  the main model and is killed with ``GROK_TURN_SUMMARY=0``;
* headless output is ``streaming-json`` NDJSON: ``text``/``thought`` deltas
  and a final ``end`` event carrying usage + sessionId (used for ``-r``
  resume on follow-up turns). Tool calls never appear in this stream — the
  full trajectory lives in ``$GROK_HOME/sessions``.

The prompt travels via ``--prompt-file`` (headless mode ignores stdin).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig


@dataclass
class GrokAgentConfig(HarnessAgentConfig):
    version: str = "1.0.4"
    # Auto-compaction baseline / request max_tokens. Grok defaults max_tokens
    # to 128000 when unset, which exceeds most Claude output ceilings.
    context_limit: int = 1000000
    output_limit: int = 65536
    # Reasoning effort: low | medium | high | xhigh. Grok's own enum tops out
    # at xhigh and 1.0.4 puts it on the wire verbatim; "max" folds to xhigh
    # (with a warning) so an all-arms max sweep still runs. None sends no
    # thinking config at all.
    effort: str | None = None
    max_turns: int | None = None
    # Gateway connect failures retry with backoff; upstream default 8 hangs
    # for minutes, so pin it low and let the batch layer own retries.
    max_retries: int = 2
    # Nothing is removed. grok resolves ask_user_question itself in headless
    # mode — measured 0.004s to a non-error result telling the model "no user is
    # available, continue with your best judgment" — so it costs nothing and
    # keeps the arm's tool surface honest. Tools that simply cannot succeed
    # inside a task pod (scheduler_*/monitor, image/video generation, web) are
    # left on the wire too: a failing tool call is acceptable, a missing
    # capability is not. Set this to disallow names for a one-off run.
    disallowed_tools: str | None = None
    # Native skill roots to hide (config ``[skills] ignore``). Empty by
    # default: skills are task capability. Set it for a dev-machine run where
    # the host's own ~/.agents/skills would otherwise reach the prompt.
    skills_ignore: tuple[str, ...] = ()


class GrokAgent(HarnessAgent):
    """Runs Grok Build inside the pod under the MiMo Agent protocol."""

    NAME = "grok"
    ERROR_STATUS = "GrokError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-grok-logs/grok.txt"]

    def __init__(self, model, env, *, config_class=GrokAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._session_id: str | None = None
        if self.config.effort == "max":
            # A sweep that runs every arm at max must not die here, so fold —
            # but say so, because unlike the other arms this one really does
            # send one level lower.
            self.logger.warning(
                "[grok] 1.0.4 has no 'max' effort (the CLI rejects it) and sends "
                "xhigh verbatim, so this arm runs one level below the others"
            )

    @property
    def _effort(self) -> str | None:
        return "xhigh" if self.config.effort == "max" else self.config.effort

    @property
    def _home_dir(self) -> str:
        # sessions are SQLite — must live on pod-local disk, never a network
        # mount; /tmp inside task pods is local.
        return f"{self._logs_dir}/home"

    def _config_toml(self) -> str:
        model = self._resolved_model
        api_key = self._api_key
        lines = [
            "[features]",
            "telemetry = false",
            "feedback = false",
            # keeps startup from prefetching the model catalog at grok.com
            "remote_fetch = false",
            "",
            # subagents/workflows are part of how the model does the task
            "[subagents]",
            "enabled = true",
            "",
            "[workflows]",
            "enabled = true",
            "",
            "[models]",
            f'default = "{model}"',
            # Auxiliary roles (session title etc.) cannot be disabled, and
            # unpinned they ship model:"grok-4.5" to the gateway. Point them
            # at a blackhole entry on a dead local port instead: the title
            # request fails instantly client-side, never reaching the
            # gateway, and the main flow is unaffected (verified).
            'session_summary = "title-blackhole"',
            'web_search = "title-blackhole"',
            'image_description = "title-blackhole"',
            'prompt_suggestion = "title-blackhole"',
        ]
        if self._effort:
            lines.append(f'default_reasoning_effort = "{self._effort}"')
        lines += [
            "",
            f"[model.{model}]",
            f'model = "{model}"',
            # request URL = {base_url}/messages (plain concatenation)
            f'base_url = "{self._gateway_base_url(v1=True)}"',
            'api_backend = "messages"',
            # sent as Authorization: Bearer; the x-api-key extra header below
            # covers gateways that only read the Anthropic-style header.
            f'api_key = "{api_key}"',
            f"context_window = {self.config.context_limit}",
            f"max_completion_tokens = {self.config.output_limit}",
            f"max_retries = {self.config.max_retries}",
            "supports_reasoning_effort = true",
            f'extra_headers = {{ "x-api-key" = "{api_key}", "anthropic-version" = "2023-06-01" }}',
            "",
            "[model.title-blackhole]",
            'model = "title-blackhole"',
            'base_url = "http://127.0.0.1:1/v1"',
            'api_backend = "messages"',
            'api_key = "none"',
            "context_window = 32768",
            "max_completion_tokens = 512",
            "max_retries = 0",
            "",
        ]
        # 1.0.4 scans the Claude/Cursor/Codex vendor trees (skills, AGENTS/rules,
        # agent definitions, MCP, hooks) of both $HOME and the repo and injects
        # what it finds into the system prompt. Skills/rules/agents are task
        # context the model is meant to use, so they stay on; MCP needs servers
        # that do not exist in a task pod, hooks need user-authored scripts, and
        # importing another vendor's session history is unrelated to the task.
        for vendor in ("claude", "cursor", "codex"):
            lines += [
                f"[compat.{vendor}]",
                "skills = true",
                "rules = true",
                "agents = true",
                "mcps = false",
                "hooks = false",
                "sessions = false",
                "",
            ]
        # grok's native skill roots stay enabled (skills are task capability).
        # ``skills_ignore`` is left as an escape hatch for a local dev-machine
        # run, where a stray ~/.agents/skills would otherwise reach the prompt;
        # inside a task pod those paths do not exist.
        if self.config.skills_ignore:
            lines += [
                "[skills]",
                f"ignore = {json.dumps(list(self.config.skills_ignore))}",
                "",
            ]
        return "\n".join(lines)

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._home_dir}")
        self._copy_text_to_pod(self._config_toml(), f"{self._home_dir}/config.toml")

    def _harness_env(self) -> dict[str, str]:
        return {
            "GROK_HOME": self._home_dir,
            "GROK_DISABLE_AUTOUPDATER": "1",
            "GROK_AUTO_UPDATE": "0",
            "GROK_TELEMETRY_ENABLED": "0",
            "GROK_TELEMETRY_TRACE_UPLOAD": "0",
            "GROK_FEEDBACK_ENABLED": "0",
            # 1.0.4 asks the *main* model for a one-line dashboard summary
            # after every turn (the [models] summary roles do not cover it),
            # which doubles the requests per turn. Off — it is not part of
            # doing the task.
            "GROK_TURN_SUMMARY": "0",
            # Subagents/workflows are task machinery, so they stay on.
            "GROK_SUBAGENTS": "1",
            "GROK_WORKFLOWS": "1",
            # Memory is off for a wire reason, not a capability one: with
            # memory_get/memory_search on the wire, subscription-style gateway
            # channels 400 the whole request as a third-party app (verified
            #  — dropping exactly these two tools makes 24/26 pass).
            "GROK_MEMORY": "0",
        }

    def _command_parts(self, task: str) -> list[str]:
        import shlex

        parts = [
            "grok",
            f"--prompt-file {self._instruction_file}",
            f"-m {shlex.quote(self._resolved_model)}",
            "--yolo",
            "--output-format streaming-json",
            "--no-auto-update",
        ]
        if self.config.disallowed_tools:
            parts.append(f"--disallowed-tools {shlex.quote(self.config.disallowed_tools)}")
        if self._effort:
            parts.append(f"--reasoning-effort {shlex.quote(self._effort)}")
        if self.config.max_turns is not None:
            parts.append(f"--max-turns {self.config.max_turns}")
        if self._turns > 0 and self._session_id:
            parts.append(f"-r {shlex.quote(self._session_id)}")
        return parts

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        # a clean run always terminates with an `end` event; rc==0 without
        # one means the stream died mid-run (or emitted nothing parseable)
        text_parts: list[str] = []
        error_text = ""
        end_event: dict[str, Any] = {}
        for ev in events:
            etype = ev.get("type")
            if etype == "text":
                text_parts.append(ev.get("data") or "")
            elif etype == "end":
                end_event = ev
            elif etype == "error":
                error_text = json.dumps(ev, ensure_ascii=False)

        if end_event.get("sessionId"):
            self._session_id = end_event["sessionId"]

        result_text = "".join(text_parts).strip()
        if error_text:
            # any error event fails the turn, even alongside partial text
            status = self.ERROR_STATUS
            result_text = result_text or error_text

        usage = end_event.get("usage") or {}
        if usage:
            # input_tokens excludes cache reads (they are listed separately);
            # grok folds cache writes into input_tokens, so cache_write stays 0.
            n_calls = sum(
                int((m or {}).get("modelCalls", 0) or 0) for m in (end_event.get("modelUsage") or {}).values()
            ) or int(end_event.get("num_turns", 1) or 1)
            call = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_write": 0,
            }
            # usage is per-run aggregate; report it as one call and bump the
            # counter separately so api_calls still reflects the session.
            self._fold_usage([call])
            try:
                self.model.n_calls += max(0, n_calls - 1)
            except Exception:
                pass

        if not any(ev.get("type") == "end" for ev in events):
            status = self.ERROR_STATUS
        if not result_text:
            status = self.ERROR_STATUS
            result_text = f"(no agent message; grok rc={rc}, {len(events)} events)"
        return result_text, status
