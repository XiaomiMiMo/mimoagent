"""DeepSeek Harness (dsh) as a MiMo Agent blackbox harness.

Node app assembled from ~60 plugin packages (payload ships the installed tree
plus a pinned node 22; ``@vscode/ripgrep-linux-x64`` carries its own ``rg``).
A run boots a *profile* — an ordered stack of plugin patch layers — and
``--patch <file>`` is the seam this adapter writes into: one YAML patch list
that repoints the model route, kills the off-trajectory plugin rows and pins
the session log format. Verified on 0.1.0-rc.6:

* the LLM layer is ``@deepseek-ai/dsh-llm-pi-ai`` over ``@earendil-works/pi-ai``
  (the same SDK Pi/omp use), whose ``anthropic`` catalog route already knows
  ``claude-opus-5`` (``forceAdaptiveThinking``, ``thinkingLevelMap``). Overriding
  only ``baseURL``/``apiKeyEnv`` on that route keeps those catalog facts, so
  ``reasoning: xhigh|max`` reaches the wire as adaptive thinking +
  ``output_config.effort`` — a hand-declared route could not (dsh refuses
  ``compat.forceAdaptiveThinking``, and pi-ai gates adaptive on it);
* ``base_url`` must NOT carry ``/v1`` (pi-ai appends ``/v1/messages``);
* historic thinking blocks replay with signatures across turns, and
  cache_control lands on 3 breakpoints;
* ``--profile headless "task"`` answers one task, prints the final assistant
  message to stdout and exits 0 (1 on a terminal error);
* ``DSH_PERMISSION_MODE=danger-full-access`` is the yolo switch (sandbox
  danger-full-access + approval never).

Two payload build patches (see scripts/harness_payloads/): the launcher's HMR /
patch-file watchers are removed (they crash the run when the host's inotify
budget is gone, and nothing reloads mid-turn), and the headless runner learns
``$DSH_SESSION_ID`` + ``$DSH_RESUME_SESSION`` so a second turn resumes the
persisted session instead of starting a fresh one.

There is no event stream: stdout is the reply. Per-step usage is read from the
session log (``assistant/chunk`` usage chunks), which the patch list pins to
uncompressed JSONL — upstream defaults to zstd, which the host would need an
extra dependency to read.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

# pi-ai's built-in anthropic catalog route: the key must stay "anthropic" or
# the claude-opus-5 catalog entry (and its adaptive-thinking compat) is lost.
_ROUTE = "anthropic"
_KEY_ENV = "DSH_GW_API_KEY"

# Plugin rows switched off: just the per-session title LLM call, an extra request
# that has nothing to do with the task. Everything else — subagents, goals, plan
# mode, ralph/workflow loops, skills, web — stays. ``user-questions`` used to be
# listed here for having no user in a pod, but the model-facing ``tool-ask-user``
# row is not part of the headless profile in the first place: re-enabling the
# service leaves the same 25 tools on the wire and no ask tool among them
# (measured against a local mock), so that row was dead config.
_DISABLED_ROWS = ("session-title-llm",)


@dataclass
class DshAgentConfig(HarnessAgentConfig):
    version: str = "0.1.0-rc.6"
    # modelOverrides on the catalog route: context window + request max_tokens.
    context_limit: int = 1000000
    output_limit: int = 65536
    # Route-level thinking level: off | minimal | low | medium | high | xhigh |
    # max. None leaves pi-ai's default (no reasoning requested).
    effort: str | None = None
    # Extra patch-list rows merged into the staged overlay, e.g.
    # [{"id": "tool-fs-search", "config": {...}}]. A row's ``config`` REPLACES
    # the upstream one (the patch layer does not deep-merge), so restate every
    # key the row needs.
    extra_rows: list[dict[str, Any]] = field(default_factory=list)


class DshAgent(HarnessAgent):
    """Runs the DeepSeek Harness inside the pod under the MiMo Agent protocol."""

    NAME = "dsh"
    ERROR_STATUS = "DshError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-dsh-logs/dsh.txt"]
    # stdout is exactly the final reply; keep boot diagnostics out of it.
    SPLIT_STDERR = True

    def __init__(self, model, env, *, config_class=DshAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        # Stable for this agent (turn 2 resumes it), unique across agents: dsh
        # refuses to resume a session persisted under a different cwd, so a
        # fixed id breaks the moment one pod serves a second task.
        self._session_id = f"session-mimo-{uuid.uuid4().hex[:16]}"

    @property
    def _home_dir(self) -> str:
        return f"{self._logs_dir}/home"

    @property
    def _sessions_dir(self) -> str:
        # session logs are plain files, but keep them pod-local anyway
        return f"{self._logs_dir}/sessions"

    @property
    def _overlay_file(self) -> str:
        return f"{self._logs_dir}/overlay.yml"

    def _overlay(self) -> str:
        """The ``--patch`` layer. YAML is a JSON superset, so the patch list is
        emitted as JSON."""
        route: dict[str, Any] = {
            "apiKeyEnv": _KEY_ENV,
            # pi-ai appends /v1/messages itself
            "baseURL": self._gateway_base_url(v1=False),
            "modelOverrides": {
                self._resolved_model: {
                    "contextWindow": self.config.context_limit,
                    "maxTokens": self.config.output_limit,
                }
            },
        }
        if self.config.effort:
            route["reasoning"] = self.config.effort
        rows: list[dict[str, Any]] = [
            {"id": "llm-pi-ai", "config": {"providers": {_ROUTE: route}}},
            {
                "id": "agent-default-model",
                "config": {"provider": _ROUTE, "model": self._resolved_model},
            },
            # usage is read from this log; zstd would need a host-side decoder
            {
                "id": "session-persistence-jsonl",
                "config": {"root": self._sessions_dir, "compression": "none"},
            },
            # nothing reloads mid-run; the remaining chokidar watchers would
            # only burn the pod's inotify budget
            {"id": "credentials", "config": {"watch": False}},
            {"id": "settings", "config": {"watch": False}},
        ]
        rows += [{"id": row, "disabled": True} for row in _DISABLED_ROWS]
        rows += self.config.extra_rows
        return json.dumps(rows, ensure_ascii=False)

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._home_dir} {self._sessions_dir}")
        self._copy_text_to_pod(self._overlay(), self._overlay_file)

    def _harness_env(self) -> dict[str, str]:
        env = {
            "DSH_HOME": self._home_dir,
            _KEY_ENV: self._api_key,
            # sandbox danger-full-access + approval never
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_TELEMETRY_DISABLED": "1",
            "DSH_TELEMETRY_MODE": "DISABLED",
            "DSH_SESSION_ID": self._session_id,
        }
        if self._turns > 0:
            # payload build patch: resume the persisted session instead of
            # minting a fresh one (upstream headless is one-shot only).
            env["DSH_RESUME_SESSION"] = "1"
        return env

    def _command_parts(self, task: str) -> list[str]:
        return [
            "dsh",
            "--profile headless",
            f"--patch {self._overlay_file}",
            # the task is a positional argument (headless reads no stdin)
            f'"$(cat {self._instruction_file})"',
        ]

    def _parse_events(self, raw: str) -> list[dict[str, Any]]:
        # No event stream — keep the raw text for _parse_result.
        return [{"type": "raw", "text": raw}]

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        raw = (events[0].get("text") or "") if events else ""
        stderr_text = self._read_pod_text(f"{self._logs_dir}/{self.NAME}.stderr.txt")
        # SPLIT_STDERR appends stderr after stdout in the event log; peel it off
        # so the reply is exactly the stdout half.
        stdout_text = raw
        if stderr_text and raw.endswith(stderr_text):
            stdout_text = raw[: -len(stderr_text)]

        self._fold_usage(self._session_usage())

        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = stdout_text.strip()
        if not result_text:
            # headless always prints the final assistant message
            status = self.ERROR_STATUS
            tail = (stderr_text or raw).strip()[-1500:]
            result_text = f"(no agent message; dsh rc={rc}) {tail}"
        return result_text, status

    def _session_usage(self) -> list[dict[str, int]]:
        """One entry per model step, from the session log's usage chunks. The
        log is cumulative across turns of the resumed session, so only the
        chunks added since the previous turn are folded."""
        raw = self._read_pod_text(
            f'"$(ls -t {self._sessions_dir}/*/{self._session_id}/session.jsonl 2>/dev/null | head -1)"'
        )
        per_call = []
        for entry in self._parse_ndjson(raw):
            if entry.get("type") != "assistant/chunk":
                continue
            chunk = (entry.get("data") or {}).get("chunk") or {}
            if chunk.get("type") != "usage":
                continue
            usage = chunk.get("usage") or {}
            per_call.append(
                {
                    "input": usage.get("inputTokens", 0),
                    "output": usage.get("outputTokens", 0),
                    "cache_read": usage.get("cacheReadTokens", 0),
                    "cache_write": usage.get("cacheWriteTokens", 0),
                }
            )
        seen = self.extra_template_vars.setdefault("_dsh_usage_seen", 0)
        new = per_call[seen:]
        self.extra_template_vars["_dsh_usage_seen"] = len(per_call)
        return new
