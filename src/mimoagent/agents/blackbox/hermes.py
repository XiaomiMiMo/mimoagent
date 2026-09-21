"""NousResearch Hermes Agent as a MiMo Agent blackbox harness.

Python app (payload ships a standalone CPython 3.12 venv built at
``/opt/mimo-hermes`` plus the source tree it is installed editable from —
0.20.1 refuses wheel builds outside Nix — plus a static ``rg``). The ``chat -Q -q`` path is used —
NOT ``-z``/oneshot, which ignores ``agent.reasoning_effort`` and max_turns.
Verified on v2026.8.13 (0.20.1):

* a named provider with ``api_mode: anthropic_messages`` forces the official
  Anthropic SDK transport (``base_url`` without ``/v1``; x-api-key auth);
* ``agent.reasoning_effort: xhigh|max`` sends adaptive thinking +
  ``output_config.effort`` for claude-opus-5; unset sends no thinking field;
* upstream strips ALL historic thinking blocks when the base_url lacks an
  ``anthropic.com`` substring ("third-party" heuristic), and even on the
  native path it keeps them only on the *latest* assistant message — the
  payload is built with two patches adding the
  ``HERMES_FORCE_NATIVE_ANTHROPIC=1`` and ``HERMES_KEEP_ALL_THINKING=1``
  escape hatches (both set by the adapter) so every request carries the full
  signed thinking history (see harness_payloads/build_harness_payloads.sh); auth/beta/
  fast-mode call sites are unaffected. Without the second one the replayed
  prefix mutates every turn, which costs a full prompt-cache miss;
* prompt caching is automatic (system + last 3 messages, 4 breakpoints);
* two beta headers ship on every request (interleaved-thinking +
  fine-grained-tool-streaming) — the gateway must tolerate them;
* tools run through an exclusive toolset whitelist (``-t``): the default
  coding set stays under subscription-style gateway channels' third-party
  tool-combination fingerprint (see ``HermesAgentConfig.toolsets``). It
  includes ``clarify``, whose stall is bounded by ``clarify_timeout`` — hermes
  is the one harness here that does not resolve its ask tool structurally.

There is no JSON event stream: ``-Q`` prints the final reply to stdout and
``session_id: <id>`` to stderr (SPLIT_STDERR keeps them apart) — but only with
``display.show_reasoning: false`` and ``security.tirith_enabled: false``, or
the thinking box and a scanner warning land in stdout too. Usage comes
from the session row (+ descendant sub-/compaction sessions) in the
pod-local ``state.db``, read via the payload's own python.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from mimoagent.agents.blackbox.harness_base import HarnessAgent, HarnessAgentConfig

_PROVIDER_ID = "gw"
_SESSION_ID_RE = re.compile(r"^session_id:\s*([\w.-]{1,64})\s*$", re.MULTILINE)


@dataclass
class HermesAgentConfig(HarnessAgentConfig):
    version: str = "2026.8.13"
    # agent.reasoning_effort: minimal|low|medium|high|xhigh|max. None omits
    # the thinking/output_config fields entirely (server-side default).
    effort: str | None = None
    # Request max_tokens; None uses hermes's model table (128000 for opus-5).
    max_tokens: int | None = None
    # Tool-calling iterations per turn (upstream default 90).
    max_turns: int | None = None
    # Seconds ``clarify`` waits for an answer before telling the model to decide
    # for itself. Unlike every other harness here, hermes does NOT resolve the
    # ask tool structurally: ``-Q`` wires the interactive callback
    # (cli_agent_setup_mixin.py) which polls a queue nobody can answer, so the
    # timeout is the only thing that bounds the stall. It must be written as
    # ``clarify.timeout``: resolve_clarify_timeout() reads that key first and
    # hermes ships its own default of 120 there, so ``agent.clarify_timeout``
    # never wins (measured — setting only the agent key leaves the stall at 120s).
    clarify_timeout: int = 5
    # Exclusive toolset whitelist (-t). Keeps everything the model works with:
    # coding (terminal/file/patch/search/todo), code_execution, delegation
    # (delegate_task = subagents), debugging, memory, context_engine and
    # clarify. Left out: web/search/browser/vision/media (no route out of a task
    # pod), cron (no daemon survives the pod), skills authoring.
    # ⚠️ this list is fingerprint-constrained, not capability-constrained: with
    # session_search + skills + cronjob also enabled, subscription-style
    # gateway channels 400 the request as a third-party app (verified
    #  by bisecting the captured body: 13/18 tools pass, and those
    # three groups are what tips it over).
    # None runs hermes's own default toolsets.
    toolsets: str | None = "terminal,file,todo,code_execution,delegation,debugging,memory,context_engine,clarify"


class HermesAgent(HarnessAgent):
    """Runs Hermes Agent inside the pod under the MiMo Agent protocol."""

    NAME = "hermes"
    ERROR_STATUS = "HermesError"
    ENV_TRAJECTORY_FILES = ["/tmp/mimo-hermes-logs/hermes.txt"]
    # -Q prints the reply to stdout and session_id/diagnostics to stderr.
    SPLIT_STDERR = True

    def __init__(self, model, env, *, config_class=HermesAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._session_id: str | None = None

    @property
    def _home_dir(self) -> str:
        # state.db is SQLite WAL — pod-local disk only, never a network mount.
        return f"{self._logs_dir}/home"

    def _config_yaml(self) -> str:
        model_block: dict[str, Any] = {
            "default": self._resolved_model,
            "provider": _PROVIDER_ID,
        }
        if self.config.max_tokens:
            model_block["max_tokens"] = self.config.max_tokens
        agent_block: dict[str, Any] = {}
        if self.config.effort:
            agent_block["reasoning_effort"] = self.config.effort
        if self.config.max_turns is not None:
            agent_block["max_turns"] = self.config.max_turns
        config: dict[str, Any] = {
            "model": model_block,
            "providers": {
                _PROVIDER_ID: {
                    # SDK strips a trailing /v1 then appends /v1/messages
                    "base_url": self._gateway_base_url(v1=False),
                    "api_mode": "anthropic_messages",
                    "api_key": self._api_key,
                }
            },
            # --yolo equivalent, declared in config as well (belt & braces).
            # Note yolo governs approvals only — it does not reach clarify.
            "approvals": {"mode": "off"},
            # bounds how long an unanswerable clarify stalls the turn
            "clarify": {"timeout": self.config.clarify_timeout},
            # title generation is an extra LLM call that is not the task
            "auxiliary": {"title_generation": {"enabled": False}},
            # memory + its curator and the LSP are working aids, so they stay
            "curator": {"enabled": True},
            "memory": {"memory_enabled": True, "user_profile_enabled": False},
            "lsp": {"enabled": True},
            # tirith is not bundled; leaving it on both prints a startup
            # warning into stdout (which -Q otherwise reserves for the reply)
            # and makes the first tool call attempt a lazy install.
            "security": {"allow_lazy_installs": False, "tirith_enabled": False},
            # -Q still streams the thinking box to stdout; without this the
            # reasoning text ends up glued to the final reply. 2026.8.13 added
            # inline diff previews for write_file/patch, which land on stdout
            # ahead of the reply (complete with CRs) — off as well.
            "display": {"show_reasoning": False, "inline_diffs": False},
            "model_catalog": {"enabled": False},
        }
        if agent_block:
            config["agent"] = agent_block
        # YAML is a JSON superset — same trick as omp's models.yml.
        return json.dumps(config, ensure_ascii=False)

    def _stage_files(self) -> None:
        self.env.execute(f"mkdir -p {self._home_dir}")
        self._copy_text_to_pod(self._config_yaml(), f"{self._home_dir}/config.yaml")
        # Pre-seed the models.dev cache; a fresh mtime (<1h) skips the network
        # fetch entirely instead of a 15s timeout + stale-cache fallback.
        self.env.execute(
            f"cp {self._install_dir}/models_dev_cache.json {self._home_dir}/ "
            f"2>/dev/null && touch {self._home_dir}/models_dev_cache.json || true"
        )

    def _harness_env(self) -> dict[str, str]:
        return {
            "HERMES_HOME": self._home_dir,
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
            # payload build patch: treat the gateway as a native Anthropic
            # endpoint so signed thinking blocks replay instead of being
            # stripped by the third-party heuristic.
            "HERMES_FORCE_NATIVE_ANTHROPIC": "1",
            # payload build patch: replay the signed thinking blocks of EVERY
            # assistant turn, not just the latest one. Upstream's trim keeps
            # the payload smaller but rewrites the cached prefix on every
            # turn (measured: one full cache miss mid-run), and the collected
            # trajectory loses the earlier reasoning.
            "HERMES_KEEP_ALL_THINKING": "1",
        }

    def _command_parts(self, task: str) -> list[str]:
        import shlex

        parts = [
            "hermes",
            "chat",
            "-Q",
            "--yolo",
            "--accept-hooks",
            f"--provider {_PROVIDER_ID}",
            f"--model {shlex.quote(self._resolved_model)}",
        ]
        if self.config.toolsets:
            parts.append(f"-t {shlex.quote(self.config.toolsets)}")
        if self.config.max_turns is not None:
            parts.append(f"--max-turns {self.config.max_turns}")
        if self._turns > 0 and self._session_id:
            parts.append(f"--resume {shlex.quote(self._session_id)} --no-restore-cwd")
        # the prompt travels as argv (no stdin path exists in 2026.8.13)
        parts.append(f'-q "$(cat {self._instruction_file})"')
        return parts

    def _parse_events(self, raw: str) -> list[dict[str, Any]]:
        # No event stream — keep the raw text for _parse_result.
        return [{"type": "raw", "text": raw}]

    def _parse_result(self, rc: int, events: list[dict[str, Any]]) -> tuple[str, str]:
        raw = (events[0].get("text") or "") if events else ""
        stderr_text = self._read_pod_text(f"{self._logs_dir}/{self.NAME}.stderr.txt")
        # SPLIT_STDERR appends stderr after stdout in the event log; peel it
        # back off so the reply is exactly the stdout half.
        stdout_text = raw
        if stderr_text and raw.endswith(stderr_text):
            stdout_text = raw[: -len(stderr_text)]

        match = _SESSION_ID_RE.search(stderr_text or raw)
        if match:
            self._session_id = match.group(1)

        self._fold_usage(self._session_usage())

        status = self.IDLE_STATUS if rc == 0 else self.ERROR_STATUS
        result_text = stdout_text.strip()
        if not result_text:
            # -Q always prints the final reply to stdout; empty = broken run
            status = self.ERROR_STATUS
            tail = (stderr_text or raw).strip()[-1500:]
            result_text = f"(no agent message; hermes rc={rc}) {tail}"
        return result_text, status

    def _session_usage(self) -> list[dict[str, int]]:
        """Session-level usage from state.db (this turn's session plus its
        descendant compaction/delegation sessions), read with the payload's
        own python so the pod needs nothing extra."""
        sid = self._session_id
        if not sid or not re.fullmatch(r"[\w.-]{1,64}", sid):
            return []
        query = (
            "WITH RECURSIVE t(id) AS ("
            "SELECT ? UNION ALL SELECT s.id FROM sessions s JOIN t ON s.parent_session_id = t.id) "
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
            "COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0), "
            "COALESCE(SUM(api_call_count),0) FROM sessions WHERE id IN (SELECT id FROM t)"
        )
        script = (
            "import sqlite3, json, sys; "
            f"db = sqlite3.connect('file:{self._home_dir}/state.db?mode=ro', uri=True); "
            f"print(json.dumps(db.execute({query!r}, ('{sid}',)).fetchone()))"
        )
        try:
            res = self.env.execute(f'{self._install_dir}/venv/bin/python -c "{script}" 2>/dev/null || true')
        except Exception as e:
            self.logger.warning(f"[{self.NAME}] could not read state.db usage: {e}")
            return []
        row = None
        for line in (res.get("output", "") or "").strip().splitlines():
            if line.startswith("["):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if not row:
            return []
        inp, outp, cread, cwrite, n_calls = (int(x or 0) for x in row)
        # usage is cumulative only within one turn's session tree; each
        # resume creates a fresh session id, so no cross-turn delta needed.
        try:
            self.model.n_calls += max(0, n_calls - 1)
        except Exception:
            pass
        return [{"input": inp, "output": outp, "cache_read": cread, "cache_write": cwrite}]
