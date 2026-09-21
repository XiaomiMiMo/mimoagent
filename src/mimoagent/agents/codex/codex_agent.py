"""Codex-aligned native tool-calling agent.

``CodexAgent`` is a :class:`DefaultAgent` variant that speaks the Codex coding-tool
catalogue. When enabled, its code-mode surface matches ``codex-rs``: a
freeform JavaScript ``exec`` tool plus ``wait``; ``exec_command`` and
``apply_patch`` remain available through the global ``tools`` object.

* **Anti-reward-hacking guard** — an ``AntiHackGuard`` registered as an
  action interceptor; the field maps in ``agents/antihack.py`` cover
  ``exec_command.cmd`` and ``apply_patch.input``, and nested calls from
  JavaScript re-enter the guard.

The conversation is chat-shaped and queried through the ordinary ``responses``
model adapter. CodexAgent is deliberately responses-only: the Chat Completions
adapter is not a supported whitebox surface.

Selected via ``agent.type: codex-agent`` in yaml. Distinct from
``agent.type: codex``, which runs the upstream Codex CLI as a blackbox.
Normal ``DefaultAgent`` tool behavior is unchanged.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimoagent import Environment, Model
from mimoagent.agents.antihack import AntiHackConfig, AntiHackGuard
from mimoagent.agents.base import (
    FormatError,
    InfraError,
    LimitsExceeded,
    NonTerminatingException,
    TerminatingException,
)
from mimoagent.agents.default import DefaultAgent, DefaultAgentConfig, _collect_tool_calls
from mimoagent.compaction import CODEX_SUMMARY_PREFIX, build_codex_compact_prompt, build_local_compacted_history
from mimoagent.tools.codex import CodexToolRegistry
from mimoagent.tools.codex.apply_patch import parse_apply_patch_args
from mimoagent.tools.codex.code_mode import CodeModeRuntime, ExecTool, WaitTool, parse_exec_call_arguments
from mimoagent.tools.codex.output_budget import (
    APPROX_BYTES_PER_TOKEN,
    DEFAULT_MAX_OUTPUT_TOKENS,
    resolve_max_tokens,
    truncate_exec_output,
)
from mimoagent.utils.cal_token import rough_token_count_estimation_for_messages

# Distilled from the gpt-5.6 Codex system prompt for headless RL: the
# behaviour-shaping rules (rg/apply_patch/scope-of-action/final-message) are
# kept; the product/CLI parts (personality, user channels, visualizations,
# skills) are dropped. See docs/codex-whitebox-core-analysis.md.
#
# Also dropped on purpose (RL does not train these contracts):
#   - "Working with the user" / mid-turn user messages — models that read it
#     tend to wait on or request a user reply.
#   - automatic-compaction / "time never runs out" — compaction is optional
#     and off by default; promising it changes rollout length vs truncation.
_CORE_PROMPT = (Path(__file__).parent / "prompts" / "codex_core.txt").read_text(encoding="utf-8").rstrip()


@dataclass
class CodexAgentConfig(DefaultAgentConfig):
    system_template: str = _CORE_PROMPT
    tools: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"tool": "exec_command"},
            {"tool": "apply_patch"},
        ]
    )
    # The one hard bound on a tool result, applied by DefaultAgent when the
    # observation is formatted - the counterpart of codex-rs's history layer,
    # where every function call output is cut to the model's truncation policy
    # times 1.2 (10000 tokens x 1.2 for the gpt-5.x entries in models.json);
    # 48000 characters is that budget at four bytes per token. The model is
    # never told about it, exactly as upstream: the budgets it can see
    # (exec_command's max_output_tokens, the exec pragma, wait's max_tokens)
    # default to 10000 tokens and are uncapped, so this cut fires only when a
    # cell raises its pragma budget above the history budget.
    max_observation_length: int = int(DEFAULT_MAX_OUTPUT_TOKENS * 1.2) * APPROX_BYTES_PER_TOKEN
    # Not rendered: under ptc the exec/wait results carry no metadata anyway,
    # and the direct exec_command surface would otherwise show a per-call
    # wall_time_seconds - a value that differs on every run.
    show_tool_metadata: bool = False
    # Independent Codex-RS-style compaction. It uses a normal Responses summary
    # turn and replaces history with the recent user context plus a handoff.
    # Codex owns this automatic compaction; there is no model-facing Compact tool.
    compaction_enabled: bool = False
    compaction_trigger_tokens: int | None = None
    compaction_user_message_tokens: int = 20_000
    compaction_context_window: int = 100_000
    compaction_buffer_tokens: int = 10_000
    # Optional override for the summarization prompt's custom instructions.
    compaction_summary_prompt: str | None = None

    # Programmatic Tool Calling. Two states, no mixed surface:
    #   ptc=True  — the model sees only ``exec`` + ``wait`` and drives every tool
    #               by writing JavaScript. Requires ``protocol: responses`` (the
    #               grammar-constrained custom tool) and the code-mode host.
    #   ptc=False — the model sees the Codex tools directly (``exec_command``,
    #               ``apply_patch``, ...) and calls them itself. No JavaScript
    #               runtime, so no host binary. Still responses-only.
    # Both modes require ``protocol: responses``; the Chat adapter is unsupported.
    # Both are trained against, so both surfaces are model-facing contracts.
    ptc: bool = True

    code_mode_host_path: str | None = None
    code_mode_exec_yield_time_ms: int = 30_000
    code_mode_wait_yield_time_ms: int = 10_000
    code_mode_request_timeout: float = 60.0
    code_mode_delegate_workers: int = 8

    # Runtime anti-reward-hacking guard (see agents/antihack.py). Default-off: an
    # absent ``antihack:`` yaml block (or ``enabled: false``) is a strict no-op.
    antihack: Any = field(default_factory=AntiHackConfig)

    # Whether a nested contract error (unknown ``tools.*`` / bad arg type that
    # escapes the JS cell) marks the turn as a ``tool_call_error``. Default-off:
    # the model already sees "Script failed" in the exec output, and counting
    # nested slips made the ptc metric noisier than the direct-tool surface.
    # Detection still happens and the model still sees the script error; only
    # the metric bit is gated.
    nested_tool_call_errors: bool = False

    def __post_init__(self) -> None:
        if self.code_mode_exec_yield_time_ms < 0 or self.code_mode_wait_yield_time_ms < 0:
            raise ValueError("code-mode yield times must be non-negative")
        if self.code_mode_request_timeout <= 0:
            raise ValueError("code_mode_request_timeout must be positive")
        if self.code_mode_delegate_workers < 1:
            raise ValueError("code_mode_delegate_workers must be at least 1")


class CodexAgent(DefaultAgent):
    """Tool-calling agent over the Codex-aligned tool catalogue."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = CodexAgentConfig,
        **kwargs,
    ):
        protocol = getattr(getattr(model, "config", None), "protocol", "responses")
        if protocol != "responses":
            raise ValueError(
                f"CodexAgent requires protocol='responses' (got {protocol!r}); "
                "the Chat Completions adapter is not a supported Codex whitebox surface"
            )
        super().__init__(model, env, config_class=config_class, **kwargs)
        # Anti-hack guard as an action interceptor: a flagged call returns the
        # dummy from execute_action and never reaches _execute_tool.
        self.antihack = AntiHackGuard(self.config.antihack)
        self.add_action_interceptor(self.antihack)
        self._nested_contract_errors: list[dict] = []  # drained into tool_call_errors per step
        self._compact_count: int = 0
        self._compaction_cooldown: bool = False
        self.current_plan: list[dict[str, str]] = []
        self.plan_explanation: str | None = None
        self._exec_tool = ExecTool()
        self._wait_tool = WaitTool()
        self._code_mode_notification_lock = threading.Lock()
        self._code_mode_notifications: list[tuple[str, str]] = []
        self._code_mode_runtime: CodeModeRuntime | None = None
        if self.config.ptc:
            self._code_mode_runtime = CodeModeRuntime(
                self,
                host_path=self.config.code_mode_host_path,
                exec_yield_time_ms=self.config.code_mode_exec_yield_time_ms,
                wait_yield_time_ms=self.config.code_mode_wait_yield_time_ms,
                request_timeout=self.config.code_mode_request_timeout,
                delegate_workers=self.config.code_mode_delegate_workers,
                fatal_delegate_errors=(InfraError,),
            )
        # ptc replaces the direct-tool definitions DefaultAgent derived from the
        # registry with the exec/wait surface.
        self._tool_definitions = self._build_tool_definitions()

    def _build_tool_registry(self) -> CodexToolRegistry:
        # ptc only changes the model-facing surface (exec/wait vs direct tools).
        # Nested tools are exactly the configured catalogue in both modes — no
        # silent view_image/update_plan injection — so description, host
        # enablement, and ALL_TOOLS stay aligned with ptc=false.
        return CodexToolRegistry.from_config(self.config.tools)

    def step(self) -> dict | None:
        response = self.query()
        tool_calls = _collect_tool_calls(response)

        if not tool_calls:
            # Whitebox: no stray-<tool_call> rescue from the CC parser. A tool-call
            # block that never became a structured call is just assistant text;
            # the rollout ends Idle so the model must learn the real tool surface.
            self.tool_call_errors.append(False)
            return None  # → base loop returns ("Idle", last assistant text)

        outcomes = self._run_tool_calls(tool_calls)
        nested, self._nested_contract_errors = self._nested_contract_errors, []
        has_format_error = any(isinstance(o, FormatError) for o in outcomes)
        self.tool_call_errors.append(has_format_error or (self.config.nested_tool_call_errors and bool(nested)))

        terminating: TerminatingException | None = None
        for call, outcome in zip(tool_calls, outcomes):
            self._emit_outcome(call, outcome)
            if isinstance(outcome, TerminatingException) and terminating is None:
                terminating = outcome

        if terminating is not None:
            raise terminating
        return {"outcomes": outcomes}

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        if self._code_mode_runtime is not None:
            return self._code_mode_runtime.model_definitions()
        return [tool.get_function_definition() for tool in self.tool_registry.tools.values()]

    def _parse_tool_call(self, call: dict) -> dict:
        """``ptc`` selects the surface: ``exec``/``wait`` only, or direct tools only."""
        fn = call.get("function") or {}
        tool_name = fn.get("name")
        if self._code_mode_runtime is not None:
            if tool_name == "exec":
                try:
                    params = parse_exec_call_arguments(fn.get("arguments", ""))
                except Exception as exc:
                    raise FormatError(str(exc)) from exc
                return {"tool": "exec", "params": params, "tool_call_id": call.get("id")}
            if tool_name == "wait":
                raw = fn.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw.strip() or "{}")
                    except json.JSONDecodeError as exc:
                        raise FormatError(f"Invalid JSON arguments for tool 'wait': {exc}") from exc
                if not isinstance(raw, dict):
                    raise FormatError("Tool arguments for 'wait' must be a JSON object")
                return {"tool": "wait", "params": raw, "tool_call_id": call.get("id")}
            raise FormatError(f"Unknown tool '{tool_name}'. Available tools: {self._available_tool_names()}")
        if tool_name != "apply_patch":
            return super()._parse_tool_call(call)
        available = self.tool_registry.list_tools()
        if "apply_patch" not in available:
            raise FormatError(f"Unknown tool 'apply_patch'. Available tools: {available}")
        # Freeform: the body is raw patch text, not a JSON object.
        return {
            "tool": "apply_patch",
            "params": parse_apply_patch_args(fn.get("arguments", {})),
            "tool_call_id": call.get("id"),
        }

    def _format_observation(self, output: dict) -> str:
        """Direct-tool surface: keep the exit code visible now that metadata is not rendered.

        Under ptc the nested result object already carries ``exit_code``, so
        this only applies to ``ptc: false``. A direct ``exec_command`` result is
        cut here the way codex-rs's ``to_response_item`` cuts it: to the call's
        ``max_output_tokens`` when that is below the model policy, else to the
        policy (``DEFAULT_MAX_OUTPUT_TOKENS``); the ``max_observation_length``
        cut in the base class follows and does not fire on top of it.
        """
        metadata = output.get("metadata") or {}
        if "max_output_tokens" in metadata:
            budget = min(resolve_max_tokens(metadata.get("max_output_tokens")), DEFAULT_MAX_OUTPUT_TOKENS)
            text, _ = truncate_exec_output(
                output.get("output", "") or "", metadata.get("output_omitted_bytes") or 0, budget
            )
            output = {**output, "output": text}
        formatted = super()._format_observation(output)
        exit_code = (output.get("metadata") or {}).get("exit_code")
        if self._code_mode_runtime is None and exit_code not in (None, 0):
            formatted += f"\nProcess exited with code {exit_code}"
        return formatted

    def _available_tool_names(self) -> list[str]:
        if self._code_mode_runtime is not None:
            return ["exec", "wait"]
        return self.tool_registry.list_tools()

    def execute_action(self, action: dict) -> dict:
        """Dispatch code-mode controls; everything else keeps the anti-hack guard.

        Nested tool calls made from JavaScript re-enter here, so they run through
        the same interceptor chain as direct calls; flagged calls never touch the
        environment.
        """

        if action["tool"] not in {"exec", "wait"}:
            return super().execute_action(action)
        runtime = self._code_mode_runtime
        tool = self._exec_tool if action["tool"] == "exec" else self._wait_tool
        try:
            result = tool.execute(
                action.get("params", {}),
                {
                    "env": self.env,
                    "model": self.model,
                    "agent": self,
                    "code_mode_runtime": runtime,
                    "tool_call_id": action.get("tool_call_id"),
                },
            )
        except LimitsExceeded:
            raise
        except InfraError:
            raise
        except Exception as exc:
            raise NonTerminatingException(str(exc)) from exc
        return result.to_dict()

    def _record_nested_contract_error(self, tool: str, kind: str) -> None:
        self._nested_contract_errors.append({"tool": tool, "kind": kind})

    def _enqueue_code_mode_notification(self, call_id: str, text: str) -> None:
        with self._code_mode_notification_lock:
            self._code_mode_notifications.append((call_id, text))

    def _flush_code_mode_notifications(self) -> None:
        with self._code_mode_notification_lock:
            notifications, self._code_mode_notifications = self._code_mode_notifications, []
        for call_id, text in notifications:
            self.add_message(
                "tool",
                text,
                name="exec",
                tool_call_id=call_id,
                code_mode_notification=True,
            )

    def _emit_outcome(self, call: dict, outcome: Any) -> None:
        self._flush_code_mode_notifications()
        super()._emit_outcome(call, outcome)

    def query(self) -> dict:
        self._flush_code_mode_notifications()
        if self._compaction_cooldown:
            self._compaction_cooldown = False
        elif self.config.compaction_enabled and self._should_auto_compact():
            self._do_codex_compact()
        return super().query()

    def _should_auto_compact(self) -> bool:
        threshold = self.config.compaction_trigger_tokens
        if threshold is None:
            threshold = self.config.compaction_context_window - self.config.compaction_buffer_tokens
        return rough_token_count_estimation_for_messages(self.messages) >= max(1, int(threshold))

    def _do_codex_compact(self) -> None:
        """Run Codex-style compaction using an ordinary Responses turn."""
        source = [dict(message) for message in self.messages]
        prompt = build_codex_compact_prompt(self.config.compaction_summary_prompt)
        try:
            # Deliberately a normal model turn with a temporary user prompt.
            # No provider-specific compact endpoint.
            response = self.model.query([*source, {"role": "user", "content": prompt}])
        except Exception as exc:
            self.logger.warning(f"Codex compaction summary request failed; keeping history: {exc}")
            return

        summary = self._codex_text_of(response).strip()
        if summary.startswith(CODEX_SUMMARY_PREFIX):
            summary = summary[len(CODEX_SUMMARY_PREFIX) :].lstrip()
        if not summary:
            self.logger.warning("Codex compaction produced an empty summary; keeping history.")
            return

        pre_tokens = rough_token_count_estimation_for_messages(self.messages)
        self.messages = build_local_compacted_history(
            source,
            summary,
            max_user_tokens=self.config.compaction_user_message_tokens,
        )
        self._compact_count += 1
        self._compaction_cooldown = True
        if self.msg_path:
            self.msg_path.write_text("")
            for message in self.messages:
                self._append_msg_to_file(message)
        self.logger.info(
            f"Codex compacted conversation (#{self._compact_count}): ~{pre_tokens} tokens -> "
            f"{len(self.messages)} messages retained."
        )

    @staticmethod
    def _codex_text_of(message: dict) -> str:
        content = message.get("content")
        if isinstance(content, list):
            return "".join(
                str(piece.get("text") or piece.get("content") or "") if isinstance(piece, dict) else str(piece)
                for piece in content
            )
        return str(content or "")

    def close(self) -> None:
        if self._code_mode_runtime is not None:
            self._code_mode_runtime.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
