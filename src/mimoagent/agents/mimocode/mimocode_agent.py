"""Native MiMo-Code non-Codex agent.

The core loop exposes MiMo's lowercase Bash/Read/Write/Edit/Grep/Glob/Task
catalogue with agent-local cwd/task state. ``actor.run`` and ``compact`` are
optional non-core controls; checkpoint/memory product features are outside
this RL agent.

Selected via ``agent.type: mimocode-agent`` in YAML.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimoagent import Environment, Model
from mimoagent.agents.antihack import AntiHackConfig, AntiHackGuard
from mimoagent.agents.base import FormatError, TerminatingException
from mimoagent.agents.default import DefaultAgent, DefaultAgentConfig, _collect_tool_calls
from mimoagent.compaction import (
    build_summary_request_message,
    create_compact_boundary_message,
    create_compact_summary_message,
    format_compact_summary,
    messages_for_compaction,
    original_user_issue_anchor_message,
    split_leading_system_messages,
)
from mimoagent.tools.mimocode import MIMOCODE_CORE_TOOL_NAMES, MimocodeToolRegistry
from mimoagent.tools.mimocode.state import new_tool_state
from mimoagent.utils.cal_token import rough_token_count_estimation_for_messages

_CORE_PROMPT = (Path(__file__).parent / "prompts" / "mimocode_core.txt").read_text(encoding="utf-8").rstrip()


@dataclass
class MimocodeAgentConfig(DefaultAgentConfig):
    system_template: str = _CORE_PROMPT
    tools: list[dict[str, Any]] = field(default_factory=lambda: [{"tool": name} for name in MIMOCODE_CORE_TOOL_NAMES])
    # The MiMo catalogue bounds its own observations the way MiMo-Code does
    # (bash: last 2000 lines / 50 KB; read: 2000 lines, 50 KB; grep/glob: 100
    # entries), so the inherited char cut is off. Stacked on top it was the
    # tighter cut and ran second: a read came back as head + tail with the
    # tool's own "Showing lines 1-2000 ... Use offset=2001" footer intact while
    # the middle it claimed to show was gone, and the banner quoted the already
    # line-clipped length as the original. Set a positive value in yaml only to
    # restore a hard character ceiling for an experiment.
    max_observation_length: int = 0
    # ToolOutput.metadata stays in the object for logs and tests but is not
    # rendered: every field is either constant per rollout (cwd), a duplicate
    # of the output text (grep/glob counts), or a signal a deployment harness
    # will not provide (bash returncode) - training on it would be learning a
    # crutch that vanishes at inference time.
    show_tool_metadata: bool = False
    # Same env-facing pool size as DefaultAgent/CCAgent. Tool calls hit the
    # environment, not the model; actor children that each run a model loop
    # are separately gated by ``actor_max_concurrency``.
    tool_parallel_workers: int = 8
    # Cap on concurrently running ``actor.run`` children for this agent.
    # Values below 1 clamp to 1. Blocking acquire — a queued actor waits.
    actor_max_concurrency: int = 1

    # Model-decided compaction (see mimoagent.compaction and the Compact tool).
    # The model frees context by calling Compact; these only size the usage
    # signal and the post-compaction history. No automatic threshold trigger.
    compaction_context_window: int = 1_000_000
    compaction_buffer_tokens: int = 10_000
    # Budget used as the denominator of the usage %. Defaults to
    # context_window - buffer when None.
    compaction_threshold_tokens: int | None = None
    # Show the <context_usage> footer on tool results once usage crosses the
    # warn fraction, giving the model a concrete trigger for when to compact.
    show_context_usage: bool = True
    context_usage_warn_fraction: float = 0.8
    # Optional override for the summarization prompt's custom instructions.
    compaction_summary_prompt: str | None = None

    # Runtime anti-reward-hacking guard (see agents/antihack.py). Default-off: an
    # absent ``antihack:`` yaml block (or ``enabled: false``) is a strict no-op.
    antihack: Any = field(default_factory=AntiHackConfig)


class MimocodeAgent(DefaultAgent):
    """Tool-calling agent over the MiMo-Code lowercase catalogue."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = MimocodeAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        # Anti-hack guard as an action interceptor: a flagged call returns the
        # dummy from execute_action and never reaches _execute_tool.
        self.antihack = AntiHackGuard(self.config.antihack)
        self.add_action_interceptor(self.antihack)
        # Model-decided compaction state. The Compact tool sets
        # ``_compact_requested`` (and optional ``_compact_instructions``); the
        # next ``query()`` performs the summarize-and-rebuild and increments
        # ``_compact_count``. Deferring to ``query()`` keeps the Compact tool
        # call paired with its tool response (see tools/cc/compact.py).
        self._compact_requested: bool = False
        self._compact_instructions: str | None = None
        self._compact_count: int = 0
        # Runtime state belongs to this agent/context, not the shared Environment.
        env_config = getattr(env, "config", None)
        self.tool_state: dict[str, Any] = new_tool_state(getattr(env_config, "cwd", "") or "")
        # Per-agent gate for actor children. ActorTool acquires this before
        # spawning a child so concurrent actor.run calls (possible once
        # tool_parallel_workers > 1) cannot stampede the inference service.
        self.actor_semaphore = threading.Semaphore(max(1, int(self.config.actor_max_concurrency)))

    def _build_tool_registry(self) -> MimocodeToolRegistry:
        # The MiMo registry is separate so its lowercase protocol can evolve
        # without changing the CC catalogue.
        return MimocodeToolRegistry.from_config(self.config.tools)

    def get_tool_context(self) -> dict[str, Any]:
        """Add MiMo's replayable per-agent state to the shared tool context."""
        return super().get_tool_context() | {"state": self.tool_state}

    def _parse_tool_call(self, call: dict) -> dict:
        """Enforce the advertised schema: a required or mistyped argument is a
        FormatError (counted in ``tool_call_errors``) instead of a silent pass."""
        action = super()._parse_tool_call(call)
        schema = self.tool_registry.get(action["tool"]).get_function_parameters()
        problem = _schema_violation(action["params"], schema)
        if problem:
            raise FormatError(f"Invalid arguments for tool '{action['tool']}': {problem}")
        return action

    # --- step: stray <tool_call> detection --------------------------------------

    def step(self) -> dict | None:
        response = self.query()
        tool_calls = _collect_tool_calls(response)

        if not tool_calls:
            stray = _extract_stray_tool_call(response.get("content") or "")
            if stray is not None:
                # Model emitted a <tool_call> block that the upstream parser
                # didn't surface as a structured tool_call (typically: SGLang
                # detector dropped an unknown tool name into normal_text and
                # the env-var forward gate didn't fire). Raise FormatError so
                # the base loop appends a user-message correction and the
                # rollout continues — same teaching signal the model gets when
                # it calls a known-but-bad-args tool.
                available = self.tool_registry.list_tools()
                self.tool_call_errors.append(True)
                raise FormatError(f"Unknown tool '{stray}'. Available tools: {available}")
            self.tool_call_errors.append(False)
            return None  # → base loop returns ("Idle", last assistant text)

        outcomes = self._run_tool_calls(tool_calls)
        self.tool_call_errors.append(any(isinstance(o, FormatError) for o in outcomes))

        terminating: TerminatingException | None = None
        for call, outcome in zip(tool_calls, outcomes):
            self._emit_outcome(call, outcome)
            if isinstance(outcome, TerminatingException) and terminating is None:
                terminating = outcome

        if terminating is not None:
            raise terminating
        return {"outcomes": outcomes}

    def _execute_tool(self, action: dict) -> dict:
        """Attach the current agent-local cwd to every real tool result.

        Only real results are wrapped: the anti-hack interceptor returns the
        dummy from ``execute_action`` before reaching here, so a blocked call
        keeps its empty metadata and can't be told apart from an ordinary failure.
        """
        result = super()._execute_tool(action)
        result.setdefault("metadata", {}).setdefault("cwd", self.tool_state.get("cwd", ""))
        return result

    # --- compaction ---------------------------------------------------------------

    def query(self) -> dict:
        if self._compact_requested:
            # Compaction was requested by a Compact tool call on the previous
            # step. Run it now — the Compact assistant+tool round is fully
            # formed in ``self.messages``, so summarizing and rebuilding here
            # leaves no orphaned tool call.
            self._do_compact()
            self._compact_requested = False
            self._compact_instructions = None
        return super().query()

    def compaction_budget_tokens(self) -> int:
        """Token budget used as the denominator for the context-usage signal."""
        if self.config.compaction_threshold_tokens is not None:
            return max(1, int(self.config.compaction_threshold_tokens))
        return max(1, self.config.compaction_context_window - self.config.compaction_buffer_tokens)

    def _do_compact(self) -> None:
        """Summarize history so far and replace it with [system…, boundary, anchor, summary].

        Best-effort: if the summarizer returns no usable summary, history is
        left untouched so the model can keep working (or retry compaction).
        """
        slice_ = messages_for_compaction(self.messages)
        instructions = self.config.compaction_summary_prompt or self._compact_instructions
        request = build_summary_request_message(instructions)
        pre_tokens = rough_token_count_estimation_for_messages(self.messages)
        try:
            # Plain-text summary turn: no tools/tool_choice, unlike a normal
            # step (see get_model_query_kwargs) so the model can't try to call
            # a tool during summarization.
            response = self.model.query([*slice_, request])
        except Exception as e:
            self.logger.warning(f"Compaction model query failed; keeping history. {e}")
            return
        summary = format_compact_summary(self._text_of(response))
        if not summary:
            self.logger.warning("Compaction produced an empty summary; keeping history.")
            return

        system_messages, _ = split_leading_system_messages(self.messages)
        anchor_msg = original_user_issue_anchor_message(self.messages)
        boundary = create_compact_boundary_message(compact_count=self._compact_count, pre_tokens=pre_tokens)
        summary_msg = create_compact_summary_message(summary)
        # Reassigning self.messages bypasses add_message's file dump; write the
        # injected messages through it so the on-disk transcript records the cut.
        compacted_messages = [*system_messages, boundary]
        if anchor_msg is not None:
            compacted_messages.append(anchor_msg)
        compacted_messages.append(summary_msg)
        self.messages = compacted_messages
        self._compact_count += 1
        if self.msg_path:
            self._append_msg_to_file(boundary)
            if anchor_msg is not None:
                self._append_msg_to_file(anchor_msg)
            self._append_msg_to_file(summary_msg)
        self.logger.info(
            f"Compacted conversation (#{self._compact_count}): ~{pre_tokens} tokens -> summary "
            f"({len(self.messages)} messages retained)."
        )

    @staticmethod
    def _text_of(message: dict) -> str:
        """Plain-text content of a message dict (handles list-of-parts content)."""
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for piece in content:
                if isinstance(piece, dict):
                    parts.append(piece.get("text") or piece.get("content") or "")
                else:
                    parts.append(str(piece))
            return "".join(parts)
        return content or ""

    # --- observation formatting -----------------------------------------------

    def _format_observation(self, output: dict) -> str:
        return super()._format_observation(output) + self._context_usage_footer()

    def _context_usage_footer(self) -> str:
        """A <context_usage> note nudging the model to Compact, shown only once
        usage crosses the warn fraction so normal turns stay clean.

        Appended after truncation so it survives, and so the model gets a live
        pressure signal it can learn to act on during RL. Suppressed when the
        agent has no Compact tool (e.g. subagents) — the nudge would be moot."""
        if not self.config.show_context_usage or "compact" not in self.tool_registry.tools:
            return ""
        budget = self.compaction_budget_tokens()
        used = rough_token_count_estimation_for_messages(self.messages)
        if used < self.config.context_usage_warn_fraction * budget:
            return ""
        pct = round(100 * used / budget)
        return (
            f"\n\n<context_usage>~{used} / ~{budget} tokens ({pct}%). "
            f"Call compact to summarize history and free context.</context_usage>"
        )


_STRAY_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# JSON Schema scalar/container types -> accepted Python types. bool is excluded
# from the numeric types on purpose (it is an int subclass).
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _schema_violation(params: dict, schema: dict) -> str | None:
    """First top-level ``required`` / ``type`` violation of ``params`` against ``schema``, else None."""
    for key in schema.get("required", ()):
        if key not in params:
            return f"missing required parameter '{key}'"
    for key, spec in schema.get("properties", {}).items():
        if key not in params:
            continue
        expected = _JSON_TYPES.get(spec.get("type", ""))
        value = params[key]
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected)):
            return f"parameter '{key}' must be of type {spec['type']}, got {type(value).__name__}"
    return None


def _extract_stray_tool_call(content: str) -> str | None:
    """Return the tool name from a stray ``<tool_call>`` block in assistant text.

    SGLang's ``mimo_detector`` re-injects unparseable / unknown-tool blocks
    back into ``normal_text`` (mimo_detector.py:175-178). The verl router's
    ``ToolCallTextFilter`` (black_box_router.py:28-83) usually strips these
    out of the streamed content, but on any path that bypasses that filter
    (non-streaming endpoint, future refactor, direct serving) the raw block
    can survive into ``message.content``. This helper detects it so the
    agent can emit a feedback turn instead of silently exiting idle.

    Returns:
        - the function name (e.g. ``"TodoWrite"``) when the block parses,
        - ``"<unparseable>"`` when a block exists but the JSON inside is
          malformed or missing a ``name`` field,
        - ``None`` when no block is present.
    """
    if "<tool_call>" not in content:
        return None
    m = _STRAY_TOOL_CALL_RE.search(content)
    if not m:
        return "<unparseable>"
    body = m.group(1).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return "<unparseable>"
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if isinstance(name, str) and name:
            return name
    return "<unparseable>"
