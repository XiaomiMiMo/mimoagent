"""Tool-calling agent.

Each step the model may emit zero or more tool calls:

* **Zero calls** — the agent is idle. ``step()`` returns ``None`` and the base
  loop exits with status ``"Idle"``. The caller can invoke ``run()`` again with
  a new user message to resume the same conversation/environment.
* **One call** — dispatched to its tool synchronously; result appended as a
  ``tool`` message.
* **Multiple calls** — dispatched to their tools **in parallel**; each result is
  appended as a ``tool`` message in the original call order.
"""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mimoagent import Environment, Model
from mimoagent.agents.base import (
    AgentConfig,
    BaseAgent,
    FormatError,
    InfraError,
    LimitsExceeded,
    NonTerminatingException,
    TerminatingException,
)
from mimoagent.environments import TransportError
from mimoagent.tools import ToolException, ToolRegistry
from mimoagent.utils.log import get_log_context, use_log_context
from mimoagent.utils.truncate import truncate_middle

_WRAP_UP_HINT = (
    "[SYSTEM NOTICE: CONTEXT BUDGET] ~{pct}% of the context window remains. Enter completion mode.",
    "Prioritize finishing a correct solution over further exploration. Complete the current approach, perform only the highest-value verification or fixes, and preserve enough budget to submit the final answer. Avoid optional investigation, refactoring, or polishing. Continue working only when it materially increases the probability of task success.",
)


def _render_wrap_up_hint(pct: int) -> str:
    """Fill ``{pct}`` (remaining context %) and join the hint lines."""
    return "\n".join(part.format(pct=pct) for part in _WRAP_UP_HINT)


@dataclass
class DefaultAgentConfig(AgentConfig):
    system_template: str = "You are a helpful assistant that can interact with a computer using tools."
    instance_template: str = "Your task: {{task}}"

    tools: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"tool": "bash"},
            {"tool": "read"},
            {"tool": "write"},
            {"tool": "edit"},
            {"tool": "agent"},
        ]
    )
    show_tool_metadata: bool = True
    # Per-tool-response character budget (NOT tokens): outputs longer than this
    # keep the first and last half, with a loud truncation notice in between.
    # For this catalogue it is the single truncation point — tools themselves
    # return untruncated text (see utils.truncate). Set it to 0 to disable the
    # cut entirely, which is what a catalogue that budgets its own output does
    # (see CodexAgentConfig) so an observation is never cut twice.
    max_observation_length: int = 40000
    tool_parallel_workers: int = 8
    # When True, inject a user-turn wrap-up hint once this request's
    # input_tokens + output_tokens crosses ``wrap_up_hint_fraction`` of the
    # context-token budget. Default off — opt in from yaml. Distinct from
    # Compact's <context_usage> footer: this tells the model to finish, not
    # to compress and continue.
    wrap_up_hint: bool = False
    wrap_up_hint_fraction: float = 0.9
    # Token-budget denominator for the hint. None falls back to the CC/Mimocode
    # compaction budget when that method exists; otherwise the hint never fires.
    wrap_up_hint_context_window: int | None = None


# Pre-dispatch hook registered with ``DefaultAgent.add_action_interceptor``.
# Called with the parsed action; return a result dict to stand in for the tool
# call (the tool never runs) or ``None`` to let the call continue.
ActionInterceptor = Callable[[dict], dict | None]


class DefaultAgent(BaseAgent):
    """Tool-calling agent. Each step executes 0..N tool calls; multiple calls run in parallel."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = DefaultAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.tool_registry = self._build_tool_registry()
        self._tool_definitions = self.tool_registry.get_function_definitions()
        # Pre-dispatch policies (e.g. the anti-hack guard), consulted in
        # registration order by ``execute_action``. Populated in ``__init__``;
        # the parallel tool-call workers read it without a lock.
        self._action_interceptors: list[ActionInterceptor] = []

    def _build_tool_registry(self) -> ToolRegistry:
        """Catalogue variants override this instead of re-implementing ``__init__``."""
        return ToolRegistry.from_config(self.config.tools)

    def add_action_interceptor(self, interceptor: ActionInterceptor) -> None:
        """Register a pre-dispatch hook.

        ``execute_action`` calls each interceptor in registration order with the
        parsed action. The first one to return a result dict short-circuits the
        call: the tool never runs and that dict is the observation. Returning
        ``None`` passes the action on. Interceptors see every call routed
        through ``execute_action``, including nested calls a tool makes back on
        its agent.
        """
        self._action_interceptors.append(interceptor)

    def step(self) -> dict | None:
        response = self.query()
        tool_calls = _collect_tool_calls(response)

        if not tool_calls:
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

    def execute_action(self, action: dict) -> dict:
        """Run a single tool call: interceptors first, then the tool.

        Two seams, one above and one below. Policies that may substitute their
        own result for a call register via ``add_action_interceptor``;
        decorators of real results override ``_execute_tool``. Keeping them
        apart means a substituted result is never decorated.
        """
        for intercept in self._action_interceptors:
            result = intercept(action)
            if result is not None:
                return result
        return self._execute_tool(action)

    def _execute_tool(self, action: dict) -> dict:
        """Run a single tool and return its output as a dict.

        A ToolException is surfaced as a NonTerminatingException whose message is
        the exception content verbatim — this becomes the body of the tool
        response message emitted for this tool call.

        A TransportError (from KubernetesEnvironment when raise_on_transport_error
        is set) is escalated to InfraError, terminating the agent loop immediately.
        """
        tool_name = action["tool"]
        try:
            tool = self.tool_registry.get(tool_name)
            # ``model`` is included so tools that need to spawn a subagent
            # (see ``tools.agent.AgentTool``) can reach the shared model. Most
            # tools ignore it and only touch ``env``. ``agent`` is the calling
            # agent itself — AgentTool registers any spawned subagent on its
            # parent's ``subagents`` list via this handle.
            result = tool.execute(action.get("params", {}), self.get_tool_context())
        except LimitsExceeded:
            raise
        except TransportError as e:
            raise InfraError(str(e))
        except ToolException as e:
            raise NonTerminatingException(str(e))
        except Exception as e:
            raise NonTerminatingException(f"Unexpected error executing tool '{tool_name}': {e}")

        return result.to_dict()

    def get_tool_context(self) -> dict[str, Any]:
        """Return execution context for tools.

        Subclasses may add agent-local runtime state without changing the
        shared tool execution contract.
        """
        return {"env": self.env, "model": self.model, "agent": self}

    def get_model_query_kwargs(self) -> dict:
        return {"tools": self._tool_definitions, "tool_choice": "auto"}

    def after_step(self) -> None:
        """Inject a wrap-up user turn when the context budget is almost gone."""
        self._maybe_nudge_wrap_up()

    def _maybe_nudge_wrap_up(self) -> None:
        if not self.config.wrap_up_hint:
            return
        text = self._wrap_up_hint_text()
        if not text:
            return
        # User turn after the tool results (or FormatError correction). Valid
        # for chat/Anthropic/Responses; Anthropic merges a trailing user nudge
        # into the same user turn as the tool results.
        self.add_message("user", text)

    def _wrap_up_hint_text(self) -> str:
        """Return the hint body, or ``""`` if the last LLM call is still under the fraction.

        Occupancy is this request's ``input_tokens + output_tokens`` from the
        usage block (``_last_request_tokens``), not a chars/4 scan of history
        and not cumulative ``token_stats``. ``{pct}`` is remaining window
        percent (0 once this call is at or past the budget). Missing usage
        (delta 0) does not fire.
        """
        fraction = self.config.wrap_up_hint_fraction
        if fraction <= 0:
            return ""
        budget = self._wrap_up_context_budget()
        if budget is None:
            return ""
        used = self._last_request_tokens
        if used <= 0 or used < fraction * budget:
            return ""
        remaining_pct = max(0, round(100 * (1 - used / budget)))
        return _render_wrap_up_hint(remaining_pct)

    def _wrap_up_context_budget(self) -> int | None:
        window = self.config.wrap_up_hint_context_window
        if window is not None:
            return max(1, int(window))
        compaction_budget = getattr(self, "compaction_budget_tokens", None)
        if callable(compaction_budget):
            return compaction_budget()
        return None

    def _run_tool_calls(self, tool_calls: list[dict]) -> list:
        """Parse + execute each call. Returns a list of dict outputs or captured exceptions.

        The returned list aligns 1:1 with ``tool_calls`` in the original order, even when
        calls run in parallel (``executor.map`` preserves input order). ``run_one``
        never raises — every call produces exactly one outcome so that each
        tool_call_id gets a matching tool response downstream.
        """

        def run_one(call: dict):
            try:
                action = self._parse_tool_call(call)
            except FormatError as e:
                return e
            try:
                return self.execute_action(action)
            except (TerminatingException, NonTerminatingException) as e:
                return e
            except Exception as e:
                return NonTerminatingException(str(e))

        if len(tool_calls) == 1:
            return [run_one(tool_calls[0])]

        # ThreadPoolExecutor workers start with an empty context. We can't share
        # a ``contextvars.copy_context()`` snapshot via ``Context.run()`` either
        # — Context.run is not re-entrant, so two concurrent workers on the
        # same Context raise ``cannot enter context: ... is already entered``.
        # Instead, snapshot the one value we actually care about and re-bind it
        # inside each worker.
        parent_log_ctx = get_log_context()

        def run_in_parent_ctx(call):
            with use_log_context(parent_log_ctx):
                return run_one(call)

        workers = min(len(tool_calls), max(1, self.config.tool_parallel_workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(run_in_parent_ctx, tool_calls))

    def _parse_tool_call(self, call: dict) -> dict:
        fn = call.get("function") or {}
        tool_name = fn.get("name")
        if not tool_name:
            raise FormatError("Tool call missing function name.")
        available = self.tool_registry.list_tools()
        if tool_name not in available:
            raise FormatError(f"Unknown tool '{tool_name}'. Available tools: {available}")
        return {
            "tool": tool_name,
            "params": _parse_arguments(tool_name, fn.get("arguments", {})),
            "tool_call_id": call.get("id"),
        }

    def _emit_outcome(self, call: dict, outcome: Any) -> None:
        """Append a `tool` message for a single call's outcome (success or error)."""
        fn = call.get("function") or {}
        tool_name = fn.get("name") or "unknown"
        tool_call_id = call.get("id")

        media: list[dict] = []
        if isinstance(outcome, dict):
            body = self._format_observation(outcome)
            media = outcome.get("media") or []
        else:
            body = truncate_middle(str(outcome), self.config.max_observation_length)

        content: Any = body
        if media:
            # Multimodal tool result: text first, then one part per media entry.
            # The model layer converts these to its wire format (and relocates
            # them if the protocol forbids media in tool messages).
            content = [{"type": "text", "text": body}] + [_media_content_part(entry) for entry in media]

        message_kwargs: dict[str, Any] = {"name": tool_name}
        if tool_call_id:
            message_kwargs["tool_call_id"] = tool_call_id
        self.add_message("tool", content, **message_kwargs)

    def _format_observation(self, output: dict) -> str:
        # Truncate the tool output alone, then append metadata: the metadata
        # line (returncode etc.) must survive even when the output is clipped.
        formatted = truncate_middle(output.get("output", "") or "", self.config.max_observation_length)
        if self.config.show_tool_metadata and output.get("metadata"):
            formatted += f"\n\nTool metadata: {output['metadata']}"
        return formatted


def _collect_tool_calls(response: dict) -> list[dict]:
    tool_calls = response.get("tool_calls") or []
    return [tc for tc in tool_calls if isinstance(tc, dict)]


def _media_content_part(entry: dict) -> dict:
    """ToolOutput media entry -> chat-format content part (MiMo dialect)."""
    kind, media_type, data = entry.get("kind"), entry.get("media_type", ""), entry.get("data", "")
    if kind == "audio":
        return {"type": "input_audio", "input_audio": {"data": data, "format": entry.get("format", "wav")}}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": f"data:{media_type};base64,{data}"}}
    image_url = {"url": f"data:{media_type};base64,{data}"}
    if entry.get("detail"):
        image_url["detail"] = entry["detail"]
    return {"type": "image_url", "image_url": image_url}


def _parse_arguments(tool_name: str, raw: Any) -> dict:
    if isinstance(raw, str):
        raw = raw.strip() or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FormatError(f"Invalid JSON arguments for tool '{tool_name}': {exc}") from exc
    else:
        parsed = raw or {}
    if not isinstance(parsed, dict):
        raise FormatError(f"Tool arguments for '{tool_name}' must be a JSON object, got {type(parsed).__name__}")
    return parsed
