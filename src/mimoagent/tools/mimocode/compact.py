"""Optional MiMo compact tool.

The model calls this when its context window is filling up (the
``<context_usage>`` footer on tool results signals the pressure). Rather than
rewriting history inside ``execute`` — which would strand this very tool call's
assistant message without its matching tool response — the tool only sets a
flag on the agent. The actual summarize-and-rebuild runs at the top of the next
``query()`` (see ``MimocodeAgent._do_compact``), once this compact round is fully
formed, so the tool-call/response pairing stays intact.
"""

from __future__ import annotations

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput


class CompactTool(BaseTool):
    """Summarize the conversation so far into a compact summary to free context."""

    @property
    def name(self) -> str:
        return "compact"

    @property
    def description(self) -> str:
        return """Summarize the conversation so far into a compact summary, freeing up context window.

Call this when the context window is getting full — the `<context_usage>` note on recent tool results
tells you the current usage. Compacting keeps you working on long tasks without the context overflowing
(which would otherwise abort the run) and without losing the thread among stale tool output.

What it does:
- Replaces the conversation history with a structured summary of the work so far (intent, files touched,
  code changes, errors and fixes, pending tasks, and where you left off). The system prompt is preserved.
- Older details NOT captured in the summary — full file contents you read, raw command output — will no
  longer be visible after compaction.

Before calling: make sure anything you still need is either already reflected in your recent work or noted
in your reasoning. Use the optional `instructions` to tell the summarizer what to keep verbatim (for
example a specific failing test's output or a file you must keep editing)."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": (
                        "Optional extra guidance for the summary, e.g. details to preserve verbatim "
                        "(a specific error message, a file you are mid-edit on)."
                    ),
                },
            },
            "required": [],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")
        context = context or {}
        agent = context.get("agent")
        if agent is None:
            raise ToolException(
                "Compact tool requires 'agent' in context. The parent agent must pass it "
                "via execute_action's context dict."
            )

        agent._compact_requested = True
        agent._compact_instructions = params.get("instructions")
        return ToolOutput(
            output=(
                "Compaction scheduled. The conversation history will be summarized before the next step; "
                "older details not captured in the summary will no longer be visible."
            ),
            success=True,
        )
