"""Codex ``update_plan``: maintain a checklist for the current agent turn."""

from __future__ import annotations

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

_STATUSES = ("pending", "in_progress", "completed")
_STATUS_SET = set(_STATUSES)


class UpdatePlanTool(BaseTool):
    """Update the agent's current checklist and return the code-rs result."""

    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return (
            "Updates the task plan.\n"
            "Provide an optional explanation and a list of plan items, each with a step and status.\n"
            "At most one step can be in_progress at a time."
        )

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "explanation": {"type": "string", "description": "Optional explanation for this plan update."},
                "plan": {
                    "type": "array",
                    "description": "The list of steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "Task step text."},
                            "status": {"type": "string", "enum": list(_STATUSES), "description": "Step status."},
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if not isinstance(params, dict):
            raise ToolException(f"update_plan params must be a dictionary, got {type(params).__name__}")
        unsupported = set(params) - {"explanation", "plan"}
        if unsupported:
            raise ToolException(f"update_plan has unsupported field(s): {', '.join(sorted(unsupported))}")
        plan = params.get("plan")
        if not isinstance(plan, list):
            raise ToolException("update_plan requires 'plan' to be a list")
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(plan):
            if not isinstance(item, dict) or set(item) != {"step", "status"}:
                raise ToolException(f"update_plan plan item {index} must contain only 'step' and 'status'")
            step, status = item["step"], item["status"]
            if not isinstance(step, str) or not step:
                raise ToolException(f"update_plan plan item {index} has an invalid step")
            if not isinstance(status, str) or status not in _STATUS_SET:
                raise ToolException(f"update_plan plan item {index} has an invalid status: {status!r}")
            normalized.append({"step": step, "status": status})
        explanation = params.get("explanation")
        if explanation is not None and not isinstance(explanation, str):
            raise ToolException("update_plan 'explanation' must be a string")

        agent = (context or {}).get("agent")
        if agent is not None:
            agent.current_plan = normalized
            agent.plan_explanation = explanation
        metadata = {"plan": normalized}
        if explanation is not None:
            metadata["explanation"] = explanation
        return ToolOutput(output="Plan updated", metadata=metadata)
