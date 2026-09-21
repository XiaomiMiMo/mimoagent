"""Small rollout-scoped task ledger for the MiMo core set.

The ledger lives in the primary agent's state, so it survives every tool call
and is serialized into tool metadata/trajectory. A child context inherits the
same nested ledger during ``actor.run``; persistence outside that rollout tree
is intentionally not part of the contract.
"""

from __future__ import annotations

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.mimocode.prompts import load_tool_prompt

_ACTIONS = ("create", "list", "get", "start", "block", "unblock", "done", "abandon", "rename")
_STATUSES = ("open", "in_progress", "blocked", "done", "abandoned")
_TERMINAL = {"done", "abandoned"}


class TaskTool(BaseTool):
    """Manage a hierarchical task ledger in agent-local persistent state."""

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return load_tool_prompt("task")

    def get_function_parameters(self) -> dict[str, Any]:
        def text(description: str | None = None) -> dict[str, Any]:
            field: dict[str, Any] = {"type": "string", "minLength": 1}
            if description:
                field["description"] = description
            return field

        def operation(
            action: str,
            properties: dict[str, Any],
            required: list[str],
        ) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"action": {"type": "string", "const": action}, **properties},
                "required": ["action", *required],
                "additionalProperties": False,
            }

        operations = [
            operation(
                "create",
                {
                    "summary": text("Task summary for a single task."),
                    "parent_id": text("Parent task id for sub-tasks."),
                },
                ["summary"],
            ),
            operation(
                "list",
                {
                    "status": {"type": "string", "enum": list(_STATUSES), "description": "Filter by status."},
                    "include_terminal": {
                        "type": "boolean",
                        "description": "Include done/abandoned tasks. Default false.",
                    },
                },
                [],
            ),
        ]
        for action, description in (
            ("get", None),
            ("start", "Short note on starting."),
            ("block", "Short reason for blocking."),
            ("unblock", "Short reason for unblocking."),
            ("done", "Short summary of what was completed."),
            ("abandon", "Short reason for abandoning."),
        ):
            properties = {"id": text("Task id, e.g. T1 or T1.1.")}
            if description:
                properties["event_summary"] = text(description)
            operations.append(operation(action, properties, ["id"]))
        operations.append(
            operation(
                "rename",
                {
                    "id": text("Task id, e.g. T1 or T1.1."),
                    "summary": text("New task summary."),
                },
                ["id", "summary"],
            )
        )
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "object",
                    "anyOf": operations,
                }
            },
            "required": ["operation"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if not isinstance(params, dict) or not isinstance(params.get("operation"), dict):
            raise ToolException("Task tool requires an operation object")
        operation = params["operation"]
        action = operation.get("action")
        if action not in _ACTIONS:
            raise ToolException(f"Unknown task action {action!r}; expected one of {list(_ACTIONS)}")
        tasks = self._tasks(context)

        if action == "create":
            summary = self._required_text(operation, "summary")
            parent_id = operation.get("parent_id")
            if parent_id and parent_id not in tasks:
                return self._error(tasks, f"Task {parent_id} not found")
            task_id = self._next_id(tasks, parent_id)
            tasks[task_id] = {
                "id": task_id,
                "parent_id": parent_id,
                "summary": summary,
                "status": "open",
            }
            return self._result(tasks, f"Created {task_id} (open): {summary}", {"id": task_id})

        if action == "list":
            status_filter = operation.get("status")
            if status_filter is not None:
                # An explicit status is the whole filter. Layering the default
                # terminal exclusion on top made `status="done"` self-cancelling:
                # the schema offers done/abandoned but the call returned nothing.
                selected = [task for task in tasks.values() if task["status"] == status_filter]
            else:
                include_terminal = bool(operation.get("include_terminal", False))
                selected = [task for task in tasks.values() if include_terminal or task["status"] not in _TERMINAL]
            if not selected:
                return self._result(tasks, "No tasks.", {"count": 0})
            lines = [f"{task['id']} [{task['status']}] {task['summary']}" for task in selected]
            return self._result(tasks, "\n".join(lines), {"count": len(selected)})

        task_id = self._required_text(operation, "id")
        task = tasks.get(task_id)
        if task is None:
            return self._error(tasks, f"Task {task_id} not found")

        if action == "get":
            return self._result(tasks, self._format_task(task), {"id": task_id, "status": task["status"]})
        if action == "rename":
            task["summary"] = self._required_text(operation, "summary")
            return self._result(tasks, f"Renamed {task_id}: {task['summary']}", {"id": task_id})

        transitions = {
            "start": "in_progress",
            "block": "blocked",
            "unblock": "open",
            "done": "done",
            "abandon": "abandoned",
        }
        task["status"] = transitions[action]
        if operation.get("event_summary"):
            task["event_summary"] = str(operation["event_summary"])
        return self._result(tasks, f"Updated {task_id}: {task['status']}", {"id": task_id, "status": task["status"]})

    @staticmethod
    def _tasks(context: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        state = (context or {}).get("state")
        if not isinstance(state, dict):
            raise ToolException("Task tool requires agent-local state")
        return state.setdefault("tasks", {})

    @staticmethod
    def _required_text(operation: dict[str, Any], field: str) -> str:
        value = operation.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ToolException(f"Task operation requires a non-empty {field!r}")
        return value.strip()

    @staticmethod
    def _next_id(tasks: dict[str, dict[str, Any]], parent_id: str | None) -> str:
        prefix = f"{parent_id}." if parent_id else "T"
        used = []
        for task_id in tasks:
            if not task_id.startswith(prefix):
                continue
            suffix = task_id[len(prefix) :]
            if suffix.isdigit():
                used.append(int(suffix))
        return f"{prefix}{max(used, default=0) + 1}"

    @staticmethod
    def _format_task(task: dict[str, Any]) -> str:
        extra = f" ({task['event_summary']})" if task.get("event_summary") else ""
        return f"{task['id']} [{task['status']}] {task['summary']}{extra}"

    def _result(self, tasks: dict[str, dict[str, Any]], output: str, metadata: dict[str, Any]) -> ToolOutput:
        return ToolOutput(output=output, metadata={**self._ledger_metadata(tasks), **metadata})

    def _error(self, tasks: dict[str, dict[str, Any]], output: str) -> ToolOutput:
        return ToolOutput(output=f"Error: {output}", success=False, metadata=self._ledger_metadata(tasks))

    @staticmethod
    def _ledger_metadata(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """A constant-size ledger summary.

        Capping the entry count is not enough: summaries are model-authored, so N
        entries of arbitrary length still escape ``max_observation_length`` \u2014
        metadata is appended after the output is truncated. The full ledger is
        already what ``list`` returns in the output, so keep only counts here.
        """
        counts: dict[str, int] = {}
        for task in tasks.values():
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        return {"task_count": len(tasks), "task_counts_by_status": counts}
