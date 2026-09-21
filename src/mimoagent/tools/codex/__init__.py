"""Codex-aligned tool catalogue.

``exec_command`` + ``apply_patch`` match the Codex nested-tool schemas
(``cmd``/``workdir``/``yield_time_ms``, the latter a hard deadline here, V4A
freeform patch). The native
``CodexAgent`` can expose them directly or through the code-rs ``exec`` +
``wait`` surface backed by the standalone sandboxed V8 host.

Used only by :class:`mimoagent.agents.codex.codex_agent.CodexAgent`
(``agent.type: codex-agent``). The CC catalogue and the blackbox ``codex``
CLI agent are untouched.
"""

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException
from mimoagent.tools.codex.apply_patch import ApplyPatchTool
from mimoagent.tools.codex.exec_command import ExecCommandTool
from mimoagent.tools.codex.update_plan import UpdatePlanTool
from mimoagent.tools.codex.view_image import ViewImageTool
from mimoagent.tools.registry import ToolRegistry

CODEX_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "exec_command": ExecCommandTool,
    "apply_patch": ApplyPatchTool,
    "view_image": ViewImageTool,
    "update_plan": UpdatePlanTool,
}


class CodexToolRegistry(ToolRegistry):
    """ToolRegistry over the Codex-aligned catalogue."""

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "CodexToolRegistry":
        registry = cls()
        for tool_spec in config:
            tool_name = tool_spec.get("tool")
            if not tool_name:
                raise ToolException(f"Tool specification missing 'tool' field: {tool_spec}")
            tool_class = CODEX_TOOL_CLASSES.get(tool_name)
            if tool_class is None:
                raise ToolException(f"Unknown tool type: {tool_name}. Codex catalogue: {sorted(CODEX_TOOL_CLASSES)}")
            registry.register(tool_class(tool_spec.get("config", {})))
        return registry


__all__ = [
    "ApplyPatchTool",
    "CODEX_TOOL_CLASSES",
    "CodexToolRegistry",
    "ExecCommandTool",
    "UpdatePlanTool",
    "ViewImageTool",
]
