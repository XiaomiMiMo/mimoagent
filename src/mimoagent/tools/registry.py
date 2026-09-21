"""Tool registry for managing available tools."""

from typing import Any

from mimoagent.tools.agent import AgentTool
from mimoagent.tools.base import BaseTool, ToolException
from mimoagent.tools.bash import BashTool
from mimoagent.tools.bashonly import BashOnlyTool
from mimoagent.tools.edit import EditTool
from mimoagent.tools.read import ReadTool
from mimoagent.tools.write import WriteTool

_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "bash": BashTool,
    "bash-only": BashOnlyTool,
    "read": ReadTool,
    "write": WriteTool,
    "edit": EditTool,
    "agent": AgentTool,
}


class ToolRegistry:
    """Registry for managing tool instances."""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self.tools:
            raise ToolException(f"Tool '{tool.name}' is already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise ToolException(f"Unknown tool '{name}'. Available tools: {list(self.tools)}")
        return self.tools[name]

    def list_tools(self) -> list[str]:
        return list(self.tools)

    def get_function_definitions(self) -> list[dict[str, Any]]:
        return [tool.get_function_definition() for tool in self.tools.values()]

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "ToolRegistry":
        """Create a registry from configuration.

        Config format::

            [
                {"tool": "bash", "config": {...}},
                {"tool": "read", "config": {...}},
                {"tool": "write"},
                {"tool": "edit"},
            ]
        """
        registry = cls()
        for tool_spec in config:
            tool_name = tool_spec.get("tool")
            if not tool_name:
                raise ToolException(f"Tool specification missing 'tool' field: {tool_spec}")
            tool_class = _TOOL_CLASSES.get(tool_name)
            if tool_class is None:
                raise ToolException(f"Unknown tool type: {tool_name}")
            registry.register(tool_class(tool_spec.get("config", {})))
        return registry
