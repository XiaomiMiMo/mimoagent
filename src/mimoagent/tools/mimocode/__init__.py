"""MiMo-Code native lowercase tool catalogue."""

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException
from mimoagent.tools.mimocode.actor import ActorTool
from mimoagent.tools.mimocode.bash import BashTool
from mimoagent.tools.mimocode.compact import CompactTool
from mimoagent.tools.mimocode.edit import EditTool
from mimoagent.tools.mimocode.glob import GlobTool
from mimoagent.tools.mimocode.grep import GrepTool
from mimoagent.tools.mimocode.read import ReadTool
from mimoagent.tools.mimocode.task import TaskTool
from mimoagent.tools.mimocode.write import WriteTool
from mimoagent.tools.registry import ToolRegistry

MIMOCODE_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "bash": BashTool,
    "read": ReadTool,
    "write": WriteTool,
    "edit": EditTool,
    "grep": GrepTool,
    "glob": GlobTool,
    "task": TaskTool,
    # Optional rollout controls. They are not in MimocodeAgentConfig.tools.
    "actor": ActorTool,
    "compact": CompactTool,
}

# MiMo default non-Codex core set. Actor and compact stay optional; there is no
# change_directory tool in the default set.
MIMOCODE_CORE_TOOL_NAMES = ("bash", "read", "write", "edit", "grep", "glob", "task")


class MimocodeToolRegistry(ToolRegistry):
    """Registry for MiMo's lowercase non-Codex tool ids."""

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "MimocodeToolRegistry":
        registry = cls()
        for tool_spec in config:
            tool_name = tool_spec.get("tool")
            if not tool_name:
                raise ToolException(f"Tool specification missing 'tool' field: {tool_spec}")
            normalized = str(tool_name).lower()
            tool_class = MIMOCODE_TOOL_CLASSES.get(normalized)
            if tool_class is None:
                raise ToolException(f"Unknown tool type: {tool_name}. MiMo catalogue: {sorted(MIMOCODE_TOOL_CLASSES)}")
            registry.register(tool_class(tool_spec.get("config", {})))
        return registry


__all__ = [
    "ActorTool",
    "BashTool",
    "CompactTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "MIMOCODE_CORE_TOOL_NAMES",
    "MIMOCODE_TOOL_CLASSES",
    "MimocodeToolRegistry",
    "ReadTool",
    "TaskTool",
    "WriteTool",
]
