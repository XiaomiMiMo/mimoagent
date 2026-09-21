"""Claude-Code-aligned tool catalogue.

This package holds the CC-aligned tool implementations — capitalized names,
millisecond Bash timeouts, ``file_path``/``offset``/``limit`` Read,
``old_string``/``new_string``/``replace_all`` Edit — plus the CC-only
Grep/Glob/Compact tools.

It coexists with the original tool layer (``mimoagent.tools``): nothing here
touches the default ``ToolRegistry`` catalogue. The CC catalogue is only used
by :class:`mimoagent.agents.cc.cc_agent.CCAgent` (``agent.type: cc-agent`` in
yaml); runs that don't opt in are byte-for-byte unaffected.
"""

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException
from mimoagent.tools.cc.agent import AgentTool
from mimoagent.tools.cc.bash import BashTool
from mimoagent.tools.cc.compact import CompactTool
from mimoagent.tools.cc.edit import EditTool
from mimoagent.tools.cc.glob import GlobTool
from mimoagent.tools.cc.grep import GrepTool
from mimoagent.tools.cc.read import ReadTool
from mimoagent.tools.cc.write import WriteTool
from mimoagent.tools.registry import ToolRegistry

CC_TOOL_CLASSES: dict[str, type[BaseTool]] = {
    "Bash": BashTool,
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Agent": AgentTool,
    "Grep": GrepTool,
    "Glob": GlobTool,
    "Compact": CompactTool,
}


class CCToolRegistry(ToolRegistry):
    """ToolRegistry over the CC-aligned catalogue.

    Tool-name lookup is case-insensitive so a yaml written for the original
    catalogue (``bash``/``read``) resolves here without edits — including the
    subagent presets in ``agents/subagents.py``, which name tools in lowercase.
    """

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "CCToolRegistry":
        by_lower = {name.lower(): tool_class for name, tool_class in CC_TOOL_CLASSES.items()}
        registry = cls()
        for tool_spec in config:
            tool_name = tool_spec.get("tool")
            if not tool_name:
                raise ToolException(f"Tool specification missing 'tool' field: {tool_spec}")
            tool_class = by_lower.get(tool_name.lower())
            if tool_class is None:
                raise ToolException(f"Unknown tool type: {tool_name}")
            registry.register(tool_class(tool_spec.get("config", {})))
        return registry


__all__ = [
    "AgentTool",
    "BashTool",
    "CC_TOOL_CLASSES",
    "CCToolRegistry",
    "CompactTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "WriteTool",
]
