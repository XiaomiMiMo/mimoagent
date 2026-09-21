"""Base tool implementation and utilities."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    """Standard output format for tool execution."""

    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Media to show the model alongside ``output``:
    # ``{"kind": "image"|"audio"|"video", "media_type": "image/png", "data": "<base64>"}``
    # (audio entries also carry ``"format"``, e.g. "wav"/"mp3"). The agent turns
    # them into multimodal content parts on the tool message; the model layer
    # converts to the wire format of its protocol.
    media: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ToolException(Exception):
    """The tool call itself could not be executed.

    Raise this only when the call is unusable: malformed/missing parameters,
    missing context (``env``/``model``), or a failure outside the requested
    operation (e.g. shipping inputs into the environment). When the call is
    well-formed but the operation fails in the environment — file not found,
    command timeout, non-unique match — return ``ToolOutput(success=False)``
    with an ``Error: ...`` message instead, so the outcome keeps its metadata
    and all failures look the same to the model.
    """


@dataclass
class ToolConfig:
    """Base configuration for tools. Subclasses extend with tool-specific fields."""


class BaseTool(ABC):
    """Abstract base class for all tools (function-calling mode).

    Subclasses must implement:
    - name / description properties
    - get_function_parameters() with JSON Schema
    - execute(params, context) returning a ToolOutput

    Optional:
    - _create_config() for a tool-specific ToolConfig dataclass
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = self._create_config(config or {})

    def _create_config(self, config_dict: dict[str, Any]) -> ToolConfig:
        """Create tool-specific config. Override in subclasses for custom configs."""
        return ToolConfig(**config_dict)

    @property
    @abstractmethod
    def name(self) -> str:
        """The name used to invoke this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Full description shown to the model (includes any constraints/notes)."""

    @abstractmethod
    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        """Execute the tool with given parameters.

        Args:
            params: Tool-specific parameters (typically a dict).
            context: Execution context (must contain ``env``).

        Returns:
            ToolOutput with the result.

        Raises:
            ToolException: If execution fails.
        """

    def get_function_parameters(self) -> dict[str, Any]:
        """JSON schema for function calling parameters."""
        return {"type": "object", "properties": {}, "required": []}

    def get_function_definition(self) -> dict[str, Any]:
        """Complete function definition in OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_function_parameters(),
            },
        }
