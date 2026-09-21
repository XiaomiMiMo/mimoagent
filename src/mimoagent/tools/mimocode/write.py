"""MiMo-Code Write tool."""

import os
import shlex
import tempfile
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.mimocode.paths import resolve_path
from mimoagent.tools.mimocode.prompts import load_tool_prompt


class WriteTool(BaseTool):
    """Create or overwrite a file with the specified contents."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return load_tool_prompt("write")

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for write execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        raw_file_path = params.get("file_path")
        if not raw_file_path:
            raise ToolException("Missing required parameter: file_path")
        file_path = resolve_path(str(raw_file_path), context)
        quoted_path = shlex.quote(file_path)

        content = params.get("content")
        if content is None:
            raise ToolException("Missing required parameter: content")

        existed_before = (
            env.execute(f"[ -f {quoted_path} ] && echo 'EXISTS' || echo 'OK'").get("output", "").strip() == "EXISTS"
        )
        state = (context or {}).get("state")
        if existed_before and (not isinstance(state, dict) or file_path not in state.get("read_files", set())):
            return ToolOutput(
                output=f"Error: You must read {file_path} before overwriting it.",
                success=False,
            )

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False, suffix=".txt") as f:
            f.write(content)
            local_path = f.name
        try:
            env.copy_to(local_path, file_path)
        except Exception as e:
            raise ToolException(f"Failed to write file to environment: {e}")
        finally:
            os.unlink(local_path)

        action = "overwritten" if existed_before else "created"
        verify_cmd = f"[ -f {quoted_path} ] && printf 'OK' || printf 'MISSING'"
        output = env.execute(verify_cmd).get("output", "").strip()
        if output != "OK":
            return ToolOutput(output="Error: Failed to write file", success=False)
        # The model just supplied the file's entire contents, so it has "read"
        # it for the purposes of the edit gate; without this a write followed by
        # an edit of the same file is refused.
        if isinstance(state, dict):
            state.setdefault("read_files", set()).add(file_path)
        return ToolOutput(
            output="Wrote file successfully.",
            metadata={"filepath": file_path, "exists": existed_before, "action": action},
        )
