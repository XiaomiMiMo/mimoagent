"""Write tool: create or overwrite a file with the given contents.

The contents travel via ``env.copy_to`` (tar stream) — never on a command
line — so arbitrarily large files work and no shell quoting is involved.
"""

import os
import tempfile
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput


class WriteTool(BaseTool):
    """Create or overwrite a file with the specified contents."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the read tool first to read the file's contents.
- Prefer the edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Parent directories are created as needed.
- `path` must be absolute (start with `/`)."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
                "file_text": {
                    "type": "string",
                    "description": "Complete contents to write to the file (overwrites if it exists)",
                },
            },
            "required": ["path", "file_text"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for write execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        path = params.get("path")
        if not path:
            raise ToolException("Missing required parameter: path")
        if not path.startswith("/"):
            raise ToolException(f"Path must be absolute (start with /), got: {path}")

        file_text = params.get("file_text")
        if file_text is None:
            raise ToolException("Missing required parameter: file_text")

        existed_before = (
            env.execute(f'[ -f "{path}" ] && echo "EXISTS" || echo "OK"').get("output", "").strip() == "EXISTS"
        )

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False, suffix=".txt") as f:
            f.write(file_text)
            local_path = f.name
        try:
            env.copy_to(local_path, path)
        except Exception as e:
            raise ToolException(f"Failed to write file to environment: {e}")
        finally:
            os.unlink(local_path)

        action = "overwritten" if existed_before else "created"
        verify_cmd = f'''
if [ -f "{path}" ]; then
    lines=$(wc -l < "{path}")
    echo "File {action} successfully: {path} ($lines lines)"
    echo "First few lines:"
    nl -ba "{path}" | head -10
else
    echo "Failed to write file"
fi
'''
        output = env.execute(verify_cmd).get("output", "").strip()
        success = "Failed to write file" not in output
        return ToolOutput(output=output, success=success)
