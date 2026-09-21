"""Read tool: text, directory listings, and opt-in image attachments."""

import base64
import binascii
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class ReadToolConfig(ToolConfig):
    max_view_lines: int = 500  # Default lines to show when `limit` is not provided
    max_depth: int = 2  # Maximum depth for directory listing
    truncate_message: str = "<response clipped>"
    # Match the default read tool's opt-in switch; CC currently supports images only.
    enable_media: bool = False
    max_image_bytes: int = 5 * 1024 * 1024


class ReadTool(BaseTool):
    """View a file with line numbers, or list a directory as a shallow tree."""

    def _create_config(self, config_dict: dict[str, Any]) -> ReadToolConfig:
        return ReadToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        description = """Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 500 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Results are returned using `cat -n`-like format, with line numbers starting at 1
- This tool can only read files, not directories. If `file_path` is a directory, the output is a shallow tree listing.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""
        if self.config.enable_media:
            description += "\n- Reads images (PNG, JPG, JPEG, GIF, WebP) and presents them visually."
        return description

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-indexed). Only provide if the file is too large to read at once.",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for read execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        file_path = params.get("file_path")
        if not file_path:
            raise ToolException("Missing required parameter: file_path")
        if not isinstance(file_path, str):
            raise ToolException("file_path must be a string")
        if not file_path.startswith("/"):
            raise ToolException(f"file_path must be absolute (start with /), got: {file_path}")

        quoted_path = shlex.quote(file_path)
        check_cmd = f"""
if [ -f {quoted_path} ]; then
    echo "FILE"
elif [ -d {quoted_path} ]; then
    echo "DIR"
else
    echo "NOTFOUND"
fi
        """
        path_type = env.execute(check_cmd).get("output", "").strip()

        if path_type == "NOTFOUND":
            return ToolOutput(output=f"Error: Path does not exist: {file_path}", success=False)

        if path_type == "DIR":
            list_cmd = rf'''
echo "{file_path}/"
find "{file_path}" -maxdepth {self.config.max_depth} -not -path "{file_path}" -not -path "*/.*" | sort | sed 's|{file_path}/||' | awk '{{
    depth = gsub(/\//, "/")
    for(i=0; i<depth; i++) printf("  ")
    split($0, parts, "/")
    print parts[length(parts)]
}}'
            '''
            output = env.execute(list_cmd).get("output", "").strip()
            return ToolOutput(output=output, success=True)

        if path_type == "FILE":
            readable_cmd = f'[ -r {quoted_path} ] && echo "OK" || echo "NO"'
            if env.execute(readable_cmd).get("output", "").strip() != "OK":
                return ToolOutput(output=f"Error: File not readable: {file_path}", success=False)

            media_type = _IMAGE_TYPES.get(PurePosixPath(file_path).suffix.lower())
            if self.config.enable_media and media_type:
                return self._read_image(env, file_path, media_type)

            raw_offset = params.get("offset")
            raw_limit = params.get("limit")

            try:
                offset = int(raw_offset) if raw_offset is not None else 1
                if offset < 1:
                    offset = 1
                limit = int(raw_limit) if raw_limit is not None else self.config.max_view_lines
                if limit <= 0:
                    raise ValueError("limit must be positive")
            except (ValueError, TypeError):
                return ToolOutput(
                    output="Error: offset must be an integer >= 1 and limit must be a positive integer",
                    success=False,
                )

            partial_view = raw_offset is not None or raw_limit is not None

            if partial_view:
                # Use 1-indexed offset + line count via tail+head, numbered with nl -ba starting at offset.
                cmd = f'tail -n +{offset} "{file_path}" | head -n {limit} | nl -ba -v{offset}'
            else:
                cmd = f'cat "{file_path}" | nl -ba | head -n {limit}'

            output = env.execute(cmd).get("output", "")

            # Append truncation marker if the read does not cover the full file.
            total_lines_cmd = f'wc -l < "{file_path}"'
            total_lines = int(env.execute(total_lines_cmd).get("output", "0").strip() or 0)
            last_line_shown = offset + limit - 1 if partial_view else limit
            if total_lines > last_line_shown:
                output += f"\n{self.config.truncate_message}"

            return ToolOutput(output=output, success=True)

        return ToolOutput(output="Error: cannot handle this path", success=False)

    def _read_image(self, env: Any, file_path: str, media_type: str) -> ToolOutput:
        quoted_path = shlex.quote(file_path)
        stat = env.execute(f"wc -c < {quoted_path}")
        try:
            size = int(stat.get("output", "").strip())
        except (TypeError, ValueError):
            return ToolOutput(output=f"Error: could not stat image file {file_path}", success=False)
        if stat.get("returncode", 0) or size <= 0:
            return ToolOutput(output=f"Error: image file is empty or unreadable: {file_path}", success=False)
        if size > self.config.max_image_bytes:
            return ToolOutput(
                output=(
                    f"Error: image file {file_path} is {size} bytes, over the "
                    f"{self.config.max_image_bytes} byte limit. Downscale or convert it first."
                ),
                success=False,
            )

        # Bound the read even if the file grows after stat. Plain base64 works on BusyBox too.
        encoded = env.execute(f"head -c {self.config.max_image_bytes + 1} {quoted_path} | base64")
        b64 = "".join(encoded.get("output", "").split())
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return ToolOutput(output=f"Error: could not read image file {file_path}", success=False)
        if encoded.get("returncode", 0) or len(data) != size:
            return ToolOutput(
                output=f"Error: image file changed or could not be read completely: {file_path}. Retry reading it.",
                success=False,
            )
        return ToolOutput(
            output=f"Read image file {file_path} ({media_type}, {size} bytes). The image is attached.",
            media=[{"kind": "image", "media_type": media_type, "data": b64}],
        )
