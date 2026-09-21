"""MiMo-Code Read tool."""

import re
import shlex
from dataclasses import dataclass
from typing import Any

from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput
from mimoagent.tools.mimocode.paths import resolve_path
from mimoagent.tools.mimocode.prompts import load_tool_prompt


@dataclass
class ReadToolConfig(ToolConfig):
    max_view_lines: int = 2000  # MiMo's default line budget when `limit` is omitted
    max_line_length: int = 2000
    # MiMo's byte budget for one read (read.ts). Counted over the content lines
    # after per-line truncation, before line-number prefixes. It stops a
    # `limit` the model sets itself from returning a whole file, and it is the
    # only size cut: MimocodeAgentConfig disables the inherited char cut.
    max_output_bytes: int = 51200

    def __post_init__(self) -> None:
        if self.max_view_lines < 1 or self.max_line_length < 1 or self.max_output_bytes < 1:
            raise ValueError("Read max_view_lines, max_line_length and max_output_bytes must be positive")


class ReadTool(BaseTool):
    """View a file with line numbers, or list a directory as a shallow tree."""

    def _create_config(self, config_dict: dict[str, Any]) -> ReadToolConfig:
        return ReadToolConfig(**config_dict)

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return load_tool_prompt("read")

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file or directory",
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

        raw_file_path = params.get("file_path")
        if not raw_file_path:
            raise ToolException("Missing required parameter: file_path")
        file_path = resolve_path(str(raw_file_path), context)
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
            list_cmd = (
                f"printf '%s/\\n' {quoted_path}; "
                f"find {quoted_path} -maxdepth 1 -mindepth 1 -print | "
                "while IFS= read -r entry; do "
                'if [ -d "$entry" ]; then printf \'%s/\\n\' "${entry##*/}"; '
                "else printf '%s\\n' \"${entry##*/}\"; fi; "
                "done | LC_ALL=C sort"
            )
            output = env.execute(list_cmd).get("output", "").strip()
            entries = output.splitlines()[1:]
            try:
                offset = max(1, int(params.get("offset") or 1))
                limit = int(params.get("limit") or self.config.max_view_lines)
                if limit <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                return ToolOutput(
                    output="Error: offset must be an integer >= 1 and limit must be positive", success=False
                )
            selected = entries[offset - 1 : offset - 1 + limit]
            entry_count = len(entries)
            if not selected and offset > 1:
                # Mirror the file branch: an out-of-range offset is an error, not
                # an empty success. Reporting "(N entries)" with nothing listed
                # reads as a complete listing and the model keeps paging.
                return ToolOutput(
                    output=f"Error: Offset {offset} is out of range for this directory ({entry_count} entries)",
                    success=False,
                )
            truncated = offset - 1 + len(selected) < entry_count
            output = "\n".join(
                [
                    f"<path>{file_path}</path>",
                    "<type>directory</type>",
                    "<entries>",
                    *selected,
                    (
                        f"(Showing {len(selected)} of {entry_count} entries. Use offset={offset + len(selected)} to continue.)"
                        if truncated
                        else f"({entry_count} entries)"
                    ),
                    "</entries>",
                ]
            )
            return ToolOutput(output=output, success=True)

        if path_type == "FILE":
            readable_cmd = f'[ -r {quoted_path} ] && echo "OK" || echo "NO"'
            if env.execute(readable_cmd).get("output", "").strip() != "OK":
                return ToolOutput(output=f"Error: File not readable: {file_path}", success=False)

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
                cmd = f"tail -n +{offset} {quoted_path} | head -n {limit} | nl -ba -v{offset}"
            else:
                cmd = f"cat {quoted_path} | nl -ba | head -n {limit}"

            raw_output = env.execute(cmd).get("output", "")
            numbered_lines = []
            content_bytes = 0
            capped = False
            last_line = offset - 1  # advanced by nl's own numbering, so odd line separators can't skew it
            for line in raw_output.splitlines():
                match = re.match(r"^\s*(\d+)\t(.*)$", line)
                text = match.group(2) if match else line
                if len(text) > self.config.max_line_length:
                    text = (
                        text[: self.config.max_line_length]
                        + f"... (line truncated to {self.config.max_line_length} chars)"
                    )
                line_bytes = len(text.encode("utf-8")) + (1 if numbered_lines else 0)
                if content_bytes + line_bytes > self.config.max_output_bytes:
                    if numbered_lines:
                        capped = True
                        break
                    # A first line over the whole budget is cut at a UTF-8
                    # boundary instead of dropped, so the budget holds and
                    # "Use offset=2" still advances. Markers ride on top.
                    text = (
                        text.encode("utf-8")[: self.config.max_output_bytes].decode("utf-8", errors="ignore")
                        + f"... (line truncated to {self.config.max_output_bytes} bytes)"
                    )
                    line_bytes = len(text.encode("utf-8"))
                numbered_lines.append(f"{match.group(1)}: {text}" if match else text)
                content_bytes += line_bytes
                if match:
                    last_line = int(match.group(1))

            # Append truncation marker if the read does not cover the full file.
            # `wc -l` misses a final line without a newline; awk's NR matches
            # MiMo's readline-based line count for both forms.
            total_lines_cmd = f"awk 'END {{print NR}}' {quoted_path}"
            total_lines = int(env.execute(total_lines_cmd).get("output", "0").strip() or 0)
            if offset > total_lines and not (total_lines == 0 and offset == 1):
                return ToolOutput(
                    output=f"Error: Offset {offset} is out of range for this file ({total_lines} lines)",
                    success=False,
                )
            if capped:
                budget = self.config.max_output_bytes
                budget_label = f"{budget // 1024} KB" if budget % 1024 == 0 else f"{budget} bytes"
                continuation = (
                    f"(Output capped at {budget_label}. "
                    f"Showing lines {offset}-{last_line}. Use offset={last_line + 1} to continue.)"
                )
            elif total_lines > last_line:
                continuation = (
                    f"(Showing lines {offset}-{last_line} of {total_lines}. Use offset={last_line + 1} to continue.)"
                )
            else:
                continuation = f"(End of file - total {total_lines} lines)"
            output = "\n".join(
                [
                    f"<path>{file_path}</path>",
                    "<type>file</type>",
                    "<content>",
                    *numbered_lines,
                    "",
                    continuation,
                    "</content>",
                ]
            )
            state = (context or {}).get("state")
            if isinstance(state, dict):
                state.setdefault("read_files", set()).add(file_path)
            return ToolOutput(output=output, success=True)

        return ToolOutput(output="Error: cannot handle this path", success=False)
