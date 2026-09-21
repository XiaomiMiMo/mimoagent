"""Read tool: view a file's contents (text; media formats opt-in via ``enable_media``)."""

from dataclasses import dataclass
from typing import Any

from mimoagent.tools.base import BaseTool, ToolConfig, ToolException, ToolOutput

# Media formats the model APIs accept; other binary formats stay rejected.
# suffix -> (kind, media_type). Audio entries also carry the ``format`` string
# the chat protocol's input_audio part wants.
_MEDIA_TYPES: dict[str, tuple[str, str]] = {
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"),
    ".wav": ("audio", "audio/wav"),
    ".mp3": ("audio", "audio/mpeg"),
    ".m4a": ("audio", "audio/mp4"),
    ".flac": ("audio", "audio/flac"),
    ".ogg": ("audio", "audio/ogg"),
    ".mp4": ("video", "video/mp4"),
    ".webm": ("video", "video/webm"),
    ".mov": ("video", "video/quicktime"),
}

_AUDIO_FORMATS = {".wav": "wav", ".mp3": "mp3", ".m4a": "m4a", ".flac": "flac", ".ogg": "ogg"}


@dataclass
class ReadToolConfig(ToolConfig):
    # Whether media files (image/audio/video) are returned as multimodal
    # content. Off by default: media suffixes go through the regular binary
    # sniff and are rejected, and the tool description does not mention media.
    enable_media: bool = False
    # Hard cap on image file size (raw bytes, before base64). Model APIs cap
    # around 5MB per image; larger files error out with a hint to downscale.
    max_image_bytes: int = 5 * 1024 * 1024
    # Hard cap on audio/video file size (raw bytes, before base64) — these ride
    # inline as base64 in the request body, so keep them bounded.
    max_media_bytes: int = 20 * 1024 * 1024


class ReadTool(BaseTool):
    """View a file with line numbers."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        if self.config.enable_media:
            path_lines = """- `path` must be an absolute path (start with `/`). Directories are not supported — use bash (`ls`, `find`) to explore directory structure.
- Media files are returned as content you can view/listen to/watch directly: images (png/jpg/jpeg/gif/webp), audio (wav/mp3/m4a/flac/ogg), video (mp4/webm/mov). Other binary files are rejected — use bash (`file`, `xxd`, `strings`) to inspect them."""
        else:
            path_lines = "- `path` must be an absolute path (start with `/`) to a text file. Directories are not supported — use bash (`ls`, `find`) to explore directory structure. Binary files are rejected — use bash (`file`, `xxd`, `strings`) to inspect them."
        return f"""View the contents of a file.

- The output shows the file contents with 1-indexed line numbers prefixed to each line.
- Very long outputs are truncated in the middle (a notice marks the cut); use `view_range` to read a large file in specific ranges instead.
{path_lines}"""

    def _create_config(self, config_dict: dict[str, Any]) -> ReadToolConfig:
        return ReadToolConfig(**config_dict)

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional line range as [start_line, end_line] (1-indexed); use -1 as the end to read to EOF. Only provide if the file is too large to read at once.",
                },
            },
            "required": ["path"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for read execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        path = params.get("path")
        if not path:
            raise ToolException("Missing required parameter: path")
        if not path.startswith("/"):
            raise ToolException(f"Path must be absolute (start with /), got: {path}")

        filename = path.rsplit("/", 1)[-1]
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if self.config.enable_media and (media := _MEDIA_TYPES.get(suffix)):
            kind, media_type = media
            return self._read_media(env, path, kind, media_type, suffix)

        # One probe round-trip: type, readability, and a binary sniff (NUL in
        # the first 8KB, git-style heuristic — full-file scan would be O(n) on
        # files we might only view a range of).
        check_cmd = f'''
if [ -d "{path}" ]; then
    echo "DIR"
elif [ ! -f "{path}" ]; then
    echo "NOTFOUND"
elif [ ! -r "{path}" ]; then
    echo "NOREAD"
elif [ "$(head -c 8192 "{path}" | tr -d '\\0' | wc -c)" -ne "$(head -c 8192 "{path}" | wc -c)" ]; then
    echo "BINARY"
else
    echo "FILE"
fi
        '''
        path_type = env.execute(check_cmd).get("output", "").strip()

        if path_type == "NOTFOUND":
            return ToolOutput(output=f"Error: Path does not exist: {path}", success=False)

        if path_type == "DIR":
            return ToolOutput(
                output=f"Error: {path} is a directory. The read tool only reads files; use bash (`ls`, `find`) to explore directories.",
                success=False,
            )

        if path_type == "NOREAD":
            return ToolOutput(output=f"Error: File not readable: {path}", success=False)

        if path_type == "BINARY":
            return ToolOutput(
                output=f"Error: {path} appears to be a binary file (contains NUL bytes); read only supports text files. Use bash tools (`file`, `xxd`, `strings`) to inspect binary content.",
                success=False,
            )

        if path_type == "FILE":
            # No tool-side truncation: full output is returned and the agent's
            # unified per-response truncation (utils.truncate) clips it if
            # needed, with an explicit notice the model can react to.
            raw_view_range = params.get("view_range")

            if raw_view_range is not None:
                try:
                    if isinstance(raw_view_range, str):
                        import ast

                        view_range = ast.literal_eval(raw_view_range)
                    elif isinstance(raw_view_range, list | tuple):
                        view_range = list(raw_view_range)
                    else:
                        raise ValueError("Invalid range format")

                    if isinstance(view_range, list) and len(view_range) == 2:
                        start_line = max(1, int(view_range[0]))
                        end_line = int(view_range[1])

                        if end_line == -1:
                            cmd = f'tail -n +{start_line} "{path}" | nl -ba -v{start_line}'
                        else:
                            if end_line < start_line:
                                start_line, end_line = end_line, start_line
                            cmd = f'sed -n "{start_line},{end_line}p" "{path}" | nl -ba -v{start_line}'
                    else:
                        raise ValueError("Invalid range format")
                except (ValueError, SyntaxError, TypeError):
                    return ToolOutput(
                        output="Error: view_range must be a list like [start_line, end_line] where both are integers",
                        success=False,
                    )
            else:
                cmd = f'nl -ba "{path}"'

            output = env.execute(cmd).get("output", "")
            return ToolOutput(output=output, success=True)

        return ToolOutput(output="Error: cannot handle this path", success=False)

    def _read_media(self, env: Any, path: str, kind: str, media_type: str, suffix: str) -> ToolOutput:
        """Return the file as base64 media (image/audio/video) the model can consume directly."""
        probe = (
            env.execute(
                f'if [ -d "{path}" ]; then echo "DIR"; '
                f'elif [ ! -f "{path}" ]; then echo "NOTFOUND"; '
                f'elif [ ! -r "{path}" ]; then echo "NOREAD"; '
                f'else wc -c < "{path}"; fi'
            )
            .get("output", "")
            .strip()
        )
        if probe == "NOTFOUND":
            return ToolOutput(output=f"Error: Path does not exist: {path}", success=False)
        if probe == "DIR":
            return ToolOutput(
                output=f"Error: {path} is a directory. The read tool only reads files; use bash (`ls`, `find`) to explore directories.",
                success=False,
            )
        if probe == "NOREAD":
            return ToolOutput(output=f"Error: File not readable: {path}", success=False)
        try:
            size = int(probe.split()[-1])
        except (ValueError, IndexError):
            return ToolOutput(output=f"Error: could not stat {kind} file {path}: {probe}", success=False)
        max_bytes = self.config.max_image_bytes if kind == "image" else self.config.max_media_bytes
        if size > max_bytes:
            return ToolOutput(
                output=(
                    f"Error: {kind} file {path} is {size} bytes, over the {max_bytes} byte limit. "
                    "Downscale/trim or convert it first (e.g. with ImageMagick/ffmpeg)."
                ),
                success=False,
            )

        # busybox base64 has no ``-w0``; strip the wrapping newlines instead.
        b64 = env.execute(f'base64 "{path}" | tr -d "\\n"').get("output", "").strip()
        if not b64:
            return ToolOutput(output=f"Error: could not base64-encode {kind} file {path}", success=False)
        entry = {"kind": kind, "media_type": media_type, "data": b64}
        if kind == "audio":
            entry["format"] = _AUDIO_FORMATS.get(suffix, suffix.lstrip("."))
        return ToolOutput(
            output=f"Read {kind} file {path} ({media_type}, {max(size, 1024) // 1024} KB). The {kind} is attached.",
            media=[entry],
        )
