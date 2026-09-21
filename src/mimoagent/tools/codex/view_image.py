"""Codex ``view_image``: load a local image as multimodal tool output."""

from __future__ import annotations

import base64
import binascii
import shlex
from pathlib import Path
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
)


def _image_media_type(data: bytes) -> str | None:
    for signature, media_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return media_type
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class ViewImageTool(BaseTool):
    """View a local image file and attach it to the model response."""

    @property
    def name(self) -> str:
        return "view_image"

    @property
    def description(self) -> str:
        return (
            "View a local image file from the filesystem when visual inspection is needed. "
            "Use this for images already available on disk."
        )

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local filesystem path to an image file."},
                "detail": {
                    "type": "string",
                    "enum": ["high", "original"],
                    "description": "Image detail level. Defaults to `high`; use `original` to preserve exact resolution.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for view_image execution")
        if not isinstance(params, dict):
            raise ToolException(f"view_image params must be a dictionary, got {type(params).__name__}")
        unsupported = set(params) - {"path", "detail"}
        if unsupported:
            raise ToolException(f"view_image has unsupported field(s): {', '.join(sorted(unsupported))}")

        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise ToolException("view_image requires a non-empty 'path' parameter")
        detail = params.get("detail")
        if detail is not None and detail not in {"high", "original"}:
            raise ToolException(
                "view_image.detail only supports `high` or `original`; "
                f"omit `detail` for default high resized behavior, got `{detail}`"
            )

        env_cwd = getattr(getattr(env, "config", None), "cwd", "") or ""
        resolved = str(Path(env_cwd) / path) if env_cwd and not Path(path).is_absolute() else path
        quoted = shlex.quote(resolved)
        probe = env.execute(
            f"if [ -d {quoted} ]; then echo DIR; elif [ ! -f {quoted} ]; then echo NOTFOUND; "
            f"elif [ ! -r {quoted} ]; then echo NOREAD; else wc -c < {quoted}; fi",
            cwd=env_cwd,
        )
        probe_output = str(probe.get("output", "")).strip()
        if probe_output == "NOTFOUND":
            return ToolOutput(output=f"Error: unable to locate image at `{path}`", success=False)
        if probe_output == "DIR":
            return ToolOutput(output=f"Error: image path `{path}` is not a file", success=False)
        if probe_output == "NOREAD":
            return ToolOutput(output=f"Error: unable to read image at `{path}`", success=False)
        try:
            size = int(probe_output.split()[-1])
        except (ValueError, IndexError):
            return ToolOutput(output=f"Error: unable to locate image at `{path}`: {probe_output}", success=False)

        # Keep inline multimodal payloads bounded, matching the regular read tool.
        if size > 5 * 1024 * 1024:
            return ToolOutput(output=f"Error: image at `{path}` exceeds the 5242880 byte limit", success=False)
        encoded = env.execute(f"base64 {quoted} | tr -d '\\n'", cwd=env_cwd)
        if encoded.get("returncode", 0) not in (0, None):
            return ToolOutput(output=f"Error: unable to read image at `{path}`", success=False)
        try:
            data = base64.b64decode(str(encoded.get("output", "")).strip(), validate=True)
        except (binascii.Error, ValueError):
            return ToolOutput(output=f"Error: unable to read image at `{path}`", success=False)
        media_type = _image_media_type(data)
        if media_type is None:
            return ToolOutput(output="Error: unable to process image: invalid or unsupported image data", success=False)

        selected_detail = detail or "high"
        return ToolOutput(
            output=f"Viewed image `{path}` ({media_type}). The image is attached.",
            media=[
                {
                    "kind": "image",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                    "detail": selected_detail,
                }
            ],
        )
