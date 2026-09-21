"""Helpers for the chat-completions message format used as the internal lingua franca.

Agents build messages in OpenAI chat-completions shape (MiMo dialect for
audio/video). ``content`` is either a plain string or a list of parts:

* ``{"type": "text", "text": ...}``
* ``{"type": "image_url", "image_url": {"url": <http(s) URL or data URL>}}``
* ``{"type": "input_audio", "input_audio": {"data": <URL / base64 / data URL>, "format": "wav"|...}}``
* ``{"type": "video_url", "video_url": {"url": <http(s) URL or data URL>}, "fps"?: ..., "media_resolution"?: ...}``

Each model translates this shape to its wire format; these helpers are the
shared parsing bits.
"""

from __future__ import annotations

from typing import Any

# part type -> media kind
_MEDIA_PART_KINDS = {
    "image_url": "image",
    "image": "image",
    "input_image": "image",
    "input_audio": "audio",
    "audio": "audio",
    "video_url": "video",
    "video": "video",
}


def media_part_kind(part: Any) -> str | None:
    """``"image"`` / ``"audio"`` / ``"video"`` for a media part, else None."""
    if isinstance(part, dict):
        return _MEDIA_PART_KINDS.get(part.get("type"))
    return None


def content_text(content: Any) -> str:
    """Concatenated text of a message content; media parts contribute nothing."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for piece in content:
            if isinstance(piece, dict):
                if media_part_kind(piece):
                    continue
                parts.append(piece.get("text") or piece.get("content") or "")
            else:
                parts.append(str(piece))
        return "".join(parts)
    return str(content)


def content_media_parts(content: Any) -> list[dict]:
    """Media parts (image/audio/video) of a message content, verbatim, in order."""
    if not isinstance(content, list):
        return []
    return [piece for piece in content if media_part_kind(piece)]


def content_image_urls(content: Any) -> list[str]:
    """Every image part's URL (http(s) or ``data:`` URL), in message order."""
    urls: list[str] = []
    if not isinstance(content, list):
        return urls
    for piece in content:
        if not isinstance(piece, dict) or piece.get("type") != "image_url":
            continue
        image_url = piece.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if url:
            urls.append(url)
    return urls


def split_data_url(url: str) -> tuple[str, str] | None:
    """``("image/png", "<base64>")`` for a data URL; ``None`` for anything else."""
    if not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    media_type = head[len("data:") :].split(";", 1)[0] or "image/png"
    return media_type, data
