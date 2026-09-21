"""Codex code-mode tools backed by the standalone V8 host.

``exec`` evaluates raw JavaScript in ``codex-code-mode-host`` and exposes the
configured Codex catalogue through the isolate's global ``tools`` object.
``wait`` resumes or terminates cells that yielded before completing.  One
runtime is owned by one native Codex agent so ``store``/``load`` state persists
across its cells while remaining isolated from other agents.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.codex.code_mode_host import (
    CodeModeHostClosedError,
    CodeModeHostError,
    CodeModeHostProtocolError,
    CodeModeHostSession,
    build_tool_definition,
    find_code_mode_host,
)
from mimoagent.tools.codex.output_budget import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    formatted_truncate_text,
    resolve_max_tokens,
    truncate_exec_output,
)

PUBLIC_TOOL_NAME = "exec"
WAIT_TOOL_NAME = "wait"
DEFAULT_EXEC_YIELD_TIME_MS = 30_000
DEFAULT_WAIT_YIELD_TIME_MS = 10_000

_PRAGMA_PREFIX = "// @exec:"
_MAX_SAFE_INTEGER = (1 << 53) - 1
_GRAMMAR = r"""
start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"""

_EXEC_DESCRIPTION_TEMPLATE = """Run JavaScript code to orchestrate/compose tool calls
- Evaluates the provided JavaScript code in a fresh V8 isolate as an async module.
- All nested tools are available on the global `tools` object, for example `await tools.exec_command(...)`.
- Nested tool methods take a JSON object as their argument (`apply_patch` takes the raw patch string) and resolve to the value declared in their signature below.
- Runs raw JavaScript -- no Node, no file system, no network access, no console.
- Accepts raw JavaScript source text, not JSON, quoted strings, or markdown code fences.
- You may optionally start the tool input with `// @exec: {{"yield_time_ms": 10000, "max_output_tokens": 1000}}`.
- `yield_time_ms` (default {yield_time_ms}): how long the cell may run before its output so far is returned as `Script running with cell ID ...`; the cell can then be resumed with `wait`.
- `max_output_tokens` sets the token budget for direct `exec` results. Defaults to {max_output_tokens} tokens.
- When the JavaScript finishes, unawaited promises are discarded.

Global helpers:
- `exit()`: ends the current script successfully.
- `text(value)`: appends text output; other values are JSON-stringified when possible.
- `image(value, detail)`: appends a base64 data URL or an image content item.
- `audio(value)`: appends a base64 data URL or an audio content item.
- `generatedImage(result)`: appends a generated image result and its optional output hint.
- `store(key, value)` and `load(key)`: persist serializable values across exec calls in this session.
- `notify(value)`: immediately emits an extra output item for this exec call.
- `setTimeout(callback, delayMs)` and `clearTimeout(id)`: timer helpers.
- `ALL_TOOLS`: metadata for all enabled nested tools.
- `yield_control()`: yields accumulated output while the script continues running.
"""

# What each nested tool resolves to, as ``_nested_result`` shapes it. Declared
# alongside the argument type so the model does not have to guess whether the
# report is a string or an object with an ``output`` field (it guessed wrong
# either way while the declaration said ``Promise<unknown>``).
_NESTED_RETURN_TYPES = {
    "exec_command": "{ output: string; exit_code?: number; wall_time_seconds?: number; timed_out?: boolean; original_token_count?: number }",
    "apply_patch": "string",
    "update_plan": "string",
    "view_image": "{ image_url: string; detail: string }",
}

# These errors are only recorded when they escape the JavaScript cell. A
# ``try/catch`` is an intentional recovery path for code-mode callers. The
# host cannot expose the resolved name for a computed property access.
# The lookbehind matches the global object rather than any identifier ending in
# ``tools``: `\b` still fires after a dot, so `ctx.tools.foo` would read as the
# nested surface. A local binding *named* ``tools`` stays indistinguishable —
# the message is byte-identical and only the exec source could separate them.
_JS_UNKNOWN_TOOL_RE = re.compile(r"(?<![\w.$])tools(?:\.([A-Za-z_$][\w$]*)|\[[^\]]+\]) is not a function")
_JS_BAD_ARGS_RE = re.compile(r"tool `(\w+)` expects a (?:JSON object for arguments|string input)")


@dataclass(frozen=True)
class ParsedExecSource:
    code: str
    yield_time_ms: int | None
    max_output_tokens: int | None


def parse_exec_source(source: str) -> ParsedExecSource:
    """Parse the optional first-line pragma using the code-rs contract."""

    if not isinstance(source, str) or not source.strip():
        raise ToolException("exec expects non-empty raw JavaScript source text")
    first_line, separator, rest = source.partition("\n")
    stripped = first_line.lstrip()
    if not stripped.startswith(_PRAGMA_PREFIX):
        return ParsedExecSource(source, None, None)
    if not separator or not rest.strip():
        raise ToolException("exec pragma must be followed by JavaScript source")
    directive = stripped.removeprefix(_PRAGMA_PREFIX).strip()
    try:
        value = json.loads(directive)
    except json.JSONDecodeError as exc:
        raise ToolException(f"exec pragma must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ToolException("exec pragma must be a JSON object")
    unsupported = set(value) - {"yield_time_ms", "max_output_tokens"}
    if unsupported:
        raise ToolException(f"exec pragma has unsupported field(s): {', '.join(sorted(unsupported))}")
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= _MAX_SAFE_INTEGER:
            raise ToolException(f"exec pragma field '{key}' must be a non-negative safe integer")
    return ParsedExecSource(rest, value.get("yield_time_ms"), value.get("max_output_tokens"))


def _function_definition(tool: BaseTool) -> dict[str, Any]:
    return build_tool_definition(
        tool.name,
        description=tool.description,
        kind="function",
        input_schema=tool.get_function_parameters(),
    )


def _freeform_definition(tool: BaseTool) -> dict[str, Any]:
    return build_tool_definition(tool.name, description=tool.description, kind="freeform")


def _schema_to_ts(schema: Any, indent: str = "") -> str:
    """Render a JSON-schema fragment as a TypeScript type.

    Mirrors code-rs ``render_json_schema_to_typescript`` closely enough for the
    Codex catalogue's schemas: objects, arrays, enums, primitives. Anything it
    does not recognise degrades to ``unknown`` rather than guessing. Deriving
    the type from the live schema is the point — a hand-written signature drifts
    from the tool the moment a parameter changes (which is how the stale
    ``max_output_tokens`` line first slipped in).

    Property descriptions become doc comments. Under ptc=true the declaration
    is the only place the nested tools' parameter docs can reach the model:
    dropping them hid, among other things, the ``exec_command`` output budget
    default, so a model raising ``max_output_tokens`` there had no way to know
    what it was raising it from.
    """

    if not isinstance(schema, dict):
        return "unknown"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return " | ".join(json.dumps(value) for value in enum)
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return "Record<string, unknown>"
        required = set(schema.get("required") or [])
        inner = indent + "  "
        lines = []
        for name, prop in properties.items():
            description = prop.get("description") if isinstance(prop, dict) else None
            if isinstance(description, str) and description.strip():
                lines.append(f"{inner}/** {' '.join(description.split()).replace('*/', '* /')} */")
            optional = "" if name in required else "?"
            lines.append(f"{inner}{name}{optional}: {_schema_to_ts(prop, inner)};")
        return "{\n" + "\n".join(lines) + f"\n{indent}}}"
    if schema_type == "array":
        return f"Array<{_schema_to_ts(schema.get('items'), indent)}>"
    if schema_type in ("integer", "number"):
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "string":
        return "string"
    return "unknown"


def _nested_tool_section(tool: BaseTool) -> str:
    """Render one nested tool as description + a TypeScript declaration.

    Follows code-rs ``render_code_mode_sample_for_definition``: the tool's own
    description comes first, then the call signature. Under ptc=true this ``exec``
    description is the only place the model learns what the nested tools do, so
    the description — apply_patch's V4A grammar in particular — must travel with
    the signature rather than being dropped.
    """

    freeform = tool.name == "apply_patch"
    input_name = "input" if freeform else "args"
    input_type = "string" if freeform else _schema_to_ts(tool.get_function_parameters(), "  ")
    returns = _NESTED_RETURN_TYPES.get(tool.name, "unknown")
    declaration = f"  {tool.name}({input_name}: {input_type}): Promise<{returns}>;"
    return (
        f"### `{tool.name}`\n{tool.description}\n\n"
        f"exec tool declaration:\n```ts\ndeclare const tools: {{\n{declaration}\n}};\n```"
    )


def _exec_description(
    tools: list[BaseTool],
    *,
    yield_time_ms: int = DEFAULT_EXEC_YIELD_TIME_MS,
) -> str:
    intro = _EXEC_DESCRIPTION_TEMPLATE.format(
        yield_time_ms=yield_time_ms,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    if not tools:
        return intro
    sections = "\n\n".join(_nested_tool_section(tool) for tool in tools)
    return f"{intro}\n\nEnabled nested tools:\n\n{sections}"


def build_exec_tool_definition(
    *,
    nested_tools: list[BaseTool] | None = None,
    yield_time_ms: int = DEFAULT_EXEC_YIELD_TIME_MS,
) -> dict[str, Any]:
    """Return the Responses freeform ``exec`` tool with configured defaults."""

    return {
        "type": "custom",
        "name": PUBLIC_TOOL_NAME,
        "description": _exec_description(nested_tools or [], yield_time_ms=yield_time_ms),
        "format": {"type": "grammar", "syntax": "lark", "definition": _GRAMMAR},
    }


def _media_entry(item: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    """Convert one media content item, or explain why it cannot be sent.

    Returns ``(entry, None)`` for a usable item, ``(None, reason)`` for a
    media-shaped item we refuse, and ``(None, None)`` for anything that is not
    media. An unusable payload used to be forwarded verbatim, so the gateway
    rejected the whole request and the step became a ModelQueryError instead of
    a tool error the cell could react to. ``view_image`` already validates the
    same way.
    """

    item_type = item.get("type")
    if item_type == "input_image":
        kind = "image"
        url = item.get("image_url")
    elif item_type == "input_audio":
        kind = "audio"
        url = item.get("audio_url")
    else:
        return None, None
    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        return None, f"{kind} dropped: expected a base64 `data:` URL"
    header, data = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    if ";base64" not in header or not media_type or not data:
        return None, f"{kind} dropped: expected a base64 `data:` URL with a media type"
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, f"{kind} dropped: the base64 payload could not be decoded"
    entry = {"kind": kind, "media_type": media_type, "data": data}
    if kind == "image" and isinstance(item.get("detail"), str):
        entry["detail"] = item["detail"]
    if kind == "audio":
        entry["format"] = media_type.rsplit("/", 1)[-1]
    return entry, None


class CodeModeRuntime:
    """Lazy, durable bridge between a Codex agent and its code-mode host."""

    def __init__(
        self,
        agent: Any,
        *,
        host_path: str | None,
        exec_yield_time_ms: int = DEFAULT_EXEC_YIELD_TIME_MS,
        wait_yield_time_ms: int = DEFAULT_WAIT_YIELD_TIME_MS,
        request_timeout: float = 60.0,
        delegate_workers: int = 8,
        fatal_delegate_errors: tuple[type[BaseException], ...] = (TransportError,),
    ):
        self._agent_ref = weakref.ref(agent)
        self.host_path = host_path
        self.exec_yield_time_ms = exec_yield_time_ms
        self.wait_yield_time_ms = wait_yield_time_ms
        # No output ceiling lives here: the cell budget is the pragma's (default
        # DEFAULT_MAX_OUTPUT_TOKENS, uncapped, as codex-rs resolve_max_tokens),
        # and the hard bound is the agent's history-layer cut.
        self.request_timeout = request_timeout
        self.delegate_workers = delegate_workers
        self.fatal_delegate_errors = fatal_delegate_errors
        self._session: CodeModeHostSession | None = None
        self._session_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._closed_event = threading.Event()

    @property
    def nested_tools(self) -> list[BaseTool]:
        return list(self.agent.tool_registry.tools.values())

    @property
    def agent(self) -> Any:
        agent = self._agent_ref()
        if agent is None:
            raise ToolException("the owning Codex agent is no longer available")
        return agent

    def model_definitions(self) -> list[dict[str, Any]]:
        """The complete model-visible surface: ``exec`` plus ``wait``.

        Nested tools are reachable only from JavaScript, so they are never
        advertised as separate function tools.
        """

        return [
            build_exec_tool_definition(
                nested_tools=self.nested_tools,
                yield_time_ms=self.exec_yield_time_ms,
            ),
            WaitTool.definition(yield_time_ms=self.wait_yield_time_ms),
        ]

    def execute(self, tool_call_id: str, source: str) -> ToolOutput:
        parsed = parse_exec_source(source)
        budget = resolve_max_tokens(parsed.max_output_tokens)
        started = time.monotonic()
        session = self._get_session()
        try:
            response = session.execute(
                tool_call_id,
                parsed.code,
                parsed.yield_time_ms if parsed.yield_time_ms is not None else self.exec_yield_time_ms,
                max_output_tokens=budget,
            )
        except self.fatal_delegate_errors:
            raise
        except (CodeModeHostClosedError, CodeModeHostProtocolError) as exc:
            self._discard_failed_session(session)
            raise ToolException(f"Code mode host failed: {exc}") from exc
        except (CodeModeHostError, FileNotFoundError, OSError) as exc:
            raise ToolException(f"Code mode host failed: {exc}") from exc
        return self._format_response(response, budget, started)

    def wait(
        self,
        cell_id: str,
        *,
        yield_time_ms: int | None,
        max_tokens: int | None,
        terminate: bool,
    ) -> ToolOutput:
        started = time.monotonic()
        session = self._get_session()
        try:
            response = session.wait(
                cell_id,
                self.wait_yield_time_ms if yield_time_ms is None else yield_time_ms,
                terminate=terminate,
            )
        except self.fatal_delegate_errors:
            raise
        except (CodeModeHostClosedError, CodeModeHostProtocolError) as exc:
            self._discard_failed_session(session)
            raise ToolException(f"Code mode host failed: {exc}") from exc
        except (CodeModeHostError, FileNotFoundError, OSError) as exc:
            raise ToolException(f"Code mode host failed: {exc}") from exc
        return self._format_response(response, resolve_max_tokens(max_tokens), started)

    def close(self) -> None:
        with self._session_lock:
            if self._closed:
                return
            wait_for_owner = self._closing
            if not wait_for_owner:
                self._closing = True
                session, self._session = self._session, None
            else:
                session = None
        if wait_for_owner:
            self._closed_event.wait()
            return
        try:
            if session is not None:
                session.close()
        finally:
            with self._session_lock:
                self._closed = True
                self._closed_event.set()

    def _get_session(self) -> CodeModeHostSession:
        with self._session_lock:
            if self._closing or self._closed:
                raise ToolException("the code-mode runtime is closed")
            if self._session is None:
                path = find_code_mode_host(self.host_path)
                definitions = [
                    _freeform_definition(tool) if tool.name == "apply_patch" else _function_definition(tool)
                    for tool in self.nested_tools
                ]
                runtime_ref = weakref.ref(self)

                def invoke_tool(name: str, raw_input: Any, kind: str) -> Any:
                    runtime = runtime_ref()
                    if runtime is None:
                        raise ToolException("the code-mode runtime was closed")
                    return runtime._invoke_tool(name, raw_input, kind)

                def notify(call_id: str, cell_id: str, text: str) -> None:
                    runtime = runtime_ref()
                    if runtime is not None:
                        runtime._notify(call_id, cell_id, text)

                self._session = CodeModeHostSession(
                    path,
                    invoke_tool,
                    enabled_tools=definitions,
                    notify=notify,
                    request_timeout=self.request_timeout,
                    delegate_workers=self.delegate_workers,
                    fatal_delegate_errors=self.fatal_delegate_errors,
                )
            return self._session

    def _discard_failed_session(self, expected: CodeModeHostSession) -> None:
        with self._session_lock:
            if self._session is expected:
                self._session = None
        expected.close()

    def _invoke_tool(self, name: str, raw_input: Any, kind: str) -> Any:
        if name in {PUBLIC_TOOL_NAME, WAIT_TOOL_NAME}:
            raise ToolException(f"{PUBLIC_TOOL_NAME} cannot invoke itself")
        self.agent.tool_registry.get(name)
        if kind == "function":
            if raw_input is None:
                params: Any = {}
            elif isinstance(raw_input, dict):
                params = raw_input
            else:
                raise ToolException(f"tool '{name}' expects a JSON object")
        elif kind == "freeform":
            if not isinstance(raw_input, str):
                raise ToolException(f"tool '{name}' expects a string input")
            # Keep freeform patch bodies visible to the same anti-hack fields
            # used by direct apply_patch calls.
            params = {"input": raw_input} if name == "apply_patch" else raw_input
        else:
            raise ToolException(f"unsupported nested tool kind: {kind}")

        result = self.agent.execute_action({"tool": name, "params": params})
        if name == "apply_patch" and not result.get("success", False):
            raise ToolException(result.get("output") or "apply_patch failed")
        return self._nested_result(name, result)

    def _notify(self, call_id: str, _cell_id: str, text: str) -> None:
        if not text.strip():
            return
        self.agent._enqueue_code_mode_notification(call_id, text)

    @staticmethod
    def _nested_result(name: str, result: dict[str, Any]) -> Any:
        if name in ("apply_patch", "update_plan") and result.get("success"):
            # The report text ("M path" per file / "Plan updated") is the whole
            # observation; an empty object left the model to guess whether the
            # patch landed and re-check with git diff.
            return result.get("output") or ""
        if name == "view_image" and result.get("success"):
            media = result.get("media") or []
            if media:
                entry = media[0]
                return {
                    "image_url": f"data:{entry.get('media_type', 'application/octet-stream')};base64,{entry.get('data', '')}",
                    "detail": entry.get("detail", "high"),
                }
        metadata = result.get("metadata") or {}
        if name == "exec_command":
            # Layer 2 of the upstream stack (``code_mode_result``): the budget the
            # call passed is applied here, on the way into JavaScript; without one
            # JavaScript gets the collected text as is.
            output, original_token_count = truncate_exec_output(
                result.get("output", ""),
                metadata.get("output_omitted_bytes") or 0,
                metadata.get("max_output_tokens"),
            )
            nested = {"output": output}
            for key in (
                "chunk_id",
                "session_id",
                "exit_code",
                "wall_time_seconds",
                # A deadline hit is a normal result here, so the flag is the only
                # way a cell can branch on it instead of catching a rejection.
                "timed_out",
            ):
                if key in metadata:
                    nested[key] = metadata[key]
            if original_token_count is not None:
                nested["original_token_count"] = original_token_count
            return nested
        return result

    def _format_response(self, response: dict[str, Any], max_tokens: int | None, started: float) -> ToolOutput:
        variant, payload = next(iter(response.items()))
        cell_id = payload["cell_id"]
        items = payload.get("content_items") or []
        text_items = [str(item.get("text", "")) for item in items if item.get("type") == "input_text"]
        media: list[dict[str, str]] = []
        for item in items:
            entry, rejection = _media_entry(item)
            if entry is not None:
                media.append(entry)
            elif rejection is not None:
                # Reported, not silently discarded: the cell has to know its
                # image never left, and this text is what tells it.
                text_items.append(rejection)
        error_text = payload.get("error_text") if variant == "Result" else None
        if variant == "Yielded":
            status = f"Script running with cell ID {cell_id}"
            success = True
        elif variant == "Terminated":
            status = "Script terminated"
            success = True
        elif error_text is None:
            status = "Script completed"
            success = True
        else:
            status = "Script failed"
            success = False
            for match in _JS_UNKNOWN_TOOL_RE.finditer(error_text):
                self.agent._record_nested_contract_error(match.group(1) or "<computed>", "unknown_tool")
            for name in _JS_BAD_ARGS_RE.findall(error_text):
                self.agent._record_nested_contract_error(name, "bad_args_type")
            text_items.append(f"Script error:\n{error_text}")
        elapsed = round(time.monotonic() - started, 1)
        # Layer 3 (``truncate_code_mode_result``): the cell budget, uncapped.
        content = formatted_truncate_text("\n".join(text_items), resolve_max_tokens(max_tokens))
        output = f"{status}\nWall time {elapsed:.1f} seconds\nOutput:\n{content}"
        return ToolOutput(output=output, success=success, media=media)


class ExecTool(BaseTool):
    """Internal dispatcher for the freeform ``exec`` tool's raw JavaScript."""

    @property
    def name(self) -> str:
        return PUBLIC_TOOL_NAME

    @property
    def description(self) -> str:
        return _exec_description([])

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        runtime = (context or {}).get("code_mode_runtime")
        call_id = (context or {}).get("tool_call_id")
        if not isinstance(runtime, CodeModeRuntime) or not call_id:
            raise ToolException("exec requires code-mode runtime context")
        if not isinstance(params, str):
            raise ToolException("exec expects raw JavaScript source text")
        return runtime.execute(call_id, params)


class WaitTool(BaseTool):
    @property
    def name(self) -> str:
        return WAIT_TOOL_NAME

    @property
    def description(self) -> str:
        return "Waits on a yielded exec cell and returns new output or completion."

    def get_function_parameters(self) -> dict[str, Any]:
        return self.definition()["function"]["parameters"]

    @classmethod
    def definition(cls, *, yield_time_ms: int = DEFAULT_WAIT_YIELD_TIME_MS) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": WAIT_TOOL_NAME,
                "description": (
                    "Waits on a yielded `exec` cell. Use only after exec returns `Script running with cell ID ...`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_id": {"type": "string", "description": "Identifier of the running exec cell."},
                        "yield_time_ms": {
                            "type": "number",
                            "description": (
                                f"How long to wait for the cell to finish before returning its output so far, "
                                f"in milliseconds. Defaults to {yield_time_ms}."
                            ),
                        },
                        "max_tokens": {
                            "type": "number",
                            "description": (
                                f"Limits how much new output this wait call returns. "
                                f"Defaults to {DEFAULT_MAX_OUTPUT_TOKENS} tokens."
                            ),
                        },
                        "terminate": {"type": "boolean", "description": "Stop the running cell."},
                    },
                    "required": ["cell_id"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        runtime = (context or {}).get("code_mode_runtime")
        if not isinstance(runtime, CodeModeRuntime):
            raise ToolException("wait requires code-mode runtime context")
        if not isinstance(params, dict):
            raise ToolException("wait arguments must be an object")
        unsupported = set(params) - {"cell_id", "yield_time_ms", "max_tokens", "terminate"}
        if unsupported:
            raise ToolException(f"wait has unsupported field(s): {', '.join(sorted(unsupported))}")
        cell_id = params.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ToolException("wait requires a non-empty cell_id")
        for key in ("yield_time_ms", "max_tokens"):
            value = params.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ToolException(f"wait field '{key}' must be a non-negative integer")
        terminate = params.get("terminate", False)
        if not isinstance(terminate, bool):
            raise ToolException("wait field 'terminate' must be a boolean")
        return runtime.wait(
            cell_id,
            yield_time_ms=params.get("yield_time_ms"),
            max_tokens=params.get("max_tokens"),
            terminate=terminate,
        )


def parse_exec_call_arguments(raw: Any) -> str:
    """``exec`` is a freeform tool, so its arguments are raw JavaScript text."""

    return raw if isinstance(raw, str) else str(raw or "")


__all__ = [
    "CodeModeRuntime",
    "DEFAULT_EXEC_YIELD_TIME_MS",
    "DEFAULT_WAIT_YIELD_TIME_MS",
    "ExecTool",
    "ParsedExecSource",
    "WaitTool",
    "build_exec_tool_definition",
    "parse_exec_call_arguments",
    "parse_exec_source",
]
