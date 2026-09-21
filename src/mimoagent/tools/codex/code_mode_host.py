"""Synchronous client for the standalone Codex code-mode host.

The host's ``stdio`` transport is not JSONL. Each message is UTF-8 JSON
prefixed by a four-byte little-endian payload length. A connection multiplexes
operations and host-initiated tool callbacks, so a dedicated reader thread is
required even for synchronous callers.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import shutil
import struct
import subprocess
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_PENDING_DELEGATES = 1024
RECENT_DELEGATE_IDS = 4096
RECENT_CLOSED_CELLS = 4096
PROTOCOL_VERSION = 1
SESSION_RESOURCE_LIMITS_CAPABILITY = "session-cell-execution-resource-limits"
# The Code Mode host ships as a standalone asset of the openai/codex GitHub
# release (static musl build, runs on glibc and musl hosts). Pin it to the same
# release as the Codex CLI adapter. Override with MIMOAGENT_CODE_MODE_HOST_URL
# (a tarball containing the executable) or MIMOAGENT_CODE_MODE_HOST_PATH.
CODE_MODE_HOST_RELEASE = "rust-v0.154.0"
DEFAULT_CODE_MODE_HOST_ARCHIVE_URL = (
    f"https://github.com/openai/codex/releases/download/{CODE_MODE_HOST_RELEASE}/"
    "codex-code-mode-host-x86_64-unknown-linux-musl.tar.gz"
)
CODE_MODE_HOST_URL_ENV = "MIMOAGENT_CODE_MODE_HOST_URL"
CODE_MODE_HOST_PATH_ENV = "MIMOAGENT_CODE_MODE_HOST_PATH"
CODE_MODE_HOST_DOWNLOAD_TIMEOUT = 30.0
MAX_HOST_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_HOST_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
_HOST_DOWNLOAD_LOCK = threading.Lock()
_NON_INHERITABLE_ENV_VARS = {
    "CODEX_EXEC_SERVER_NOISE_AUTH_TOKEN",
    "NODE_REPL_AUTH_TOKEN",
    "OPENAI_FEDERATION_RULE_ID",
    "OPENAI_IDENTITY_TOKEN_FILE",
    "OPENAI_WORKLOAD_IDENTITY_CONTEXT",
}

InvokeTool = Callable[[str, Any, str], Any]
Notify = Callable[[str, str, str], None]


class CodeModeHostError(RuntimeError):
    """Base error raised by the code-mode host client."""


class CodeModeHostProtocolError(CodeModeHostError):
    """The host sent an invalid or unsupported protocol message."""


class CodeModeHostOperationError(CodeModeHostError):
    """The host returned ``status: error`` for an operation."""


class CodeModeHostTimeoutError(CodeModeHostError):
    """A host handshake or operation exceeded its local deadline."""


class CodeModeHostClosedError(CodeModeHostError):
    """The host connection is closed or failed."""


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_executable(value: str | os.PathLike[str]) -> Path | None:
    raw = os.fspath(value)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if _is_executable_file(path):
        return path.resolve()
    return None


def _host_environment() -> dict[str, str]:
    restricted = {name.casefold() for name in _NON_INHERITABLE_ENV_VARS}
    return {name: value for name, value in os.environ.items() if name.casefold() not in restricted}


def _code_mode_host_cache_root() -> Path:
    configured = os.environ.get("MIMOAGENT_CODE_MODE_HOST_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser() / "mimoagent" / "codex-code-mode-host"
    return Path.home() / ".cache" / "mimoagent" / "codex-code-mode-host"


def _cached_code_mode_host(root: Path) -> Path | None:
    executable_names = {"codex-code-mode-host", "codex-code-mode-host.exe"}
    if not root.is_dir():
        return None
    for candidate in root.rglob("*"):
        if candidate.name not in executable_names and not candidate.name.startswith("codex-code-mode-host-"):
            continue
        if _is_executable_file(candidate):
            return candidate.resolve()
    return None


def _safe_extract_host_archive(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    extracted_bytes = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            name = member.name
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts:
                raise CodeModeHostProtocolError(f"unsafe code-mode host archive member: {name!r}")
            if member.issym() or member.islnk() or not (member.isdir() or member.isreg()):
                raise CodeModeHostProtocolError(f"unsupported code-mode host archive member: {name!r}")
            if member.isdir():
                continue
            if member.size < 0 or member.size > MAX_HOST_ARCHIVE_MEMBER_BYTES:
                raise CodeModeHostProtocolError(f"code-mode host archive member is too large: {name!r}")
            extracted_bytes += member.size
            if extracted_bytes > MAX_HOST_ARCHIVE_BYTES:
                raise CodeModeHostProtocolError("code-mode host archive expands beyond the size limit")
            target = (destination / path).resolve()
            if target != destination and destination not in target.parents:
                raise CodeModeHostProtocolError(f"unsafe code-mode host archive member: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise CodeModeHostProtocolError(f"failed to read code-mode host archive member: {name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)


def _download_code_mode_host() -> Path:
    cache_root = _code_mode_host_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = _cached_code_mode_host(cache_root)
    if cached is not None:
        return cached

    with _HOST_DOWNLOAD_LOCK:
        cached = _cached_code_mode_host(cache_root)
        if cached is not None:
            return cached
        staging = Path(tempfile.mkdtemp(prefix=".codex-host-", dir=cache_root))
        archive = staging / "host.tar.gz"
        try:
            url = os.environ.get(CODE_MODE_HOST_URL_ENV) or DEFAULT_CODE_MODE_HOST_ARCHIVE_URL
            logger.info("downloading Code Mode host from %s", url)
            with urllib.request.urlopen(url, timeout=CODE_MODE_HOST_DOWNLOAD_TIMEOUT) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_HOST_ARCHIVE_BYTES:
                    raise CodeModeHostProtocolError("code-mode host archive is larger than the size limit")
                total = 0
                with archive.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_HOST_ARCHIVE_BYTES:
                            raise CodeModeHostProtocolError("code-mode host archive is larger than the size limit")
                        output.write(chunk)
            extracted = staging / "extracted"
            extracted.mkdir()
            _safe_extract_host_archive(archive, extracted)
            host = _cached_code_mode_host(extracted)
            if host is None:
                raise CodeModeHostProtocolError("downloaded archive contains no executable Code Mode host")
            archive.unlink()
            install = cache_root / f"host-{uuid.uuid4().hex}"
            os.replace(extracted, install)
            staging.rmdir()
            resolved = _cached_code_mode_host(install)
            if resolved is None:
                raise CodeModeHostProtocolError("downloaded Code Mode host disappeared from the cache")
            logger.info("using cached Code Mode host at %s", resolved)
            return resolved
        except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise FileNotFoundError(
                f"failed to download Code Mode host from {url}: {exc}. Set {CODE_MODE_HOST_PATH_ENV} (or the "
                f"agent's code_mode_host_path) to a local codex-code-mode-host binary, or "
                f"{CODE_MODE_HOST_URL_ENV} to a mirror of the release tarball."
            ) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def find_code_mode_host(host_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the Code Mode host executable.

    Resolution order: an explicit ``host_path`` (the agent's
    ``code_mode_host_path``), then ``MIMOAGENT_CODE_MODE_HOST_PATH``, then the
    cached download of the pinned openai/codex release asset (or
    ``MIMOAGENT_CODE_MODE_HOST_URL``). Hosts without network access point one
    of the path knobs at a local copy of the binary.
    """

    host_path = host_path or os.environ.get(CODE_MODE_HOST_PATH_ENV)
    if host_path:
        resolved = _resolve_executable(host_path)
        if resolved is None:
            raise FileNotFoundError(f"configured code-mode host is not executable: {os.fspath(host_path)}")
        return resolved
    return _download_code_mode_host()


def build_tool_definition(
    name: str,
    *,
    description: str = "",
    kind: str = "function",
    input_schema: Any = None,
    output_schema: Any = None,
    tool_name: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Build the exact V1 wire shape for one nested tool definition."""

    if kind not in {"function", "freeform"}:
        raise ValueError(f"unsupported code-mode tool kind: {kind!r}")
    return {
        "name": name,
        "tool_name": {"name": tool_name or name, "namespace": namespace},
        "description": description,
        "kind": kind,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }


@dataclass
class _PendingRequest:
    operation_ready: threading.Event = field(default_factory=threading.Event)
    initial_ready: threading.Event = field(default_factory=threading.Event)
    operation: dict[str, Any] | None = None
    initial: dict[str, Any] | None = None
    error: BaseException | None = None


def _read_exact(stream: BinaryIO, size: int, *, eof_at_boundary: bool = False) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            if not data and eof_at_boundary:
                return None
            raise EOFError(f"code-mode host closed stdout after {len(data)} of {size} bytes")
        data.extend(chunk)
    return bytes(data)


def _read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    header = _read_exact(stream, 4, eof_at_boundary=True)
    if header is None:
        return None
    length = struct.unpack("<I", header)[0]
    if length > MAX_FRAME_BYTES:
        raise CodeModeHostProtocolError(f"code-mode IPC frame length {length} exceeds {MAX_FRAME_BYTES} bytes")
    payload = _read_exact(stream, length)
    assert payload is not None
    try:
        message = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeModeHostProtocolError(f"failed to decode code-mode IPC frame: {exc}") from exc
    if not isinstance(message, dict):
        raise CodeModeHostProtocolError("code-mode IPC message must be a JSON object")
    return message


def _encode_frame(message: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise CodeModeHostProtocolError(f"failed to encode code-mode IPC frame: {exc}") from exc
    if len(payload) > MAX_FRAME_BYTES:
        raise CodeModeHostProtocolError(f"code-mode IPC frame length {len(payload)} exceeds {MAX_FRAME_BYTES} bytes")
    return struct.pack("<I", len(payload)) + payload


def _require_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing or unexpected:
        raise CodeModeHostProtocolError(
            f"invalid {label} fields (missing={sorted(missing)}, unexpected={sorted(unexpected)}): {value!r}"
        )


def _write_all(stream: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None:
            # Buffered binary streams conventionally return None only in
            # non-blocking mode, which is not used for the host pipe.
            raise BrokenPipeError("code-mode host pipe made no write progress")
        if written <= 0:
            raise BrokenPipeError("code-mode host pipe closed during frame write")
        remaining = remaining[written:]


class CodeModeHostSession:
    """One durable code-mode session backed by a local host subprocess.

    ``invoke_tool`` is called as ``invoke_tool(name, input, kind)`` on a worker
    thread. Calls may overlap and must therefore be thread-safe. Namespaced tool
    names are flattened using the same separator rule as Codex; plain tool names
    are passed unchanged.
    """

    def __init__(
        self,
        host_path: str | os.PathLike[str],
        invoke_tool: InvokeTool,
        *,
        session_id: str | None = None,
        enabled_tools: Sequence[Mapping[str, Any]] | None = None,
        notify: Notify | None = None,
        startup_timeout: float = 30.0,
        request_timeout: float = 60.0,
        close_timeout: float = 5.0,
        delegate_workers: int = 8,
        cell_execution_limits: Mapping[str, int | None] | None = None,
        fatal_delegate_errors: tuple[type[BaseException], ...] = (),
    ):
        resolved_host = _resolve_executable(host_path)
        if resolved_host is None:
            raise FileNotFoundError(f"code-mode host is not executable: {os.fspath(host_path)}")
        if not callable(invoke_tool):
            raise TypeError("invoke_tool must be callable")
        if delegate_workers < 1:
            raise ValueError("delegate_workers must be at least 1")
        if not isinstance(fatal_delegate_errors, tuple) or not all(
            isinstance(error_type, type) and issubclass(error_type, BaseException)
            for error_type in fatal_delegate_errors
        ):
            raise TypeError("fatal_delegate_errors must be a tuple of exception classes")
        if min(startup_timeout, request_timeout, close_timeout) <= 0:
            raise ValueError("code-mode host timeouts must be positive")

        self.host_path = resolved_host
        self.session_id = session_id or f"mimo-{uuid.uuid4()}"
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        self.enabled_tools = [dict(tool) for tool in (enabled_tools or [])]
        self._invoke_tool = invoke_tool
        self._notify = notify
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._close_timeout = close_timeout
        self._cell_execution_limits = dict(cell_execution_limits or {})
        self._fatal_delegate_error_types = fatal_delegate_errors

        self._state_lock = threading.RLock()
        self._write_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self._next_request_id = 1
        self._pending: dict[int, _PendingRequest] = {}
        self._delegate_futures: dict[int, concurrent.futures.Future | None] = {}
        self._delegate_cells: dict[int, str] = {}
        self._delegate_committed: set[int] = set()
        self._cancelled_delegates: set[int] = set()
        self._recent_delegate_ids: deque[int] = deque()
        self._recent_delegate_id_set: set[int] = set()
        self._pending_delegates_by_cell: dict[str, list[dict[str, Any]]] = {}
        self._active_cells: set[str] = set()
        self._starting_executions = 0
        self._fatal_delegate_errors_by_cell: dict[str, BaseException] = {}
        self._closed_cells: set[str] = set()
        self._failure: BaseException | None = None
        self._closing = False
        self._closed = False
        self._closed_event = threading.Event()
        self._session_open = False
        self._hello: dict[str, Any] | None = None
        self._hello_ready = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._delegate_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=delegate_workers,
            thread_name_prefix="codex-code-mode-delegate",
        )

        try:
            self._process = subprocess.Popen(
                [str(self.host_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=os.name != "nt",
                env=_host_environment(),
            )
        except OSError:
            self._delegate_executor.shutdown(wait=False, cancel_futures=True)
            raise

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="codex-code-mode-writer",
            daemon=True,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="codex-code-mode-reader",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name="codex-code-mode-stderr",
            daemon=True,
        )
        self._writer_thread.start()
        self._reader_thread.start()
        self._stderr_thread.start()

        try:
            self._send_frame(
                {
                    "type": "connection/hello",
                    "supportedVersions": [PROTOCOL_VERSION],
                    "requiredCapabilities": [],
                    "optionalCapabilities": [SESSION_RESOURCE_LIMITS_CAPABILITY],
                }
            )
            self._wait_event(self._hello_ready, startup_timeout, "host handshake")
            self._validate_hello()
            self._open_session(startup_timeout)
        except BaseException:
            self._abort_startup()
            raise

    @property
    def stderr_tail(self) -> str:
        """Return the recent host stderr log, for diagnostics only."""

        with self._state_lock:
            return "\n".join(self._stderr_tail)

    @property
    def closed_cells(self) -> frozenset[str]:
        with self._state_lock:
            return frozenset(self._closed_cells)

    def execute(
        self,
        tool_call_id: str,
        source: str,
        yield_time_ms: int | None = None,
        *,
        max_output_tokens: int | None = None,
        enabled_tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute one JavaScript cell and return its first runtime response."""

        if not tool_call_id:
            raise ValueError("tool_call_id must not be empty")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be non-empty JavaScript text")
        selected_tools = self.enabled_tools if enabled_tools is None else enabled_tools
        request = {
            "tool_call_id": tool_call_id,
            "enabled_tools": [dict(tool) for tool in selected_tools],
            "source": source,
            "yield_time_ms": yield_time_ms,
            "max_output_tokens": max_output_tokens,
        }
        timeout = self._runtime_timeout(yield_time_ms)
        request_id, pending = self._register_request(allow_closing=False)
        with self._state_lock:
            self._starting_executions += 1
        starting_registered = True
        cell_id: str | None = None
        try:
            self._send_frame(
                {
                    "type": "operation/request",
                    "id": request_id,
                    "request": {
                        "method": "session/execute",
                        "sessionId": self.session_id,
                        "request": request,
                    },
                }
            )
            self._wait_event(pending.operation_ready, timeout, f"operation {request_id}")
            started = self._unwrap_result(self._pending_message(pending, initial=False))
            _require_fields(started, required={"type", "cellId"}, label="execution started")
            if started.get("type") != "execution/started" or not isinstance(started.get("cellId"), str):
                raise CodeModeHostProtocolError(f"invalid execution-started response: {started!r}")
            cell_id = started["cellId"]
            self._admit_cell(cell_id)
            starting_registered = False
            self._wait_event(pending.initial_ready, timeout, f"execute initial response {request_id}")
            initial = self._unwrap_result(self._pending_message(pending, initial=True))
        except CodeModeHostTimeoutError:
            try:
                self._send_frame({"type": "operation/cancel", "id": request_id})
            except CodeModeHostError:
                pass
            if cell_id is not None:
                self._terminate_started_cell(cell_id)
            raise
        finally:
            if starting_registered:
                self._retire_starting_execution()
            with self._state_lock:
                self._pending.pop(request_id, None)

        try:
            response = self._validate_runtime_response(initial)
        except CodeModeHostProtocolError as error:
            self._fail_connection(error)
            raise
        if self._runtime_cell_id(response) != cell_id:
            raise CodeModeHostProtocolError(
                f"execute response cell does not match started cell {cell_id!r}: {response!r}"
            )
        self._raise_fatal_delegate_error(response)
        return response

    def _terminate_started_cell(self, cell_id: str) -> None:
        """Best-effort cleanup after execute started but never produced an initial result."""

        try:
            response, _ = self._perform_operation(
                {
                    "method": "session/terminate",
                    "sessionId": self.session_id,
                    "cellId": cell_id,
                },
                expect_initial=False,
                timeout=self._close_timeout,
            )
            if response.get("type") != "wait/completed":
                raise CodeModeHostProtocolError(f"invalid timeout termination response: {response!r}")
            _require_fields(response, required={"type", "outcome"}, label="timeout termination")
            with self._state_lock:
                self._fatal_delegate_errors_by_cell.pop(cell_id, None)
        except CodeModeHostError as cleanup_error:
            error = CodeModeHostClosedError(f"failed to terminate timed-out code-mode cell {cell_id}: {cleanup_error}")
            self._fail_connection(error)
            raise error from cleanup_error

    def wait(
        self,
        cell_id: str,
        yield_time_ms: int = 10_000,
        *,
        terminate: bool = False,
    ) -> dict[str, Any]:
        """Wait for new cell output, or terminate the cell when requested."""

        if not cell_id:
            raise ValueError("cell_id must not be empty")
        if terminate:
            request = {
                "method": "session/terminate",
                "sessionId": self.session_id,
                "cellId": cell_id,
            }
        else:
            request = {
                "method": "session/wait",
                "sessionId": self.session_id,
                "request": {"cell_id": cell_id, "yield_time_ms": yield_time_ms},
            }
        response, _ = self._perform_operation(
            request,
            expect_initial=False,
            timeout=self._runtime_timeout(0 if terminate else yield_time_ms),
        )
        if response.get("type") != "wait/completed":
            raise CodeModeHostProtocolError(f"invalid wait response: {response!r}")
        _require_fields(response, required={"type", "outcome"}, label="wait response")
        outcome = response.get("outcome")
        if not isinstance(outcome, dict) or len(outcome) != 1:
            raise CodeModeHostProtocolError(f"invalid wait outcome: {outcome!r}")
        state, runtime_response = next(iter(outcome.items()))
        if state not in {"LiveCell", "MissingCell"}:
            raise CodeModeHostProtocolError(f"unknown wait outcome state: {state!r}")
        try:
            runtime_response = self._validate_runtime_response(runtime_response)
        except CodeModeHostProtocolError as error:
            self._fail_connection(error)
            raise
        if self._runtime_cell_id(runtime_response) != cell_id:
            raise CodeModeHostProtocolError(
                f"wait response cell does not match requested cell {cell_id!r}: {runtime_response!r}"
            )
        self._raise_fatal_delegate_error(runtime_response)
        return runtime_response

    def close(self) -> None:
        """Close the logical session and reap the host process. Idempotent."""

        with self._state_lock:
            if self._closed:
                return
            wait_for_owner = self._closing
            if not wait_for_owner:
                self._closing = True
        if wait_for_owner:
            self._closed_event.wait()
            return

        try:
            self._close_owned_session()
        finally:
            with self._state_lock:
                self._closed = True
                self._fail_pending_locked(CodeModeHostClosedError("code-mode host session is closed"))
                self._closed_event.set()

    def _close_owned_session(self) -> None:
        """Close implementation run by exactly one caller."""

        # A running delegate may still be mutating the workspace, and the host
        # may be waiting for its response before it can process shutdown. Drain
        # callbacks first; the single writer preserves response-before-shutdown
        # ordering on the wire. This wait intentionally uses each nested tool's
        # own execution timeout instead of close_timeout: returning while a tool
        # can still edit the workspace would race reward calculation/cleanup.
        self._delegate_executor.shutdown(wait=True, cancel_futures=True)
        if self._session_open and self._process.poll() is None and self._failure is None:
            try:
                response, _ = self._perform_operation(
                    {"method": "session/shutdown", "sessionId": self.session_id},
                    expect_initial=False,
                    timeout=self._close_timeout,
                    allow_closing=True,
                )
                if response.get("type") != "session/closed":
                    raise CodeModeHostProtocolError(f"invalid session-close response: {response!r}")
                _require_fields(response, required={"type", "sessionId"}, label="session close")
            except CodeModeHostError:
                logger.debug("failed to close code-mode session cleanly", exc_info=True)
        self._session_open = False

        self._stop_writer()
        self._close_process_streams()
        self._reap_process()
        self._writer_thread.join(timeout=self._close_timeout)
        self._reader_thread.join(timeout=self._close_timeout)
        self._stderr_thread.join(timeout=self._close_timeout)

    def __enter__(self) -> CodeModeHostSession:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _runtime_timeout(self, yield_time_ms: int | None) -> float:
        if yield_time_ms is None:
            return self._request_timeout
        if not isinstance(yield_time_ms, int) or isinstance(yield_time_ms, bool) or yield_time_ms < 0:
            raise ValueError("yield_time_ms must be a non-negative integer")
        return max(self._request_timeout, yield_time_ms / 1000.0 + 5.0)

    def _open_session(self, timeout: float) -> None:
        request: dict[str, Any] = {"method": "session/open", "sessionId": self.session_id}
        if self._cell_execution_limits:
            capabilities = self._hello.get("capabilities", []) if self._hello else []
            if SESSION_RESOURCE_LIMITS_CAPABILITY not in capabilities:
                raise CodeModeHostProtocolError("code-mode host did not negotiate session resource limits")
            limits: dict[str, int] = {}
            key_map = {
                "max_yield_time_ms": "maxYieldTimeMs",
                "max_heap_size_bytes": "maxHeapSizeBytes",
                "maxYieldTimeMs": "maxYieldTimeMs",
                "maxHeapSizeBytes": "maxHeapSizeBytes",
            }
            for key, value in self._cell_execution_limits.items():
                if key not in key_map:
                    raise ValueError(f"unknown cell execution limit: {key}")
                if value is not None:
                    limits[key_map[key]] = value
            request["cellExecutionLimits"] = limits
        response, _ = self._perform_operation(request, expect_initial=False, timeout=timeout)
        if response != {"type": "session/ready", "sessionId": self.session_id}:
            raise CodeModeHostProtocolError(f"invalid session-ready response: {response!r}")
        self._session_open = True

    def _perform_operation(
        self,
        request: Mapping[str, Any],
        *,
        expect_initial: bool,
        timeout: float,
        allow_closing: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        request_id, pending = self._register_request(allow_closing=allow_closing)
        try:
            self._send_frame({"type": "operation/request", "id": request_id, "request": dict(request)})
            self._wait_event(pending.operation_ready, timeout, f"operation {request_id}")
            operation = self._pending_message(pending, initial=False)
            value = self._unwrap_result(operation)
            if not expect_initial:
                return value, None
            self._wait_event(pending.initial_ready, timeout, f"execute initial response {request_id}")
            initial = self._pending_message(pending, initial=True)
            return value, self._unwrap_result(initial)
        except CodeModeHostTimeoutError:
            try:
                self._send_frame({"type": "operation/cancel", "id": request_id})
            except CodeModeHostError:
                pass
            raise
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    def _register_request(self, *, allow_closing: bool) -> tuple[int, _PendingRequest]:
        with self._state_lock:
            self._raise_if_unavailable_locked(allow_closing=allow_closing)
            request_id = self._next_request_id
            self._next_request_id += 1
            pending = _PendingRequest()
            self._pending[request_id] = pending
            return request_id, pending

    def _pending_message(self, pending: _PendingRequest, *, initial: bool) -> dict[str, Any]:
        if pending.error is not None:
            raise CodeModeHostClosedError(str(pending.error)) from pending.error
        message = pending.initial if initial else pending.operation
        if message is None:
            raise CodeModeHostProtocolError("host signaled a response without a message")
        return message

    def _wait_event(self, event: threading.Event, timeout: float, description: str) -> None:
        if not event.wait(timeout):
            raise CodeModeHostTimeoutError(f"timed out waiting for code-mode {description}")
        with self._state_lock:
            if self._failure is not None:
                raise CodeModeHostClosedError(str(self._failure)) from self._failure

    @staticmethod
    def _unwrap_result(message: Mapping[str, Any]) -> dict[str, Any]:
        result = message.get("result")
        if not isinstance(result, dict):
            raise CodeModeHostProtocolError(f"host response has no wire result: {message!r}")
        status = result.get("status")
        if status == "error":
            _require_fields(result, required={"status", "message"}, label="error result")
            raise CodeModeHostOperationError(str(result.get("message") or "code-mode operation failed"))
        value = result.get("value")
        if status != "ok" or not isinstance(value, dict):
            raise CodeModeHostProtocolError(f"invalid host wire result: {result!r}")
        _require_fields(result, required={"status", "value"}, label="success result")
        return value

    @staticmethod
    def _validate_runtime_response(response: Any) -> dict[str, Any]:
        if not isinstance(response, dict) or len(response) != 1:
            raise CodeModeHostProtocolError(f"invalid runtime response: {response!r}")
        variant, payload = next(iter(response.items()))
        if variant not in {"Yielded", "Terminated", "Result"} or not isinstance(payload, dict):
            raise CodeModeHostProtocolError(f"invalid runtime response: {response!r}")
        if not isinstance(payload.get("cell_id"), str) or not isinstance(payload.get("content_items"), list):
            raise CodeModeHostProtocolError(f"invalid runtime response payload: {payload!r}")
        expected_keys = (
            {"cell_id", "content_items", "error_text"}
            if variant == "Result"
            else {
                "cell_id",
                "content_items",
            }
        )
        if set(payload) != expected_keys:
            raise CodeModeHostProtocolError(f"unexpected runtime response fields: {payload!r}")
        if variant == "Result" and payload["error_text"] is not None and not isinstance(payload["error_text"], str):
            raise CodeModeHostProtocolError(f"invalid runtime error text: {payload!r}")
        for item in payload["content_items"]:
            if not isinstance(item, dict):
                raise CodeModeHostProtocolError(f"invalid runtime content item: {item!r}")
            item_type = item.get("type")
            if item_type == "input_text":
                valid = set(item) == {"type", "text"} and isinstance(item.get("text"), str)
            elif item_type == "input_image":
                detail = item.get("detail")
                valid = (
                    set(item) <= {"type", "image_url", "detail"}
                    and set(item) >= {"type", "image_url"}
                    and isinstance(item.get("image_url"), str)
                    and (detail is None or (isinstance(detail, str) and detail in {"auto", "low", "high", "original"}))
                )
            elif item_type == "input_audio":
                valid = set(item) == {"type", "audio_url"} and isinstance(item.get("audio_url"), str)
            else:
                valid = False
            if not valid:
                raise CodeModeHostProtocolError(f"invalid runtime content item: {item!r}")
        return response

    @staticmethod
    def _runtime_cell_id(response: Mapping[str, Any]) -> str:
        payload = next(iter(response.values()))
        return payload["cell_id"]

    def _validate_hello(self) -> None:
        hello = self._hello
        if hello is None:
            raise CodeModeHostProtocolError("code-mode host returned no handshake response")
        if hello.get("type") == "connection/rejected":
            _require_fields(hello, required={"type", "reason"}, label="handshake rejection")
            raise CodeModeHostProtocolError(f"code-mode host rejected handshake: {hello.get('reason')!r}")
        _require_fields(
            hello,
            required={"type", "selectedVersion", "capabilities"},
            label="host handshake",
        )
        if hello.get("type") != "connection/ready" or hello.get("selectedVersion") != PROTOCOL_VERSION:
            raise CodeModeHostProtocolError(f"invalid code-mode host handshake: {hello!r}")
        capabilities = hello.get("capabilities")
        if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
            raise CodeModeHostProtocolError(f"invalid code-mode host capabilities: {capabilities!r}")
        if len(capabilities) != len(set(capabilities)) or set(capabilities) - {SESSION_RESOURCE_LIMITS_CAPABILITY}:
            raise CodeModeHostProtocolError(f"unexpected code-mode host capabilities: {capabilities!r}")

    def _send_frame(self, message: Mapping[str, Any]) -> None:
        frame = _encode_frame(message)
        with self._state_lock:
            self._raise_if_unavailable_locked(allow_closing=True)
        try:
            self._write_queue.put(frame, timeout=min(0.25, self._request_timeout))
        except queue.Full as exc:
            error = CodeModeHostClosedError("code-mode host outgoing queue is full")
            self._fail_connection(error)
            raise error from exc

    def _writer_loop(self) -> None:
        try:
            process_stdin = self._process.stdin
            if process_stdin is None:
                raise CodeModeHostClosedError("spawned code-mode host has no stdin")
            while True:
                frame = self._write_queue.get()
                if frame is None:
                    return
                _write_all(process_stdin, frame)
                process_stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            with self._state_lock:
                expected = self._closing or self._closed
            if not expected:
                self._fail_connection(exc)

    def _reader_loop(self) -> None:
        try:
            process_stdout = self._process.stdout
            if process_stdout is None:
                raise CodeModeHostClosedError("spawned code-mode host has no stdout")
            while True:
                message = _read_frame(process_stdout)
                if message is None:
                    raise CodeModeHostClosedError("code-mode host closed stdout")
                self._dispatch_message(message)
        except BaseException as exc:
            with self._state_lock:
                expected = self._closing or self._closed
            if not expected:
                self._fail_connection(exc)

    def _dispatch_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type in {"connection/ready", "connection/rejected"}:
            with self._state_lock:
                if self._hello is not None:
                    raise CodeModeHostProtocolError("received a second code-mode host hello")
                self._hello = message
                self._hello_ready.set()
            return
        if message_type in {"operation/response", "execute/initialResponse"}:
            _require_fields(message, required={"type", "id", "result"}, label=message_type)
            request_id = message.get("id")
            if not isinstance(request_id, int) or isinstance(request_id, bool):
                raise CodeModeHostProtocolError(f"invalid operation response ID: {request_id!r}")
            with self._state_lock:
                pending = self._pending.get(request_id)
                if pending is None:
                    logger.debug("ignoring response for retired code-mode request %s", request_id)
                    return
                if message_type == "operation/response":
                    if pending.operation is not None:
                        raise CodeModeHostProtocolError(f"duplicate operation response for request {request_id}")
                    pending.operation = message
                    pending.operation_ready.set()
                else:
                    if pending.initial is not None:
                        raise CodeModeHostProtocolError(f"duplicate initial response for request {request_id}")
                    pending.initial = message
                    pending.initial_ready.set()
            return
        if message_type == "delegate/request":
            _require_fields(
                message,
                required={"type", "id", "sessionId", "request"},
                label="delegate request",
            )
            self._dispatch_delegate(message)
            return
        if message_type == "delegate/cancel":
            _require_fields(message, required={"type", "id"}, label="delegate cancellation")
            self._cancel_delegate(message)
            return
        if message_type == "cell/closed":
            _require_fields(
                message,
                required={"type", "sessionId", "cellId"},
                label="closed cell",
            )
            cell_id = message.get("cellId")
            if message.get("sessionId") != self.session_id or not isinstance(cell_id, str):
                raise CodeModeHostProtocolError(f"invalid closed-cell message: {message!r}")
            self._close_cell(cell_id)
            return
        raise CodeModeHostProtocolError(f"unexpected code-mode host message: {message!r}")

    def _admit_cell(self, cell_id: str) -> None:
        with self._state_lock:
            self._starting_executions -= 1
            already_closed = cell_id in self._closed_cells
            if not already_closed:
                self._active_cells.add(cell_id)
            queued = [] if already_closed else self._pending_delegates_by_cell.pop(cell_id, [])
            orphaned = bool(self._pending_delegates_by_cell) and self._starting_executions == 0
        for message in queued:
            self._schedule_delegate(message)
        if orphaned:
            self._fail_connection(CodeModeHostProtocolError("delegate request targets an unknown cell"))

    def _retire_starting_execution(self) -> None:
        with self._state_lock:
            self._starting_executions = max(0, self._starting_executions - 1)
            orphaned = bool(self._pending_delegates_by_cell) and self._starting_executions == 0
        if orphaned:
            self._fail_connection(
                CodeModeHostProtocolError("received delegate calls for an execution that never started")
            )

    def _close_cell(self, cell_id: str) -> None:
        with self._state_lock:
            self._active_cells.discard(cell_id)
            self._pending_delegates_by_cell.pop(cell_id, None)
            if len(self._closed_cells) >= RECENT_CLOSED_CELLS:
                self._closed_cells.pop()
            self._closed_cells.add(cell_id)
            delegate_ids = [
                delegate_id
                for delegate_id, delegate_cell_id in self._delegate_cells.items()
                if delegate_cell_id == cell_id and delegate_id not in self._delegate_committed
            ]
            for delegate_id in delegate_ids:
                self._cancelled_delegates.add(delegate_id)
                future = self._delegate_futures.get(delegate_id)
                if future is None or future.cancel():
                    self._finish_delegate_state_locked(delegate_id)

    def _dispatch_delegate(self, message: dict[str, Any]) -> None:
        delegate_id = message.get("id")
        request = message.get("request")
        if (
            not isinstance(delegate_id, int)
            or isinstance(delegate_id, bool)
            or message.get("sessionId") != self.session_id
            or not isinstance(request, dict)
        ):
            raise CodeModeHostProtocolError(f"invalid delegate request: {message!r}")
        cell_id = self._delegate_cell_id(request)
        with self._state_lock:
            if delegate_id in self._recent_delegate_id_set:
                raise CodeModeHostProtocolError(f"duplicate delegate request ID {delegate_id}")
            if len(self._delegate_futures) >= MAX_PENDING_DELEGATES:
                raise CodeModeHostProtocolError("code-mode host has too many pending delegate calls")
            if cell_id in self._closed_cells:
                raise CodeModeHostProtocolError(f"delegate request targets closed cell {cell_id}")
            if cell_id not in self._active_cells and self._starting_executions == 0:
                raise CodeModeHostProtocolError(f"delegate request targets unknown cell {cell_id}")
            if len(self._recent_delegate_ids) >= RECENT_DELEGATE_IDS:
                retired = self._recent_delegate_ids.popleft()
                self._recent_delegate_id_set.discard(retired)
            self._recent_delegate_ids.append(delegate_id)
            self._recent_delegate_id_set.add(delegate_id)
            self._delegate_futures[delegate_id] = None
            self._delegate_cells[delegate_id] = cell_id
            if cell_id not in self._active_cells:
                self._pending_delegates_by_cell.setdefault(cell_id, []).append(message)
                return
        self._schedule_delegate(message)

    @staticmethod
    def _delegate_cell_id(request: Mapping[str, Any]) -> str:
        if request.get("type") == "tool/invoke":
            _require_fields(request, required={"type", "invocation"}, label="tool delegate")
            invocation = request.get("invocation")
            if not isinstance(invocation, dict):
                raise CodeModeHostProtocolError(f"tool delegate has no invocation: {request!r}")
            _require_fields(
                invocation,
                required={"cell_id", "runtime_tool_call_id", "tool_name", "tool_kind"},
                optional={"input"},
                label="nested tool invocation",
            )
            if not isinstance(invocation.get("runtime_tool_call_id"), str):
                raise CodeModeHostProtocolError(f"invalid runtime tool call ID: {invocation!r}")
            tool_name = invocation.get("tool_name")
            if not isinstance(tool_name, dict):
                raise CodeModeHostProtocolError(f"invalid nested tool name: {invocation!r}")
            _require_fields(tool_name, required={"name", "namespace"}, label="nested tool name")
            cell_id = invocation.get("cell_id")
        elif request.get("type") == "notification/send":
            _require_fields(
                request,
                required={"type", "callId", "cellId", "text"},
                label="notification delegate",
            )
            cell_id = request.get("cellId")
        else:
            raise CodeModeHostProtocolError(f"unknown delegate request type: {request.get('type')!r}")
        if not isinstance(cell_id, str) or not cell_id:
            raise CodeModeHostProtocolError(f"delegate request has invalid cell ID: {request!r}")
        return cell_id

    def _schedule_delegate(self, message: Mapping[str, Any]) -> None:
        delegate_id = message["id"]
        request = message["request"]
        with self._state_lock:
            if delegate_id not in self._delegate_futures or delegate_id in self._cancelled_delegates:
                self._finish_delegate_state_locked(delegate_id)
                return
        try:
            future = self._delegate_executor.submit(self._run_delegate, delegate_id, request)
        except RuntimeError as exc:
            with self._state_lock:
                self._finish_delegate_state_locked(delegate_id)
            raise CodeModeHostClosedError("code-mode delegate executor is closed") from exc
        with self._state_lock:
            if delegate_id in self._delegate_futures:
                self._delegate_futures[delegate_id] = future
                cancel_requested = delegate_id in self._cancelled_delegates
            else:
                cancel_requested = True
        if cancel_requested and future.cancel():
            with self._state_lock:
                self._finish_delegate_state_locked(delegate_id)

    def _cancel_delegate(self, message: Mapping[str, Any]) -> None:
        delegate_id = message.get("id")
        if not isinstance(delegate_id, int) or isinstance(delegate_id, bool):
            raise CodeModeHostProtocolError(f"invalid delegate cancel ID: {delegate_id!r}")
        with self._state_lock:
            if delegate_id not in self._delegate_futures:
                return
            if delegate_id in self._delegate_committed:
                return
            self._cancelled_delegates.add(delegate_id)
            future = self._delegate_futures.get(delegate_id)
        if future is not None and future.cancel():
            with self._state_lock:
                self._finish_delegate_state_locked(delegate_id)

    def _finish_delegate_state_locked(self, delegate_id: int) -> None:
        self._delegate_futures.pop(delegate_id, None)
        self._delegate_cells.pop(delegate_id, None)
        self._delegate_committed.discard(delegate_id)
        self._cancelled_delegates.discard(delegate_id)

    def _run_delegate(self, delegate_id: int, request: Mapping[str, Any]) -> None:
        cell_id: str | None = None
        try:
            request_type = request.get("type")
            if request_type == "tool/invoke":
                invocation = request.get("invocation")
                if not isinstance(invocation, dict):
                    raise CodeModeHostProtocolError("tool delegate has no invocation")
                invocation_cell_id = invocation.get("cell_id")
                if not isinstance(invocation_cell_id, str):
                    raise CodeModeHostProtocolError(f"invalid tool invocation cell: {invocation!r}")
                cell_id = invocation_cell_id
                tool_name = invocation.get("tool_name")
                kind = invocation.get("tool_kind")
                if not isinstance(tool_name, dict) or kind not in {"function", "freeform"}:
                    raise CodeModeHostProtocolError(f"invalid tool invocation: {invocation!r}")
                name = self._flatten_tool_name(tool_name)
                tool_input = invocation.get("input")
                if kind == "function":
                    if tool_input is None:
                        tool_input = {}
                    if not isinstance(tool_input, dict):
                        raise ValueError(f"tool `{name}` expects a JSON object for arguments")
                elif not isinstance(tool_input, str):
                    raise ValueError(f"tool `{name}` expects a string input")
                result = self._invoke_tool(name, tool_input, kind)
                value = {"type": "tool/result", "result": result}
            elif request_type == "notification/send":
                call_id = request.get("callId")
                notification_cell_id = request.get("cellId")
                text = request.get("text")
                if not all(isinstance(value, str) for value in (call_id, notification_cell_id, text)):
                    raise CodeModeHostProtocolError(f"invalid notification delegate: {request!r}")
                cell_id = notification_cell_id
                if self._notify is not None:
                    self._notify(call_id, cell_id, text)
                value = {"type": "notification/delivered"}
            else:
                raise CodeModeHostProtocolError(f"unknown delegate request type: {request_type!r}")
            response: dict[str, Any] = {"status": "ok", "value": value}
        except BaseException as exc:
            if cell_id is not None and isinstance(exc, self._fatal_delegate_error_types):
                with self._state_lock:
                    if delegate_id not in self._cancelled_delegates and cell_id in self._active_cells:
                        self._fatal_delegate_errors_by_cell.setdefault(cell_id, exc)
            response = {"status": "error", "message": str(exc) or type(exc).__name__}

        with self._state_lock:
            cancelled = delegate_id in self._cancelled_delegates or (
                cell_id is not None and cell_id not in self._active_cells
            )
            # During graceful shutdown the host may still be waiting for this
            # result before it can close the session. Only a fully closed or
            # failed connection makes the response undeliverable.
            unavailable = self._closed or self._failure is not None
            if not cancelled and not unavailable:
                self._delegate_committed.add(delegate_id)
        if cancelled or unavailable:
            with self._state_lock:
                self._finish_delegate_state_locked(delegate_id)
            return
        try:
            self._send_frame({"type": "delegate/response", "id": delegate_id, "result": response})
        except CodeModeHostProtocolError as exc:
            if response.get("status") == "error":
                self._fail_connection(exc)
                return
            try:
                self._send_frame(
                    {
                        "type": "delegate/response",
                        "id": delegate_id,
                        "result": {"status": "error", "message": str(exc)},
                    }
                )
            except CodeModeHostError as fallback_error:
                self._fail_connection(fallback_error)
        except CodeModeHostError:
            return
        finally:
            with self._state_lock:
                self._finish_delegate_state_locked(delegate_id)

    @staticmethod
    def _flatten_tool_name(tool_name: Mapping[str, Any]) -> str:
        name = tool_name.get("name")
        namespace = tool_name.get("namespace")
        if not isinstance(name, str) or not name:
            raise CodeModeHostProtocolError(f"invalid nested tool name: {tool_name!r}")
        if namespace is None:
            return name
        if not isinstance(namespace, str) or not namespace:
            raise CodeModeHostProtocolError(f"invalid nested tool namespace: {tool_name!r}")
        separator = "" if namespace.endswith("_") or name.startswith("_") else "__"
        return f"{namespace}{separator}{name}"

    def _raise_fatal_delegate_error(self, runtime_response: Mapping[str, Any]) -> None:
        payload = next(iter(runtime_response.values()))
        cell_id = payload.get("cell_id")
        with self._state_lock:
            error = self._fatal_delegate_errors_by_cell.pop(cell_id, None)
        if error is not None:
            raise error

    def _stderr_loop(self) -> None:
        process_stderr = self._process.stderr
        if process_stderr is None:
            return
        try:
            while line := process_stderr.readline():
                text = line.decode(errors="replace").rstrip()
                with self._state_lock:
                    self._stderr_tail.append(text)
                logger.debug("code-mode host stderr: %s", text)
        except (OSError, ValueError):
            return

    def _raise_if_unavailable_locked(self, *, allow_closing: bool) -> None:
        if self._failure is not None:
            raise CodeModeHostClosedError(str(self._failure)) from self._failure
        if self._closed or (self._closing and not allow_closing):
            raise CodeModeHostClosedError("code-mode host session is closed")
        returncode = self._process.poll()
        if returncode is not None:
            raise CodeModeHostClosedError(f"code-mode host exited with status {returncode}")

    def _fail_connection(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                stderr = "\n".join(self._stderr_tail)
                message = str(error)
                if stderr:
                    message = f"{message}; host stderr:\n{stderr}"
                self._failure = CodeModeHostClosedError(message)
            self._hello_ready.set()
            self._fail_pending_locked(self._failure)

    def _fail_pending_locked(self, error: BaseException) -> None:
        for pending in self._pending.values():
            if pending.error is None:
                pending.error = error
            pending.operation_ready.set()
            pending.initial_ready.set()

    def _abort_startup(self) -> None:
        with self._state_lock:
            self._closing = True
        self._stop_writer()
        self._close_process_streams()
        self._reap_process()
        self._delegate_executor.shutdown(wait=False, cancel_futures=True)
        self._writer_thread.join(timeout=self._close_timeout)
        with self._state_lock:
            self._closed = True
            self._closed_event.set()

    def _stop_writer(self) -> None:
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass

    def _close_process_streams(self) -> None:
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _reap_process(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._process.wait(timeout=self._close_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        self._process.terminate()
        try:
            self._process.wait(timeout=self._close_timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=self._close_timeout)


__all__ = [
    "CodeModeHostClosedError",
    "CodeModeHostError",
    "CodeModeHostOperationError",
    "CodeModeHostProtocolError",
    "CodeModeHostSession",
    "CodeModeHostTimeoutError",
    "MAX_FRAME_BYTES",
    "build_tool_definition",
    "find_code_mode_host",
]
