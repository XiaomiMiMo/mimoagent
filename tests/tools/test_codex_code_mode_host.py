from __future__ import annotations

import io
import os
import stat
import sys
import tarfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

import mimoagent.tools.codex.code_mode_host as host_module
from mimoagent.tools.codex.code_mode_host import (
    CodeModeHostSession,
    CodeModeHostTimeoutError,
    _write_all,
    build_tool_definition,
    find_code_mode_host,
)


def _write_fake_host(tmp_path: Path) -> Path:
    host = tmp_path / "codex-code-mode-host"
    host.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            r"""
            import json
            import struct
            import sys

            def read_exact(size):
                data = b""
                while len(data) < size:
                    chunk = sys.stdin.buffer.read(size - len(data))
                    if not chunk:
                        raise EOFError
                    data += chunk
                return data

            def receive():
                size = struct.unpack("<I", read_exact(4))[0]
                return json.loads(read_exact(size))

            def send(message):
                payload = json.dumps(message, separators=(",", ":")).encode()
                sys.stdout.buffer.write(struct.pack("<I", len(payload)) + payload)
                sys.stdout.buffer.flush()

            hello = receive()
            assert hello == {
                "type": "connection/hello",
                "supportedVersions": [1],
                "requiredCapabilities": [],
                "optionalCapabilities": ["session-cell-execution-resource-limits"],
            }
            send({
                "type": "connection/ready",
                "selectedVersion": 1,
                "capabilities": ["session-cell-execution-resource-limits"],
            })

            opened = receive()
            assert opened["type"] == "operation/request"
            assert opened["request"]["method"] == "session/open"
            session_id = opened["request"]["sessionId"]
            send({
                "type": "operation/response",
                "id": opened["id"],
                "result": {
                    "status": "ok",
                    "value": {"type": "session/ready", "sessionId": session_id},
                },
            })

            next_cell = 1
            while True:
                message = receive()
                assert message["type"] == "operation/request"
                request_id = message["id"]
                request = message["request"]
                method = request["method"]
                if method == "session/execute":
                    cell_id = str(next_cell)
                    next_cell += 1
                    send({
                        "type": "operation/response",
                        "id": request_id,
                        "result": {
                            "status": "ok",
                            "value": {"type": "execution/started", "cellId": cell_id},
                        },
                    })
                    source = request["request"]["source"]
                    tools = request["request"]["enabled_tools"]
                    assert tools and tools[0]["kind"] == "function"
                    if source == "hang":
                        cancel = receive()
                        assert cancel == {"type": "operation/cancel", "id": request_id}
                        terminate = receive()
                        assert terminate["type"] == "operation/request"
                        assert terminate["request"] == {
                            "method": "session/terminate",
                            "sessionId": session_id,
                            "cellId": cell_id,
                        }
                        send({
                            "type": "operation/response",
                            "id": terminate["id"],
                            "result": {
                                "status": "ok",
                                "value": {
                                    "type": "wait/completed",
                                    "outcome": {
                                        "LiveCell": {
                                            "Terminated": {
                                                "cell_id": cell_id,
                                                "content_items": [],
                                            },
                                        },
                                    },
                                },
                            },
                        })
                        send({"type": "cell/closed", "sessionId": session_id, "cellId": cell_id})
                        continue
                    if source == "parallel":
                        delegate_ids = [101, 102]
                    elif source == "cancel":
                        delegate_ids = [201, 202]
                    else:
                        delegate_ids = [103]
                    sent_delegates = []
                    for delegate_id in delegate_ids:
                        name = "alpha" if delegate_id in {101, 201} else "beta" if delegate_id in {102, 202} else "explode"
                        delegate_message = {
                            "type": "delegate/request",
                            "id": delegate_id,
                            "sessionId": session_id,
                            "request": {
                                "type": "tool/invoke",
                                "invocation": {
                                    "cell_id": cell_id,
                                    "runtime_tool_call_id": f"tool-{delegate_id}",
                                    "tool_name": {"name": name, "namespace": None},
                                    "tool_kind": "function",
                                    "input": {"delegate_id": delegate_id},
                                },
                            },
                        }
                        sent_delegates.append(delegate_message)
                        send(delegate_message)
                    if source == "cancel":
                        send({"type": "delegate/cancel", "id": 202})
                        expected_response_ids = {201}
                    else:
                        expected_response_ids = set(delegate_ids)
                    responses = [receive() for _ in expected_response_ids]
                    assert {response["id"] for response in responses} == expected_response_ids
                    if source in {"parallel", "cancel", "blocking", "replay"}:
                        assert all(response["result"]["status"] == "ok" for response in responses)
                        error_text = None
                    else:
                        assert responses[0]["result"]["status"] == "error"
                        error_text = responses[0]["result"]["message"]
                    if source == "replay":
                        send(sent_delegates[0])
                        continue
                    send({
                        "type": "execute/initialResponse",
                        "id": request_id,
                        "result": {
                            "status": "ok",
                            "value": {
                                "Result": {
                                    "cell_id": cell_id,
                                    "content_items": [{"type": "input_text", "text": source}],
                                    "error_text": error_text,
                                },
                            },
                        },
                    })
                    send({"type": "cell/closed", "sessionId": session_id, "cellId": cell_id})
                elif method == "session/wait":
                    cell_id = request["request"]["cell_id"]
                    runtime = {
                        "Yielded": {
                            "cell_id": cell_id,
                            "content_items": [{"type": "input_text", "text": "waited"}],
                        },
                    }
                    send({
                        "type": "operation/response",
                        "id": request_id,
                        "result": {
                            "status": "ok",
                            "value": {
                                "type": "wait/completed",
                                "outcome": {"LiveCell": runtime},
                            },
                        },
                    })
                elif method == "session/terminate":
                    cell_id = request["cellId"]
                    runtime = {"Terminated": {"cell_id": cell_id, "content_items": []}}
                    send({
                        "type": "operation/response",
                        "id": request_id,
                        "result": {
                            "status": "ok",
                            "value": {
                                "type": "wait/completed",
                                "outcome": {"LiveCell": runtime},
                            },
                        },
                    })
                elif method == "session/shutdown":
                    send({
                        "type": "operation/response",
                        "id": request_id,
                        "result": {
                            "status": "ok",
                            "value": {"type": "session/closed", "sessionId": session_id},
                        },
                    })
                    break
                else:
                    raise AssertionError(method)
            """,
        ),
        encoding="utf-8",
    )
    host.chmod(host.stat().st_mode | stat.S_IXUSR)
    return host


def _tools():
    return [
        build_tool_definition("alpha", input_schema={"type": "object"}),
        build_tool_definition("beta", input_schema={"type": "object"}),
        build_tool_definition("explode", input_schema={"type": "object"}),
    ]


def test_stdio_session_demultiplexes_parallel_delegates_and_runtime_operations(tmp_path):
    host = _write_fake_host(tmp_path)
    barrier = threading.Barrier(2)
    callbacks = []

    def invoke_tool(name, value, kind):
        callbacks.append((name, value, kind))
        barrier.wait(timeout=2)
        return {"name": name, "value": value}

    with CodeModeHostSession(host, invoke_tool, enabled_tools=_tools()) as session:
        assert session.execute("outer-call", "parallel", 1000) == {
            "Result": {
                "cell_id": "1",
                "content_items": [{"type": "input_text", "text": "parallel"}],
                "error_text": None,
            }
        }
        assert session.wait("running-cell", 50) == {
            "Yielded": {
                "cell_id": "running-cell",
                "content_items": [{"type": "input_text", "text": "waited"}],
            }
        }
        assert session.wait("running-cell", terminate=True) == {
            "Terminated": {"cell_id": "running-cell", "content_items": []}
        }
        assert "1" in session.closed_cells

    assert {name for name, _, _ in callbacks} == {"alpha", "beta"}


def test_fatal_delegate_error_is_re_raised_after_host_receives_error_response(tmp_path):
    class FatalDelegateError(Exception):
        pass

    def invoke_tool(_name, _value, _kind):
        raise FatalDelegateError("transport failed")

    with CodeModeHostSession(
        _write_fake_host(tmp_path),
        invoke_tool,
        enabled_tools=_tools(),
        fatal_delegate_errors=(FatalDelegateError,),
    ) as session:
        with pytest.raises(FatalDelegateError, match="transport failed"):
            session.execute("outer-call", "fatal", 1000)


def test_started_cell_is_terminated_when_initial_response_times_out(tmp_path):
    with CodeModeHostSession(
        _write_fake_host(tmp_path),
        lambda *_args: None,
        enabled_tools=_tools(),
        request_timeout=0.1,
    ) as session:
        with pytest.raises(CodeModeHostTimeoutError, match="initial response"):
            session.execute("outer-call", "hang")


def test_cancelled_queued_delegate_does_not_run_or_leak(tmp_path):
    callbacks = []

    def invoke_tool(name, _value, _kind):
        callbacks.append(name)
        time.sleep(0.05)
        return {"name": name}

    with CodeModeHostSession(
        _write_fake_host(tmp_path),
        invoke_tool,
        enabled_tools=_tools(),
        delegate_workers=1,
    ) as session:
        assert session.execute("outer-call", "cancel", 1000)["Result"]["error_text"] is None
        assert session._delegate_futures == {}
        assert session._cancelled_delegates == set()

    assert callbacks == ["alpha"]


def test_close_waits_for_running_delegate_to_leave_execution_boundary(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def invoke_tool(_name, _value, _kind):
        started.set()
        assert release.wait(timeout=3)
        return {"done": True}

    session = CodeModeHostSession(_write_fake_host(tmp_path), invoke_tool, enabled_tools=_tools())
    execution = threading.Thread(target=lambda: session.execute("outer-call", "blocking", 1000))
    execution.start()
    assert started.wait(timeout=2)

    closer = threading.Thread(target=session.close)
    closer.start()
    second_closer = threading.Thread(target=session.close)
    second_closer.start()
    time.sleep(0.1)
    assert closer.is_alive(), "close returned while a nested tool was still running"
    assert second_closer.is_alive(), "a concurrent close returned before cleanup completed"
    release.set()
    execution.join(timeout=3)
    closer.join(timeout=3)
    second_closer.join(timeout=3)
    assert not execution.is_alive()
    assert not closer.is_alive()
    assert not second_closer.is_alive()


def test_nonfatal_delegate_error_remains_a_script_result(tmp_path):
    def invoke_tool(_name, _value, _kind):
        raise ValueError("bad tool input")

    with CodeModeHostSession(_write_fake_host(tmp_path), invoke_tool, enabled_tools=_tools()) as session:
        result = session.execute("outer-call", "nonfatal", 1000)

    assert result["Result"]["error_text"] == "bad tool input"


def test_completed_delegate_id_cannot_be_replayed(tmp_path):
    calls = 0

    def invoke_tool(_name, _value, _kind):
        nonlocal calls
        calls += 1
        return {"ok": True}

    with CodeModeHostSession(
        _write_fake_host(tmp_path),
        invoke_tool,
        enabled_tools=_tools(),
        request_timeout=0.2,
    ) as session:
        with pytest.raises(Exception, match="duplicate delegate request"):
            session.execute("outer-call", "replay")

    assert calls == 1


def test_host_discovery_downloads_only_when_no_path_is_configured(monkeypatch, tmp_path):
    downloaded = tmp_path / "downloaded-host"
    calls = []

    def fake_download():
        calls.append("download")
        return downloaded

    monkeypatch.setattr(host_module, "_download_code_mode_host", fake_download)

    # No configured path: fall back to the release-asset download.
    assert find_code_mode_host() == downloaded
    assert find_code_mode_host(None) == downloaded
    assert calls == ["download", "download"]


def test_configured_host_path_is_used_and_never_downloads(monkeypatch, tmp_path):
    def fail_download():
        raise AssertionError("download must not run when a host_path is configured")

    monkeypatch.setattr(host_module, "_download_code_mode_host", fail_download)

    local_host = tmp_path / "codex-code-mode-host"
    local_host.write_text("#!/bin/sh\n", encoding="utf-8")
    local_host.chmod(local_host.stat().st_mode | stat.S_IXUSR)

    # Hosts without network copy the binary locally and point at it via config;
    # that path must win over the download.
    assert find_code_mode_host(str(local_host)) == local_host.resolve()


def test_host_path_env_var_is_used_and_never_downloads(monkeypatch, tmp_path):
    def fail_download():
        raise AssertionError("download must not run when MIMOAGENT_CODE_MODE_HOST_PATH is set")

    monkeypatch.setattr(host_module, "_download_code_mode_host", fail_download)
    local_host = tmp_path / "codex-code-mode-host"
    local_host.write_text("#!/bin/sh\n", encoding="utf-8")
    local_host.chmod(local_host.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(host_module.CODE_MODE_HOST_PATH_ENV, str(local_host))
    assert find_code_mode_host() == local_host.resolve()


def test_configured_host_path_that_is_not_executable_fails_clearly(tmp_path):
    host = tmp_path / "not-executable"
    host.write_text("placeholder", encoding="utf-8")
    host.chmod(host.stat().st_mode & ~os.X_OK)
    with pytest.raises(FileNotFoundError, match="configured code-mode host is not executable"):
        find_code_mode_host(str(host))


def test_host_path_must_be_executable(tmp_path):
    host = tmp_path / "not-executable"
    host.write_text("placeholder", encoding="utf-8")
    host.chmod(host.stat().st_mode & ~os.X_OK)
    with pytest.raises(FileNotFoundError):
        CodeModeHostSession(host, lambda *_args: None)


def test_write_all_handles_short_pipe_writes():
    class ShortWriter:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            chunk = bytes(data[:3])
            self.data.extend(chunk)
            return len(chunk)

    writer = ShortWriter()
    _write_all(writer, b"a full framed payload")
    assert writer.data == b"a full framed payload"


def test_release_host_download_is_extracted_and_cached(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        content = b"#!/bin/sh\n"
        info = tarfile.TarInfo("codex-code-mode-host-x86_64-unknown-linux-musl")
        info.mode = 0o755
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    payload_bytes = payload.getvalue()

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload_bytes))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        assert url == host_module.DEFAULT_CODE_MODE_HOST_ARCHIVE_URL
        return Response(payload_bytes)

    monkeypatch.setenv("MIMOAGENT_CODE_MODE_HOST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv(host_module.CODE_MODE_HOST_URL_ENV, raising=False)
    monkeypatch.setattr(host_module.urllib.request, "urlopen", fake_urlopen)

    first = host_module._download_code_mode_host()
    second = host_module._download_code_mode_host()
    assert first == second
    assert first.name == "codex-code-mode-host-x86_64-unknown-linux-musl"
    assert first.stat().st_mode & os.X_OK
    assert first.read_bytes() == b"#!/bin/sh\n"
    assert len(calls) == 1
