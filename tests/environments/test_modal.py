"""ModalEnvironment against a fake ``modal`` SDK.

The SDK is replaced in ``sys.modules`` so these tests need neither Modal
credentials nor network; they pin down what the backend sends to the SDK
(image, sandbox options, exec argv, stdin framing) and how it classifies
what comes back (exit codes, deadlines, stream errors).
"""

import io
import sys
import tarfile
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from mimoagent.environments import TransportError, get_environment_class
from mimoagent.environments.modal import _APP_CACHE, ModalEnvironment, ModalEnvironmentConfig

# -- fake SDK ------------------------------------------------------------------------


class FakeReader:
    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error


class FakeWriter:
    def __init__(self):
        self.pending = bytearray()
        self.sent = bytearray()
        self.drains = 0
        self.eof = False
        self.eof_sent = False
        self.max_pending = 0

    def write(self, data):
        self.pending.extend(data)
        self.max_pending = max(self.max_pending, len(self.pending))

    def drain(self):
        self.drains += 1
        self.sent.extend(self.pending)
        self.pending.clear()
        if self.eof:
            self.eof_sent = True

    def write_eof(self):
        self.eof = True


class FakeProcess:
    def __init__(self, rc=0, stdout=b"", stderr=b"", *, stdout_error=None, stderr_error=None, wait_error=None):
        self.stdout = FakeReader([stdout] if stdout else [], stdout_error)
        self.stderr = FakeReader([stderr] if stderr else [], stderr_error)
        self.stdin = FakeWriter()
        self.rc = rc
        self.wait_error = wait_error

    def wait(self):
        if self.wait_error is not None:
            raise self.wait_error
        return self.rc


class FakeSandbox:
    """Answers ``exec`` from ``handler(args, kwargs) -> FakeProcess``; the
    readiness probe issued by ``start`` is answered with rc 0 unless the
    handler claims it."""

    def __init__(self, handler=None, object_id="sb-test1234"):
        self.handler = handler
        self.object_id = object_id
        self.execs = []
        self.tags = None
        self.terminated = False

    def exec(self, *args, **kwargs):
        self.execs.append((args, kwargs))
        if self.handler is not None:
            proc = self.handler(args, kwargs)
            if proc is not None:
                return proc
        return FakeProcess(0)

    def set_tags(self, tags):
        self.tags = dict(tags)

    def terminate(self):
        self.terminated = True


class FakeModal:
    """Stand-in for the ``modal`` package: records lookups and creations."""

    def __init__(self):
        fake = self
        self.lookups = []
        self.creates = []
        self.images = []
        self.secrets = []
        self.create_results = []  # FakeSandbox instances or exceptions, consumed in order
        self.output_enabled = 0

        class _Error(Exception):
            pass

        exc = types.ModuleType("modal.exception")
        exc.Error = _Error
        for name in (
            "AuthError",
            "InvalidError",
            "NotFoundError",
            "PermissionDeniedError",
            "ImageBuildError",
            "ExecTimeoutError",
            "SandboxTerminatedError",
            "ConnectionError",
        ):
            setattr(exc, name, type(name, (_Error,), {}))
        self.exception = exc

        class App:
            @staticmethod
            def lookup(name, *, environment_name=None, create_if_missing=False):
                fake.lookups.append((name, environment_name, create_if_missing))
                return ("app", name, environment_name)

        class Image:
            @staticmethod
            def from_registry(tag, secret=None, *, add_python=None):
                img = ("registry", tag, secret, add_python)
                fake.images.append(img)
                return img

            @staticmethod
            def from_id(image_id):
                img = ("id", image_id)
                fake.images.append(img)
                return img

        class Secret:
            @staticmethod
            def from_name(name, *, environment_name=None):
                s = ("secret", name, environment_name)
                fake.secrets.append(s)
                return s

        class Sandbox:
            @staticmethod
            def create(*args, **kwargs):
                fake.creates.append((args, kwargs))
                result = fake.create_results.pop(0) if fake.create_results else FakeSandbox()
                if isinstance(result, Exception):
                    raise result
                return result

        @contextmanager
        def enable_output():
            fake.output_enabled += 1
            yield

        self.App = App
        self.Image = Image
        self.Secret = Secret
        self.Sandbox = Sandbox
        self.enable_output = enable_output


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_modal(monkeypatch):
    fake = FakeModal()
    module = types.ModuleType("modal")
    for attr in ("App", "Image", "Secret", "Sandbox", "enable_output", "exception"):
        setattr(module, attr, getattr(fake, attr))
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setitem(sys.modules, "modal.exception", fake.exception)
    _APP_CACHE.clear()
    yield fake
    _APP_CACHE.clear()


@pytest.fixture
def clock():
    c = FakeClock()
    with patch("mimoagent.environments.modal.time", c):
        yield c


def make_env(**kwargs) -> ModalEnvironment:
    kwargs.setdefault("image", "example/test:latest")
    return ModalEnvironment(**kwargs)


def started_env(handler=None, **kwargs) -> tuple[ModalEnvironment, FakeSandbox]:
    """An environment wired to a FakeSandbox without going through start()."""
    env = make_env(**kwargs)
    sb = FakeSandbox(handler)
    env.sandbox = sb
    env.sandbox_id = sb.object_id
    return env, sb


def bash_script(args) -> str:
    """The script handed to ``/bin/bash -lc`` in an exec argv."""
    assert args[-3:-1] == ("/bin/bash", "-lc"), args
    return args[-1]


# -- registry / config ------------------------------------------------------------------


def test_registered_under_modal_shorthand():
    assert get_environment_class("modal") is ModalEnvironment


def test_config_defaults():
    cfg = ModalEnvironmentConfig(image="python:3.12-slim")
    assert cfg.cwd == "/testbed"
    assert cfg.timeout == 30
    assert cfg.sandbox_timeout == 7200
    assert cfg.entrypoint == ["sleep", "infinity"]
    assert cfg.tags == {"app": "mimoagent"}
    assert cfg.block_network is False
    assert cfg.idle_timeout is None and cfg.cpu is None and cfg.memory is None and cfg.gpu is None


def test_template_vars_round_trip_into_constructor():
    """deepswe rebuilds a sibling environment from get_template_vars()."""
    env = make_env(cwd="/app", env={"A": "1"}, secrets=["s1"], tags={"app": "x"})
    clone = ModalEnvironment(**env.get_template_vars())
    assert clone.get_template_vars() == env.get_template_vars()


def test_import_error_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    env = make_env()
    with pytest.raises(ImportError, match="uv sync --extra modal"):
        env.start()


# -- start ---------------------------------------------------------------------------------


def test_start_creates_sandbox_from_registry_image(fake_modal, clock):
    env = make_env(cwd="/testbed", env={"PYTHONDONTWRITEBYTECODE": "1"})
    env.instance_id = "django__django-11099"
    env.start()

    assert fake_modal.lookups == [("mimoagent", None, True)]
    assert fake_modal.images == [("registry", env.config.image, None, None)]
    ((args, kwargs),) = fake_modal.creates
    assert args == ("sleep", "infinity")
    assert kwargs["app"] == ("app", "mimoagent", None)
    assert kwargs["image"] == fake_modal.images[0]
    assert kwargs["timeout"] == 7200
    assert kwargs["env"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert kwargs["block_network"] is False
    # unset knobs are not passed as None
    for key in ("idle_timeout", "cpu", "memory", "gpu", "region", "cloud", "secrets", "environment_name"):
        assert key not in kwargs
    sb = env.sandbox
    assert env.sandbox_id == "sb-test1234"
    assert sb.tags == {"app": "mimoagent", "instance_id": "django__django-11099"}
    # readiness probe: a login shell that must succeed
    ((probe_args, probe_kwargs),) = sb.execs
    assert probe_args == ("/bin/bash", "-lc", "true")
    assert probe_kwargs["timeout"] == ModalEnvironment._READY_TIMEOUT_S


def test_start_passes_resource_and_placement_knobs(fake_modal, clock):
    env = make_env(
        app_name="rollouts",
        environment_name="staging",
        sandbox_timeout=3600,
        idle_timeout=600,
        cpu=2.0,
        memory=8192,
        gpu="T4",
        region="us-east",
        cloud="aws",
        block_network=True,
        cidr_allowlist=["10.0.0.0/8"],
        secrets=["hf-token", "gateway-key"],
        registry_secret="dockerhub",
        add_python="3.12",
        entrypoint=["/bin/sh", "-c", "sleep 1d"],
    )
    env.start()

    assert fake_modal.lookups == [("rollouts", "staging", True)]
    assert fake_modal.images == [("registry", env.config.image, ("secret", "dockerhub", "staging"), "3.12")]
    ((args, kwargs),) = fake_modal.creates
    assert args == ("/bin/sh", "-c", "sleep 1d")
    assert kwargs["timeout"] == 3600
    assert kwargs["idle_timeout"] == 600
    assert kwargs["cpu"] == 2.0
    assert kwargs["memory"] == 8192
    assert kwargs["gpu"] == "T4"
    assert kwargs["region"] == "us-east"
    assert kwargs["cloud"] == "aws"
    assert kwargs["block_network"] is True
    assert kwargs["outbound_cidr_allowlist"] == ["10.0.0.0/8"]
    assert kwargs["environment_name"] == "staging"
    assert kwargs["secrets"] == [("secret", "hf-token", "staging"), ("secret", "gateway-key", "staging")]


def test_start_uses_prebuilt_image_id(fake_modal, clock):
    env = make_env(image="im-abc123")
    env.start()
    assert fake_modal.images == [("id", "im-abc123")]


def test_start_forwards_host_env(fake_modal, clock, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    env = make_env(forward_env=["HTTPS_PROXY", "NO_PROXY"], env={"HTTPS_PROXY": "override"})
    env.start()
    ((_, kwargs),) = fake_modal.creates
    # only variables set on the host are forwarded; env wins on conflict
    assert kwargs["env"] == {"HTTPS_PROXY": "override"}


def test_start_caches_app_lookup(fake_modal, clock):
    make_env().start()
    make_env().start()
    assert len(fake_modal.lookups) == 1


def test_start_verbose_enables_output(fake_modal, clock):
    make_env(verbose=True).start()
    assert fake_modal.output_enabled == 1


def test_start_retries_transient_create_errors(fake_modal, clock):
    fake_modal.create_results = [RuntimeError("503"), fake_modal.exception.ConnectionError("reset"), FakeSandbox()]
    env = make_env()
    env.start()
    assert len(fake_modal.creates) == 3
    assert env.sandbox is not None


def test_start_does_not_retry_permanent_errors(fake_modal, clock):
    fake_modal.create_results = [fake_modal.exception.AuthError("bad token")]
    with pytest.raises(fake_modal.exception.AuthError):
        make_env().start()
    assert len(fake_modal.creates) == 1

    fake_modal.create_results = [fake_modal.exception.ImageBuildError("not linux/amd64")]
    with pytest.raises(fake_modal.exception.ImageBuildError):
        make_env().start()
    assert len(fake_modal.creates) == 2


def test_start_terminates_sandbox_when_probe_fails(fake_modal, clock):
    def handler(args, kwargs):
        return FakeProcess(127, stderr=b"/bin/bash: not found\n")

    sb = FakeSandbox(handler)
    fake_modal.create_results = [sb]
    env = make_env()
    with pytest.raises(RuntimeError, match="not usable"):
        env.start()
    assert sb.terminated
    assert env.sandbox is None


def test_start_terminates_sandbox_when_probe_raises(fake_modal, clock):
    def handler(args, kwargs):
        raise fake_modal.exception.SandboxTerminatedError("gone")

    sb = FakeSandbox(handler)
    fake_modal.create_results = [sb]
    with pytest.raises(fake_modal.exception.SandboxTerminatedError):
        make_env().start()
    assert sb.terminated


def test_tagging_failure_is_not_fatal(fake_modal, clock):
    class NoTags(FakeSandbox):
        def set_tags(self, tags):
            raise RuntimeError("tags unsupported")

    fake_modal.create_results = [NoTags()]
    env = make_env()
    env.start()
    assert env.sandbox is not None


# -- execute -------------------------------------------------------------------------------


def test_execute_argv_env_and_timeout(fake_modal, clock, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    env, sb = started_env(
        lambda a, k: FakeProcess(0, b"hello\n"), cwd="/testbed", env={"A": "1"}, forward_env=["HF_TOKEN"]
    )
    res = env.execute("echo hello", timeout=42)

    assert res == {"output": "hello\n", "returncode": 0, "reason": "ok"}
    ((args, kwargs),) = sb.execs
    assert args[:2] == ("timeout", "42")
    assert bash_script(args) == "exec 2>&1 </dev/null\ncd /testbed && echo hello"
    assert kwargs["timeout"] == 42 + ModalEnvironment._EXEC_GRACE_S
    assert kwargs["env"] == {"HF_TOKEN": "hf_x", "A": "1"}
    assert kwargs["text"] is False


def test_execute_uses_config_cwd_and_timeout_by_default(fake_modal, clock):
    env, sb = started_env(cwd="/app", timeout=7)
    env.execute("pwd")
    ((args, _),) = sb.execs
    assert args[1] == "7"
    assert "cd /app && pwd" in bash_script(args)
    env.execute("pwd", cwd="/other dir")
    assert "cd '/other dir' && pwd" in bash_script(sb.execs[1][0])


def test_execute_nonzero_rc_is_ok_reason(fake_modal, clock):
    env, _ = started_env(lambda a, k: FakeProcess(3, b"", b"boom\n"))
    res = env.execute("false")
    assert res == {"output": "boom\n", "returncode": 3, "reason": "ok"}


def test_execute_decodes_invalid_utf8(fake_modal, clock):
    env, _ = started_env(lambda a, k: FakeProcess(0, b"ok \xff\xfe\n"))
    assert env.execute("cat bin")["output"] == "ok ��\n"


def test_execute_rc124_is_pod_timeout(fake_modal, clock):
    env, _ = started_env(lambda a, k: FakeProcess(124, b"partial\n"))
    res = env.execute("sleep 999", timeout=5)
    assert res["returncode"] == 124
    assert res["reason"] == "pod_timeout"
    assert res["output"].startswith("partial\n")
    assert "timed out (pod) after 5s" in res["output"]


def test_execute_modal_deadline_is_client_timeout(fake_modal, clock):
    env, _ = started_env(lambda a, k: FakeProcess(-1, b"partial\n"))
    res = env.execute("hang", timeout=5)
    assert res["returncode"] is None
    assert res["reason"] == "client_timeout"
    assert "timed out (client) after 5s" in res["output"]


def test_execute_stream_exec_timeout_is_client_timeout(fake_modal, clock):
    err = fake_modal.exception.ExecTimeoutError("deadline")
    env, _ = started_env(lambda a, k: FakeProcess(0, b"some\n", stdout_error=err))
    res = env.execute("hang", timeout=5)
    assert res["reason"] == "client_timeout"
    assert res["output"].startswith("some\n")


def test_execute_other_stream_error_is_transport_error(fake_modal, clock):
    err = fake_modal.exception.ConnectionError("router reset")
    env, _ = started_env(lambda a, k: FakeProcess(0, b"", stderr_error=err))
    res = env.execute("ls")
    assert res == {
        "output": "Sandbox exec stream failed: ConnectionError: router reset",
        "returncode": None,
        "reason": "transport_error",
    }


def test_execute_exec_failure_is_transport_error(fake_modal, clock):
    def handler(args, kwargs):
        raise fake_modal.exception.SandboxTerminatedError("sandbox gone")

    env, _ = started_env(handler)
    res = env.execute("ls")
    assert res["reason"] == "transport_error"
    assert res["returncode"] is None
    assert "SandboxTerminatedError" in res["output"]


def test_execute_wait_failure_is_transport_error(fake_modal, clock):
    env, _ = started_env(lambda a, k: FakeProcess(0, b"out", wait_error=RuntimeError("lost")))
    res = env.execute("ls")
    assert res["reason"] == "transport_error"
    assert res["output"] == "out"


def test_execute_raises_transport_error_when_configured(fake_modal, clock):
    def handler(args, kwargs):
        raise RuntimeError("down")

    env, _ = started_env(handler, raise_on_transport_error=True)
    with pytest.raises(TransportError):
        env.execute("ls")


def test_execute_keeps_only_output_tail(fake_modal, clock, monkeypatch):
    monkeypatch.setattr(ModalEnvironment, "_MAX_OUTPUT_BYTES", 8)

    def handler(args, kwargs):
        proc = FakeProcess(0)
        proc.stdout = FakeReader([b"aaaa", b"bbbb", b"cccc"])
        return proc

    env, _ = started_env(handler)
    assert env.execute("spam")["output"] == "bbbbcccc"


def test_execute_requires_started_sandbox(fake_modal, clock):
    with pytest.raises(AssertionError):
        make_env().execute("ls")


# -- copy_to / copy_out -----------------------------------------------------------------


def _received_tar(sb: FakeSandbox, proc: FakeProcess):
    return tarfile.open(fileobj=io.BytesIO(bytes(proc.stdin.sent)), mode="r:*")


def test_copy_to_streams_tar_on_stdin(fake_modal, clock, tmp_path):
    src = tmp_path / "payload.txt"
    src.write_text("hello sandbox\n")
    procs = []

    def handler(args, kwargs):
        proc = FakeProcess(0)
        procs.append(proc)
        return proc

    env, sb = started_env(handler)
    env.copy_to(str(src), "/tmp/staging/renamed.txt")

    ((args, kwargs),) = sb.execs
    assert args[:2] == ("/bin/sh", "-c")
    (proc,) = procs
    tar_len = len(bytes(proc.stdin.sent))
    assert args[2] == f"mkdir -p /tmp/staging && head -c {tar_len} | tar xmf - -C /tmp/staging"
    assert kwargs["timeout"] == 300
    assert proc.stdin.eof and proc.stdin.eof_sent
    with _received_tar(sb, proc) as tf:
        assert tf.getnames() == ["renamed.txt"]
        assert tf.extractfile("renamed.txt").read() == b"hello sandbox\n"


def test_copy_to_directory_and_chunked_stdin(fake_modal, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(ModalEnvironment, "_STDIN_CHUNK_BYTES", 4096)
    src = tmp_path / "tree"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_bytes(b"x" * 20000)
    (src / "sub" / "b.txt").write_text("b")
    procs = []

    def handler(args, kwargs):
        proc = FakeProcess(0)
        procs.append(proc)
        return proc

    env, sb = started_env(handler)
    env.copy_to(str(src), "/work/tree")

    (proc,) = procs
    assert proc.stdin.max_pending <= 4096
    assert proc.stdin.drains > 5
    with _received_tar(sb, proc) as tf:
        assert sorted(tf.getnames()) == ["tree", "tree/a.txt", "tree/sub", "tree/sub/b.txt"]


def test_copy_to_retries_then_raises(fake_modal, clock, tmp_path):
    src = tmp_path / "f"
    src.write_text("x")
    replies = [FakeProcess(1, stderr=b"tar: short read"), FakeProcess(0)]
    env, sb = started_env(lambda a, k: replies.pop(0))
    env.copy_to(str(src), "/tmp/f")
    assert len(sb.execs) == 2

    env, sb = started_env(lambda a, k: FakeProcess(2, stderr=b"disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        env.copy_to(str(src), "/tmp/f", max_retries=2)
    assert len(sb.execs) == 2


def test_copy_to_missing_source(fake_modal, clock, tmp_path):
    env, _ = started_env()
    with pytest.raises(FileNotFoundError):
        env.copy_to(str(tmp_path / "nope"), "/tmp/nope")


def _tar_bytes(entries: dict[str, bytes], root: str | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        if root:
            info = tarfile.TarInfo(root)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_copy_out_file(fake_modal, clock, tmp_path):
    payload = _tar_bytes({"result.json": b'{"ok": true}'})
    env, sb = started_env(lambda a, k: FakeProcess(0, payload))
    dest = tmp_path / "out" / "renamed.json"
    env.copy_out("/testbed/result.json", str(dest))

    ((args, _),) = sb.execs
    assert args == ("/bin/sh", "-c", "test -e /testbed/result.json && tar cf - -C /testbed result.json")
    assert dest.read_bytes() == b'{"ok": true}'


def test_copy_out_directory_replaces_existing(fake_modal, clock, tmp_path):
    payload = _tar_bytes({"logs/a.log": b"a", "logs/b.log": b"b"}, root="logs")
    env, _ = started_env(lambda a, k: FakeProcess(0, payload))
    dest = tmp_path / "logs"
    dest.mkdir()
    (dest / "stale").write_text("old")
    env.copy_out("/root/.claude/logs", str(dest))
    assert sorted(p.name for p in dest.iterdir()) == ["a.log", "b.log"]


def test_copy_out_missing_source_raises_after_retries(fake_modal, clock):
    env, sb = started_env(lambda a, k: FakeProcess(1))
    with pytest.raises(RuntimeError, match="tar failed"):
        env.copy_out("/nope", "/tmp/nope", max_retries=3)
    assert len(sb.execs) == 3


def test_copy_out_is_not_truncated_by_output_cap(fake_modal, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(ModalEnvironment, "_MAX_OUTPUT_BYTES", 16)
    payload = _tar_bytes({"big.bin": b"z" * 5000})
    env, _ = started_env(lambda a, k: FakeProcess(0, payload))
    dest = tmp_path / "big.bin"
    env.copy_out("/testbed/big.bin", str(dest))
    assert dest.stat().st_size == 5000


# -- detached / cleanup ------------------------------------------------------------------


def test_execute_detached_uses_shared_poller(fake_modal, clock):
    env, _ = started_env()
    scripted = iter(
        [
            {"output": "__MIMO_DETACHED_LAUNCHED\n", "returncode": 0, "reason": "ok"},
            {"output": f"{ModalEnvironment._DETACHED_MARK_RC}0\n", "returncode": 0, "reason": "ok"},
            {"output": "harness done\n", "returncode": 0, "reason": "ok"},
        ]
    )
    staged = {}
    env.execute = lambda command, cwd="", timeout=None: next(scripted)
    env.copy_to = lambda src, dst, **kw: staged.__setitem__(dst, Path(src).read_text())
    with patch("mimoagent.environments.detached.time", clock):
        res = env.execute_detached("run-harness", timeout=3600, idle_files=["/tmp/h.log"], idle_timeout=600)
    assert res == {"output": "harness done\n", "returncode": 0, "reason": "ok"}
    ((path, script),) = staged.items()
    assert path.endswith("/run.sh")
    assert "cd /testbed && run-harness" in script


def test_execute_detached_requires_started_sandbox(fake_modal, clock):
    with pytest.raises(AssertionError):
        make_env().execute_detached("x", timeout=1)


def test_cleanup_terminates_and_is_idempotent(fake_modal, clock):
    env, sb = started_env()
    env.cleanup()
    assert sb.terminated
    assert env.sandbox is None and env.sandbox_id is None
    env.cleanup()  # no sandbox: no-op


def test_cleanup_swallows_terminate_errors(fake_modal, clock):
    class Flaky(FakeSandbox):
        def terminate(self):
            raise RuntimeError("already gone")

    env = make_env()
    env.sandbox = Flaky()
    env.sandbox_id = env.sandbox.object_id
    env.cleanup()
    assert env.sandbox is None
