"""Modal Sandbox environment for executing commands in modal.com Sandboxes.

Uses the ``modal`` SDK (``uv sync --extra modal``). Authentication is the
SDK's own: ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` in the environment, or
the ``~/.modal.toml`` written by ``modal token new`` (``MODAL_PROFILE`` selects
a profile). Each environment instance creates one Sandbox from the task's
registry image and terminates it in ``cleanup``.
"""

import io
import os
import random
import shlex
import shutil
import tarfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.environments.detached import DetachedExecMixin
from mimoagent.utils.log import get_logger as _get_logger


def _lazy_import_modal():
    try:
        import modal
        import modal.exception  # noqa: F401  (make ``modal.exception`` an attribute)
    except ImportError:
        raise ImportError("ModalEnvironment requires the modal SDK. Install it with: uv sync --extra modal")
    return modal


@dataclass
class ModalEnvironmentConfig:
    image: str
    """Registry image the sandbox runs (``docker.io`` / any ``linux/amd64``
    registry tag, pulled and cached by Modal), or a Modal image id (``im-...``)
    for a prebuilt or snapshotted image."""

    cwd: str = "/testbed"
    """Working directory in which to execute commands."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables set for every command."""

    forward_env: list[str] = field(default_factory=list)
    """Host environment variables to forward (only when set on the host).
    In case of conflict with ``env``, the ``env`` variables take precedence."""

    timeout: int = 30
    """Default command timeout in seconds."""

    app_name: str = field(default_factory=lambda: os.getenv("MODAL_APP_NAME", "mimoagent"))
    """Modal App the sandboxes are created under (looked up, created when
    missing). Default: ``$MODAL_APP_NAME`` or ``mimoagent``."""

    environment_name: str | None = None
    """Modal environment (workspace sub-namespace). ``None`` uses the SDK
    default (``$MODAL_ENVIRONMENT`` / the active profile)."""

    sandbox_timeout: int = 7200
    """Maximum sandbox lifetime in seconds (Modal caps it at 24 hours)."""

    idle_timeout: int | None = None
    """Terminate the sandbox after this many seconds without a running exec.
    ``None`` keeps it alive until ``sandbox_timeout`` or ``cleanup``. Detached
    runs probe the sandbox every 20 seconds, so keep this well above that."""

    cpu: float | None = None
    """CPU cores requested (fractional allowed). ``None`` uses Modal's default."""

    memory: int | None = None
    """Memory in MiB. ``None`` uses Modal's default."""

    gpu: str | None = None
    """GPU spec such as ``T4`` or ``A100:2``. ``None`` for CPU-only sandboxes."""

    region: str | list[str] | None = None
    """Region(s) to schedule the sandbox in, e.g. ``us-east``."""

    cloud: str | None = None
    """Cloud provider to schedule on (``aws``, ``gcp``, ``oci``, ...)."""

    block_network: bool = False
    """Run the sandbox without outbound network access. Blackbox agents install
    their CLI at setup time and need the network unless a payload is used."""

    cidr_allowlist: list[str] | None = None
    """Outbound CIDR allowlist; ``None`` allows every destination."""

    secrets: list[str] = field(default_factory=list)
    """Names of Modal Secrets whose keys are injected as environment variables."""

    registry_secret: str | None = None
    """Name of a Modal Secret holding ``REGISTRY_USERNAME`` / ``REGISTRY_PASSWORD``
    for pulling ``image`` from a private registry."""

    add_python: str | None = None
    """Python version Modal adds to images that ship without one (``"3.12"``).
    Task images normally carry their own interpreter; leave unset."""

    entrypoint: list[str] = field(default_factory=lambda: ["sleep", "infinity"])
    """Sandbox entrypoint; it only has to keep the sandbox alive, commands run
    through ``exec``."""

    tags: dict[str, str] = field(default_factory=lambda: {"app": "mimoagent"})
    """Tags set on the sandbox (``modal.Sandbox.list(tags=...)`` filters on them);
    the instance id is added when known."""

    verbose: bool = False
    """Print Modal's image build / sandbox creation output to the console."""

    raise_on_transport_error: bool = False
    """Raise TransportError on connection failures (the framework escalates it to InfraError)."""

    answer_leak_blocklist: list[str] | None = None
    """Domains to blackhole (``0.0.0.0`` in ``/etc/hosts``) at the end of
    environment setup; see ``KubernetesEnvironmentConfig.answer_leak_blocklist``."""


class _ExecOutcome:
    """Result of one ``Sandbox.exec``: exit code plus the drained streams."""

    __slots__ = ("returncode", "stdout", "stderr", "stream_error")

    def __init__(self, returncode: int | None, stdout: bytes, stderr: bytes, stream_error: Exception | None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stream_error = stream_error


_APP_CACHE: dict[tuple[str, str | None], Any] = {}
_APP_CACHE_LOCK = threading.Lock()


class ModalEnvironment(DetachedExecMixin):
    """Modal Sandbox execution environment."""

    # Maximum bytes to keep for command output per exec/copy call (50 MB).
    # Output beyond this is shed from the head so only the tail survives.
    _MAX_OUTPUT_BYTES: int = 50 * 1024 * 1024
    # Bytes queued per stdin drain; the SDK rejects single buffers above 16 MiB.
    _STDIN_CHUNK_BYTES: int = 1 << 20
    # Exec deadline margin past the in-sandbox ``timeout`` so the sandbox
    # reports rc=124 itself before Modal's own deadline (rc=-1) steps in.
    _EXEC_GRACE_S: int = 5
    # How long the first probe may wait for the sandbox to be scheduled and
    # its container started before ``start`` gives up.
    _READY_TIMEOUT_S: int = 900
    _DETACHED_LOG_TAG = "sandbox_exec_detached"

    @property
    def logger(self):
        return _get_logger()

    def __init__(self, *, config_class: type = ModalEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.sandbox = None
        self.sandbox_id: str | None = None
        self.instance_id: str | None = None
        self._sandbox_start_time: float | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self):
        """Create the sandbox and wait until it executes commands."""
        modal = _lazy_import_modal()
        start_t0 = time.time()

        app = self._lookup_app(modal)
        image = self._build_image(modal)
        create_kwargs = self._sandbox_create_kwargs(modal, app, image)

        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Creating sandbox: app={self.config.app_name} image={self.config.image}{id_tag}")

        # Stagger creation a little so a batch does not hit the API in lockstep.
        time.sleep(random.uniform(0, 3))

        max_retries = 5
        base_backoff = 5
        for attempt in range(1, max_retries + 1):
            try:
                if self.config.verbose:
                    with modal.enable_output():
                        self.sandbox = modal.Sandbox.create(*self.config.entrypoint, **create_kwargs)
                else:
                    self.sandbox = modal.Sandbox.create(*self.config.entrypoint, **create_kwargs)
                break
            except Exception as e:
                if attempt < max_retries and not self._is_permanent_create_error(modal, e):
                    backoff = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 3)
                    self.logger.warning(
                        f"Create sandbox attempt {attempt}/{max_retries} failed "
                        f"({type(e).__name__}: {e}), retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                self.logger.error(f"Failed to create sandbox (attempt {attempt}/{max_retries}): {e}")
                raise

        self.sandbox_id = self.sandbox.object_id
        create_duration = time.time() - start_t0
        self._set_tags()

        # The first exec blocks until the container is scheduled and running;
        # it also surfaces images without /bin/bash or coreutils right here
        # instead of at the agent's first tool call.
        ready_t0 = time.time()
        try:
            probe = self._exec(["/bin/bash", "-lc", "true"], timeout=self._READY_TIMEOUT_S)
        except Exception:
            self.cleanup()
            raise
        if probe.returncode != 0:
            detail = (probe.stdout + probe.stderr).decode("utf-8", "replace")[-400:]
            if probe.stream_error is not None:
                detail = f"{detail} ({type(probe.stream_error).__name__}: {probe.stream_error})"
            self.cleanup()
            raise RuntimeError(
                f"Sandbox {self.sandbox_id} is not usable (rc={probe.returncode} from /bin/bash -lc true): {detail}"
            )
        self._sandbox_start_time = time.time()
        self.logger.info(
            f"Sandbox {self.sandbox_id} ready | total={time.time() - start_t0:.1f}s "
            f"create={create_duration:.1f}s wait_ready={time.time() - ready_t0:.1f}s"
        )

    def _lookup_app(self, modal):
        key = (self.config.app_name, self.config.environment_name)
        with _APP_CACHE_LOCK:
            app = _APP_CACHE.get(key)
            if app is None:
                app = modal.App.lookup(
                    self.config.app_name,
                    environment_name=self.config.environment_name,
                    create_if_missing=True,
                )
                _APP_CACHE[key] = app
        return app

    def _build_image(self, modal):
        if self.config.image.startswith("im-"):
            return modal.Image.from_id(self.config.image)
        kwargs: dict[str, Any] = {}
        if self.config.registry_secret:
            kwargs["secret"] = modal.Secret.from_name(
                self.config.registry_secret, environment_name=self.config.environment_name
            )
        if self.config.add_python:
            kwargs["add_python"] = self.config.add_python
        return modal.Image.from_registry(self.config.image, **kwargs)

    def _sandbox_create_kwargs(self, modal, app, image) -> dict[str, Any]:
        """Keyword arguments for ``modal.Sandbox.create``; unset knobs are left
        to the SDK defaults rather than passed as ``None``."""
        kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": self.config.sandbox_timeout,
            "block_network": self.config.block_network,
        }
        env_vars = self._build_env_vars()
        if env_vars:
            kwargs["env"] = env_vars
        if self.config.environment_name:
            kwargs["environment_name"] = self.config.environment_name
        if self.config.idle_timeout is not None:
            kwargs["idle_timeout"] = self.config.idle_timeout
        if self.config.cpu is not None:
            kwargs["cpu"] = self.config.cpu
        if self.config.memory is not None:
            kwargs["memory"] = self.config.memory
        if self.config.gpu:
            kwargs["gpu"] = self.config.gpu
        if self.config.region:
            kwargs["region"] = self.config.region
        if self.config.cloud:
            kwargs["cloud"] = self.config.cloud
        if self.config.cidr_allowlist:
            kwargs["outbound_cidr_allowlist"] = list(self.config.cidr_allowlist)
        if self.config.secrets:
            kwargs["secrets"] = [
                modal.Secret.from_name(name, environment_name=self.config.environment_name)
                for name in self.config.secrets
            ]
        return kwargs

    @staticmethod
    def _is_permanent_create_error(modal, exc: Exception) -> bool:
        """Errors that no retry can fix: bad credentials, unknown image or
        secret, an image that does not build, or a rejected request."""
        names = ("AuthError", "PermissionDeniedError", "NotFoundError", "InvalidError", "ImageBuildError")
        permanent = tuple(t for t in (getattr(modal.exception, n, None) for n in names) if t is not None)
        return isinstance(exc, permanent)

    def _set_tags(self) -> None:
        tags = {str(k): str(v) for k, v in self.config.tags.items()}
        if self.instance_id:
            tags["instance_id"] = str(self.instance_id)
        if not tags:
            return
        try:
            self.sandbox.set_tags(tags)
        except Exception as e:
            self.logger.warning(f"Could not tag sandbox {self.sandbox_id}: {e}")

    def cleanup(self):
        """Terminate the sandbox."""
        if self.sandbox is None:
            return
        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Cleaning up sandbox: {self.sandbox_id}{id_tag}")
        cleanup_start = time.time()
        try:
            self.sandbox.terminate()
        except Exception as e:
            self.logger.warning(f"Sandbox cleanup error for {self.sandbox_id}: {e}")
        finally:
            lifetime = f"{time.time() - self._sandbox_start_time:.1f}s" if self._sandbox_start_time else "unknown"
            self.logger.info(
                f"Sandbox {self.sandbox_id} terminated in {time.time() - cleanup_start:.1f}s (lifetime={lifetime})"
            )
            self.sandbox = None
            self.sandbox_id = None

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    # ── exec plumbing ───────────────────────────────────────────────────────

    def _build_env_vars(self) -> dict[str, str]:
        """Forwarded host variables first, then the configured ``env`` (which wins)."""
        env_vars: dict[str, str] = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env_vars[key] = value
        env_vars.update(self.config.env)
        return env_vars

    def _return_transport_error(self, output: str) -> dict[str, Any]:
        if self.config.raise_on_transport_error:
            raise TransportError(output)
        return {"output": output, "returncode": None, "reason": "transport_error"}

    def _detached_assert_started(self) -> None:
        assert self.sandbox is not None, "Sandbox not started"

    def _detached_target(self) -> str:
        return f"sandbox={self.sandbox_id}"

    @staticmethod
    def _trim_chunks(chunks: list[bytes], total: int, max_bytes: int) -> int:
        """Drop leading chunks so the joined size is at most ``max_bytes``."""
        while chunks and total > max_bytes:
            total -= len(chunks.pop(0))
        return total

    def _drain_stream(self, reader, max_bytes: int | None) -> tuple[bytes, Exception | None]:
        """Read a stream to EOF, keeping at most the last ``max_bytes`` (all of
        it when ``None``).

        A read error is returned rather than raised so the caller can still
        collect the exit code and the other stream.
        """
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in reader:
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", "replace")
                chunks.append(chunk)
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    total = self._trim_chunks(chunks, total, max_bytes)
        except Exception as e:
            return b"".join(chunks), e
        return b"".join(chunks), None

    def _exec(
        self,
        argv: list[str],
        *,
        timeout: int,
        stdin: bytes | None = None,
        max_output_bytes: int | None = None,
    ) -> _ExecOutcome:
        """Run ``argv`` in the sandbox and drain both streams concurrently.

        ``max_output_bytes`` keeps only the tail of each stream (``None``: all
        of it). Modal's exec deadline does not raise: ``wait()`` returns ``-1``
        once it passes, and the stream iterators may raise ``ExecTimeoutError``;
        both end up as ``returncode=-1`` / ``stream_error`` for the caller to
        classify. Transport exceptions from ``exec`` itself propagate.
        """
        assert self.sandbox is not None, "Sandbox not started"
        env_vars = self._build_env_vars()
        proc = self.sandbox.exec(*argv, timeout=timeout, env=env_vars or None, text=False)

        if stdin is not None:
            writer = proc.stdin
            for offset in range(0, len(stdin), self._STDIN_CHUNK_BYTES):
                writer.write(stdin[offset : offset + self._STDIN_CHUNK_BYTES])
                writer.drain()
            writer.write_eof()
            writer.drain()

        stderr_result: list[tuple[bytes, Exception | None]] = []

        def read_stderr():
            stderr_result.append(self._drain_stream(proc.stderr, max_output_bytes))

        t = threading.Thread(target=read_stderr, name=f"modal-stderr-{self.sandbox_id}", daemon=True)
        t.start()
        stdout, out_err = self._drain_stream(proc.stdout, max_output_bytes)
        t.join()
        stderr, err_err = stderr_result[0] if stderr_result else (b"", None)

        try:
            rc = proc.wait()
        except Exception as e:
            return _ExecOutcome(None, stdout, stderr, out_err or err_err or e)
        return _ExecOutcome(rc, stdout, stderr, out_err or err_err)

    @staticmethod
    def _is_exec_timeout(exc: Exception | None) -> bool:
        if exc is None:
            return False
        cls = getattr(_lazy_import_modal().exception, "ExecTimeoutError", None)
        return cls is not None and isinstance(exc, cls)

    def execute(self, command: str, cwd: str = "", timeout: int = None) -> dict[str, Any]:
        """Execute a command in the sandbox and return the result as a dict.

        Returns ``{"output": str, "returncode": int | None, "reason": str}``
        with reason ∈ {"ok", "pod_timeout", "client_timeout",
        "transport_error"}. The reason names follow ``KubernetesEnvironment``
        because the tool layer matches on them: ``pod_timeout`` is the
        in-sandbox ``timeout`` firing (rc 124), ``client_timeout`` is Modal's
        exec deadline expiring first (rc -1, returncode None).
        """
        exec_t0 = time.monotonic()
        if timeout is None:
            timeout = self.config.timeout
        cwd = cwd or self.config.cwd
        assert self.sandbox is not None, "Sandbox not started"

        # Merge stderr into stdout inside the shell so the model sees the two
        # interleaved the way a terminal would, not stdout then stderr; stdin
        # is /dev/null so a command that reads it gets EOF instead of hanging
        # on the exec's open pipe.
        full_command = f"exec 2>&1 </dev/null\ncd {shlex.quote(cwd)} && {command}"
        # Plain `timeout N cmd` — works on both GNU coreutils and BusyBox.
        argv = ["timeout", str(timeout), "/bin/bash", "-lc", full_command]

        try:
            outcome = self._exec(argv, timeout=timeout + self._EXEC_GRACE_S, max_output_bytes=self._MAX_OUTPUT_BYTES)
        except Exception as e:
            self.logger.error(f"Sandbox exec failed: {e}")
            result = self._return_transport_error(f"Sandbox exec failed: {type(e).__name__}: {e}")
        else:
            raw = (outcome.stdout + outcome.stderr).decode("utf-8", "replace")
            rc = outcome.returncode
            if rc == -1 or self._is_exec_timeout(outcome.stream_error):
                self.logger.error(f"Command timed out (client deadline) after {timeout}s")
                result = {
                    "output": raw + f"\nCommand timed out (client) after {timeout}s",
                    "returncode": None,
                    "reason": "client_timeout",
                }
            elif rc is None or outcome.stream_error is not None:
                err = outcome.stream_error
                detail = f"{type(err).__name__}: {err}" if err is not None else "no exit status"
                self.logger.error(f"Sandbox exec stream failed ({detail}); bytes={len(raw)}")
                result = self._return_transport_error(raw or f"Sandbox exec stream failed: {detail}")
            elif rc == 124:
                result = {
                    "output": raw + f"\nCommand timed out (pod) after {timeout}s",
                    "returncode": 124,
                    "reason": "pod_timeout",
                }
            else:
                result = {"output": raw, "returncode": rc, "reason": "ok"}

        exec_duration = time.monotonic() - exec_t0
        cmd_short = command[:200] + ("..." if len(command) > 200 else "")
        log_fn = self.logger.debug if result["reason"] == "ok" else self.logger.info
        log_fn(
            f"[sandbox_exec] sandbox={self.sandbox_id} duration={exec_duration:.2f}s "
            f"rc={result['returncode']} reason={result['reason']} cmd={cmd_short!r}"
        )
        return result

    # ── file transfer ───────────────────────────────────────────────────────

    def copy_to(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None:
        """Copy a local file or directory into the sandbox.

        The source is tarred locally and streamed on the stdin of a ``tar x``
        exec, so files, directories, symlinks and modes all travel the same
        way. The remote read is bounded with ``head -c`` in case the EOF is
        lost, so a stalled transfer fails instead of hanging the exec.
        """
        copy_t0 = time.time()
        assert self.sandbox is not None, "Sandbox not started"
        src_path = os.path.abspath(src_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"source not found: {src_path}")

        dest_dir = os.path.dirname(dest_path) or "."
        arcname = os.path.basename(dest_path) or os.path.basename(src_path)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            tf.add(src_path, arcname=arcname, recursive=True)
        tar_bytes = buf.getvalue()

        untar = [
            "/bin/sh",
            "-c",
            f"mkdir -p {shlex.quote(dest_dir)} && head -c {len(tar_bytes)} | tar xmf - -C {shlex.quote(dest_dir)}",
        ]

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                outcome = self._exec(untar, timeout=timeout, stdin=tar_bytes)
                if outcome.returncode != 0:
                    stderr = outcome.stderr.decode("utf-8", "replace")[-400:]
                    raise RuntimeError(f"untar failed with rc={outcome.returncode}, stderr={stderr}")
                self.logger.info(
                    f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_to "
                    f"src={src_path} dest={dest_path} bytes={len(tar_bytes)} "
                    f"duration={time.time() - copy_t0:.2f}s attempts={attempt}"
                )
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"[copy_to_sandbox] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s"
                    )
                    time.sleep(5)
                    continue
                self.logger.error(
                    f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_to_failed "
                    f"src={src_path} dest={dest_path} duration={time.time() - copy_t0:.2f}s attempts={attempt}"
                )
                raise last_err

    def copy_out(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None:
        """Copy a sandbox file or directory to the local filesystem.

        Streams ``src_path`` out as a tar archive on the exec stdout, then
        extracts it so the entry lands at ``dest_path`` (the remote basename is
        repacked under ``dest_path``'s basename, as ``copy_to`` does).
        """
        assert self.sandbox is not None, "Sandbox not started"

        src_dir = os.path.dirname(src_path) or "."
        src_base = os.path.basename(src_path)
        if not src_base:
            raise ValueError(f"invalid src_path (no basename): {src_path!r}")

        dest_path = os.path.abspath(dest_path)
        dest_parent = os.path.dirname(dest_path) or "."
        dest_name = os.path.basename(dest_path) or src_base

        tar_cmd = [
            "/bin/sh",
            "-c",
            f"test -e {shlex.quote(src_path)} && tar cf - -C {shlex.quote(src_dir)} {shlex.quote(src_base)}",
        ]

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                outcome = self._exec(tar_cmd, timeout=timeout)
                stderr_text = outcome.stderr.decode("utf-8", "replace")
                if outcome.returncode != 0 or outcome.stream_error is not None:
                    raise RuntimeError(
                        f"tar failed with rc={outcome.returncode}, stderr={stderr_text[-400:]}"
                        + (f", stream error: {outcome.stream_error}" if outcome.stream_error else "")
                    )
                if not outcome.stdout:
                    raise RuntimeError(f"empty tar stream from sandbox, stderr={stderr_text[-400:]}")
                _extract_single_entry(outcome.stdout, src_base, dest_parent, dest_name)
                self.logger.info(
                    f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_out "
                    f"src={src_path} dest={dest_path} bytes={len(outcome.stdout)} attempts={attempt}"
                )
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"[copy_out_sandbox] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s"
                    )
                    time.sleep(5)
                    continue
                self.logger.error(f"[copy_out_sandbox] failed after {max_retries} attempts: {e}")
                raise last_err


def _extract_single_entry(tar_bytes: bytes, src_base: str, dest_parent: str, dest_name: str) -> None:
    """Extract an archive with one top-level member into ``dest_parent`` and
    move that member onto ``dest_parent/dest_name``, replacing what was there."""
    os.makedirs(dest_parent, exist_ok=True)
    tmp_dir = os.path.join(dest_parent, f".copy_out_{uuid.uuid4().hex[:12]}")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            tf.extractall(tmp_dir)
        extracted = os.path.join(tmp_dir, src_base)
        if not os.path.exists(extracted):
            entries = os.listdir(tmp_dir)
            if len(entries) != 1:
                raise RuntimeError(f"unexpected archive layout: {entries}")
            extracted = os.path.join(tmp_dir, entries[0])

        final_path = os.path.join(dest_parent, dest_name)
        if os.path.exists(final_path):
            if os.path.isdir(final_path) and not os.path.islink(final_path):
                shutil.rmtree(final_path)
            else:
                os.remove(final_path)
        shutil.move(extracted, final_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
