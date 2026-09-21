"""CubeSandbox environment for executing commands in Cube sandboxes.

Uses the cubesandbox SDK (pip install cubesandbox>=0.2.0) which provides
native support for snapshot, rollback, and clone operations.
"""

import io
import json
import os
import random
import shlex
import shutil
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.utils.log import get_logger as _get_logger


def _lazy_import_cubesandbox():
    try:
        from cubesandbox import Sandbox

        return Sandbox
    except ImportError:
        raise ImportError(
            "CubeEnvironment requires the cubesandbox SDK. Install it with: pip install 'cubesandbox>=0.2.0'"
        )


def _lazy_import_config():
    from cubesandbox import Config

    return Config


@dataclass
class CubeEnvironmentConfig:
    template_id: str
    """Cube sandbox template id (e.g. ``tpl-...``), or a snapshot id."""

    cwd: str = "/testbed"
    """Working directory inside the sandbox."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables exported before every command."""

    forward_env: list[str] = field(default_factory=list)
    """Host environment variables to forward (only when set on the host)."""

    timeout: int = 30
    """Default command timeout in seconds."""

    api_url: str = field(default_factory=lambda: os.getenv("CUBE_API_URL", ""))
    """CubeSandbox API endpoint (``$CUBE_API_URL``). Required."""

    sandbox_domain: str = field(default_factory=lambda: os.getenv("CUBE_SANDBOX_DOMAIN", ""))
    """Domain the sandbox proxy serves sandboxes under (``$CUBE_SANDBOX_DOMAIN``). Required."""

    proxy_node_ip: str | None = field(default_factory=lambda: os.getenv("CUBE_PROXY_NODE_IP"))
    """Optional proxy node IP; when set, DNS resolution of ``sandbox_domain`` is bypassed."""

    proxy_port: int = field(default_factory=lambda: int(os.getenv("CUBE_PROXY_PORT_HTTP", "80")))
    """HTTP port of the sandbox proxy."""

    host_mounts: list[dict[str, str]] = field(default_factory=list)
    """Host directories to mount into the sandbox: ``{hostPath, mountPath, readOnly}`` entries."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Extra metadata passed to ``Sandbox.create()``."""

    labels: dict[str, str] = field(default_factory=lambda: {"app": "mimoagent"})
    """Labels stored in the sandbox metadata (for bulk management / cleanup)."""

    ssl_cert_file: str = field(default_factory=lambda: os.getenv("SSL_CERT_FILE", ""))
    """CA bundle for the API endpoint, when it uses a private certificate authority."""

    raise_on_transport_error: bool = False
    """Raise TransportError on connection failures (the framework escalates it to InfraError)."""


class CubeEnvironment:
    """CubeSandbox-based execution environment with snapshot/rollback/clone support."""

    @property
    def logger(self):
        return _get_logger()

    def __init__(self, *, config_class: type = CubeEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.sandbox = None
        self.sandbox_id: str | None = None
        self.instance_id: str | None = None
        self._sandbox_start_time: float | None = None
        self._snapshots: list[str] = []

    def _build_config(self):
        """Build the ``cubesandbox.Config`` object."""
        if not self.config.api_url or not self.config.sandbox_domain:
            raise ValueError(
                "CubeEnvironment needs api_url and sandbox_domain: set them in the yaml environment block "
                "or via CUBE_API_URL / CUBE_SANDBOX_DOMAIN"
            )
        Config = _lazy_import_config()
        if self.config.ssl_cert_file:
            os.environ.setdefault("SSL_CERT_FILE", self.config.ssl_cert_file)
        return Config(
            api_url=self.config.api_url,
            sandbox_domain=self.config.sandbox_domain,
            proxy_node_ip=self.config.proxy_node_ip,
            proxy_port=self.config.proxy_port,
        )

    def start(self):
        """Create the sandbox and wait for it to be ready."""
        Sandbox = _lazy_import_cubesandbox()
        sandbox_start_time = time.time()
        cfg = self._build_config()

        metadata = dict(self.config.metadata)
        if self.config.host_mounts:
            metadata["host-mount"] = json.dumps(self.config.host_mounts)
        if self.config.labels:
            metadata["labels"] = json.dumps(self.config.labels)
        if self.instance_id:
            metadata["instance_id"] = str(self.instance_id)

        max_retries = 10
        base_backoff = 5

        jitter = random.uniform(0, 10)
        time.sleep(jitter)

        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Creating sandbox: template={self.config.template_id}{id_tag}")

        for attempt in range(max_retries):
            try:
                self.sandbox = Sandbox.create(
                    template=self.config.template_id,
                    metadata=metadata or None,
                    config=cfg,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = base_backoff * (2**attempt) + random.uniform(0, 3)
                    self.logger.warning(
                        f"Create sandbox attempt {attempt + 1}/{max_retries} "
                        f"failed ({type(e).__name__}: {e}), retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                else:
                    self.logger.error(f"Failed to create sandbox after {max_retries} attempts: {e}")
                    raise

        self.sandbox_id = self.sandbox.sandbox_id
        self._sandbox_start_time = time.time()

        duration = time.time() - sandbox_start_time
        self.logger.info(f"Sandbox {self.sandbox_id} ready in {duration:.1f}s (template={self.config.template_id})")

    def execute(self, command: str, cwd: str = "", timeout: int = None) -> dict[str, Any]:
        """Run a command in the sandbox.

        Same result shape as KubernetesEnvironment:
        ``{"output": str, "returncode": int | None, "reason": str}`` with
        ``reason`` in {"ok", "timeout", "transport_error"}.
        """
        exec_t0 = time.monotonic()

        if timeout is None:
            timeout = self.config.timeout

        cwd = cwd or self.config.cwd
        assert self.sandbox is not None, "Sandbox not started"

        env_prefix = self._build_env_prefix()
        full_command = f"{env_prefix}cd {shlex.quote(cwd)} && {command}"

        try:
            result = self.sandbox.commands.run(full_command, timeout=timeout)
            output = (result.stdout or "") + (result.stderr or "")
            ret = {
                "output": output,
                "returncode": result.exit_code,
                "reason": "ok",
            }
        except TimeoutError:
            ret = {
                "output": f"Command timed out after {timeout}s",
                "returncode": 124,
                "reason": "timeout",
            }
        except Exception as e:
            self.logger.error(f"Sandbox exec failed: {e}")
            ret = self._return_transport_error(f"Sandbox exec failed: {e}")

        exec_duration = time.monotonic() - exec_t0
        cmd_short = command[:200] + ("..." if len(command) > 200 else "")
        log_fn = self.logger.debug if ret["reason"] == "ok" else self.logger.info
        log_fn(
            f"[sandbox_exec] sandbox={self.sandbox_id} duration={exec_duration:.2f}s "
            f"rc={ret['returncode']} reason={ret['reason']} cmd={cmd_short!r}"
        )
        return ret

    def execute_detached(self, command: str, cwd: str = "", timeout: int = None, **_ignored) -> dict[str, Any]:
        """The cubesandbox SDK owns the command transport (no k8s exec websocket): a plain execute."""
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None:
        """Copy a local file or directory into the sandbox.

        Single file: written directly with ``sandbox.files.write()``.
        Directory: tarred locally, written, then extracted in the sandbox.
        """
        copy_t0 = time.time()
        assert self.sandbox is not None, "Sandbox not started"

        src_path = os.path.abspath(src_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"source not found: {src_path}")

        is_dir = os.path.isdir(src_path)
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                if not is_dir:
                    self._copy_to_file(src_path, dest_path)
                else:
                    self._copy_to_directory(src_path, dest_path, timeout=timeout)

                copy_duration = time.time() - copy_t0
                self.logger.info(
                    f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_to "
                    f"src={src_path} dest={dest_path} "
                    f"duration={copy_duration:.2f}s attempts={attempt}"
                )
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"[copy_to_sandbox] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s"
                    )
                    time.sleep(5)
                else:
                    copy_duration = time.time() - copy_t0
                    self.logger.error(
                        f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_to_failed "
                        f"src={src_path} dest={dest_path} "
                        f"duration={copy_duration:.2f}s attempts={attempt}"
                    )
                    raise last_err

    def _copy_to_file(self, src_path: str, dest_path: str) -> None:
        """Write one file straight into the sandbox."""
        dest_dir = os.path.dirname(dest_path)
        if dest_dir and dest_dir != "/":
            self.sandbox.commands.run(f"mkdir -p {shlex.quote(dest_dir)}", timeout=30)

        with open(src_path, "rb") as f:
            content = f.read()
        self.sandbox.files.write(dest_path, content)

    def _copy_to_directory(self, src_path: str, dest_path: str, *, timeout: int = 300) -> None:
        """Ship a directory as a tar stream and unpack it in the sandbox."""
        dest_dir = os.path.dirname(dest_path) or "/"
        arcname = os.path.basename(dest_path) or os.path.basename(src_path)
        tar_name = f"/tmp/_copy_{uuid.uuid4().hex[:12]}.tar"

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            tf.add(src_path, arcname=arcname, recursive=True)
        tar_bytes = buf.getvalue()

        self.sandbox.files.write(tar_name, tar_bytes)

        result = self.sandbox.commands.run(
            f"mkdir -p {shlex.quote(dest_dir)} && tar xf {tar_name} -C {shlex.quote(dest_dir)} && rm -f {tar_name}",
            timeout=timeout,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"untar failed with rc={result.exit_code}, stderr={(result.stderr or '')[-400:]}")

    def copy_out(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None:
        """Copy a sandbox file or directory to the local filesystem.

        Single file: read directly with ``sandbox.files.read()``.
        Directory: tarred in the sandbox, read back, extracted locally.
        """
        assert self.sandbox is not None, "Sandbox not started"

        dest_path = os.path.abspath(dest_path)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                check = self.sandbox.commands.run(
                    f"test -d {shlex.quote(src_path)} && echo DIR || echo FILE",
                    timeout=30,
                )
                is_dir = "DIR" in (check.stdout or "")

                if not is_dir:
                    self._copy_out_file(src_path, dest_path)
                else:
                    self._copy_out_directory(src_path, dest_path, timeout=timeout)

                self.logger.info(
                    f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_out "
                    f"src={src_path} dest={dest_path} attempts={attempt}"
                )
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"[copy_out_sandbox] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s"
                    )
                    time.sleep(5)
                else:
                    self.logger.error(
                        f"[sandbox_copy] sandbox={self.sandbox_id} event=copy_out_failed "
                        f"src={src_path} dest={dest_path} attempts={attempt}"
                    )
                    raise last_err

    def _copy_out_file(self, src_path: str, dest_path: str) -> None:
        """Read one file from the sandbox to a local path."""
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        content = self.sandbox.files.read(src_path)
        with open(dest_path, "w") as f:
            f.write(content)

    def _copy_out_directory(self, src_path: str, dest_path: str, *, timeout: int = 300) -> None:
        """Tar a sandbox directory, ship it base64-encoded, and extract it locally."""
        src_dir = os.path.dirname(src_path) or "/"
        src_base = os.path.basename(src_path)
        tar_name = f"/tmp/_copyout_{uuid.uuid4().hex[:12]}.tar"

        result = self.sandbox.commands.run(
            f"tar cf {tar_name} -C {shlex.quote(src_dir)} {shlex.quote(src_base)}",
            timeout=timeout,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"tar cf failed with rc={result.exit_code}, stderr={(result.stderr or '')[-400:]}")

        b64_result = self.sandbox.commands.run(
            f"base64 {tar_name} && rm -f {tar_name}",
            timeout=timeout,
        )
        if b64_result.exit_code != 0:
            raise RuntimeError(f"base64 read failed: {b64_result.stderr}")

        import base64

        tar_bytes = base64.b64decode(b64_result.stdout.strip())

        if not tar_bytes:
            raise RuntimeError(f"empty tar stream from sandbox for {src_path}")

        dest_parent = os.path.dirname(dest_path) or "."
        dest_name = os.path.basename(dest_path) or src_base
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

    def get_host(self, port: int) -> str:
        """Return the externally reachable host for a sandbox port."""
        assert self.sandbox is not None, "Sandbox not started"
        return self.sandbox.get_host(port)

    def get_template_vars(self) -> dict[str, Any]:
        """Return the config as template variables."""
        return asdict(self.config)

    # ── Snapshot / Clone / Rollback ────────────────────────────────────────

    def snapshot(self, name: str | None = None) -> str:
        """Create a persistent snapshot and return its id.

        Snapshots outlive the sandbox (they survive ``kill``) and feed
        ``from_snapshot()``, ``rollback()`` and ``clone()``.
        """
        assert self.sandbox is not None, "Sandbox not started"

        self.logger.info(f"Creating snapshot for sandbox {self.sandbox_id}" + (f" name={name!r}" if name else ""))
        t0 = time.time()

        info = self.sandbox.create_snapshot(name=name)
        snapshot_id = info.snapshot_id
        self._snapshots.append(snapshot_id)

        duration = time.time() - t0
        self.logger.info(f"Snapshot created: {snapshot_id} in {duration:.1f}s (source={self.sandbox_id})")
        return snapshot_id

    def clone(self, n: int = 1, *, concurrency: int = 1) -> list["CubeEnvironment"]:
        """Clone the running sandbox into ``n`` independent copies.

        Each copy inherits the source's full runtime state (memory and
        filesystem); later writes are isolated. The source keeps running.
        Internally: snapshot → create N → delete snapshot (managed by the SDK).

        Args:
            n: number of clones.
            concurrency: parallel creations (a thread pool when > 1).

        Returns:
            ``n`` independent CubeEnvironment instances.
        """
        assert self.sandbox is not None, "Sandbox not started"

        self.logger.info(f"Cloning sandbox {self.sandbox_id} into {n} copies (concurrency={concurrency})")
        t0 = time.time()

        cloned_sandboxes = self.sandbox.clone(n=n, concurrency=concurrency)

        cloned_envs = []
        for sb in cloned_sandboxes:
            env = CubeEnvironment(config_class=type(self.config), **asdict(self.config))
            env.sandbox = sb
            env.sandbox_id = sb.sandbox_id
            env._sandbox_start_time = time.time()
            cloned_envs.append(env)

        duration = time.time() - t0
        self.logger.info(f"Cloned {len(cloned_envs)} sandboxes from {self.sandbox_id} in {duration:.1f}s")
        return cloned_envs

    def rollback(self, snapshot_id: str) -> None:
        """Roll the sandbox back to a snapshot in place.

        The sandbox id is unchanged; its state returns to the snapshot's.
        Commands and further snapshots work normally afterwards.
        """
        assert self.sandbox is not None, "Sandbox not started"

        self.logger.info(f"Rolling back {self.sandbox_id} to snapshot {snapshot_id}")
        t0 = time.time()

        self.sandbox.rollback(snapshot_id)

        duration = time.time() - t0
        self.logger.info(f"Rollback complete: {self.sandbox_id} restored to snapshot={snapshot_id} in {duration:.1f}s")

    @classmethod
    def from_snapshot(cls, snapshot_id: str, **config_kwargs) -> "CubeEnvironment":
        """Create a fresh environment from a persistent snapshot.

        The base-state reuse pattern: set up once → snapshot → many from_snapshot.
        """
        Sandbox = _lazy_import_cubesandbox()

        config_kwargs.setdefault("template_id", snapshot_id)
        env = cls(**config_kwargs)
        cfg = env._build_config()

        env.logger.info(f"Creating sandbox from snapshot {snapshot_id}")
        t0 = time.time()

        env.sandbox = Sandbox.create(template=snapshot_id, config=cfg)
        env.sandbox_id = env.sandbox.sandbox_id
        env._sandbox_start_time = time.time()

        duration = time.time() - t0
        env.logger.info(f"Sandbox {env.sandbox_id} created from snapshot {snapshot_id} in {duration:.1f}s")
        return env

    def pause(self) -> None:
        """Pause the sandbox, keeping memory and disk state; ``resume()`` continues it."""
        assert self.sandbox is not None, "Sandbox not started"
        self.logger.info(f"Pausing sandbox {self.sandbox_id}")
        self.sandbox.pause()
        self.logger.info(f"Sandbox {self.sandbox_id} paused")

    def resume(self, timeout: int = 300) -> None:
        """Resume a paused sandbox.

        Args:
            timeout: idle timeout (seconds) of the resumed sandbox.
        """
        Sandbox = _lazy_import_cubesandbox()
        cfg = self._build_config()

        assert self.sandbox_id is not None, "No sandbox to resume"
        self.logger.info(f"Resuming sandbox {self.sandbox_id}")

        import requests

        s = requests.Session()
        resp = s.post(
            f"{cfg.api_url}/sandboxes/{self.sandbox_id}/connect",
            json={"timeout": timeout},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 300:
            from cubesandbox._exceptions import ApiError

            raise ApiError(resp.text or f"HTTP {resp.status_code}", resp.status_code)

        self.sandbox = Sandbox(resp.json(), config=cfg)
        self.logger.info(f"Sandbox {self.sandbox_id} resumed")

    def list_snapshots(self) -> list[dict]:
        """List every snapshot of the current sandbox."""
        Sandbox = _lazy_import_cubesandbox()
        cfg = self._build_config()

        all_snapshots = []
        next_token = None
        while True:
            items, next_token = Sandbox.list_snapshots(
                sandbox_id=self.sandbox_id,
                next_token=next_token,
                config=cfg,
            )
            for item in items:
                all_snapshots.append(
                    {
                        "snapshot_id": item.snapshot_id,
                        "names": item.names,
                    }
                )
            if next_token is None:
                break
        return all_snapshots

    @staticmethod
    def delete_snapshot(snapshot_id: str, **config_kwargs) -> None:
        """Delete a snapshot."""
        Sandbox = _lazy_import_cubesandbox()
        cfg = None
        if config_kwargs:
            Config = _lazy_import_config()
            cfg = Config(**config_kwargs)
        Sandbox.delete_snapshot(snapshot_id, config=cfg)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def cleanup(self, *, delete_snapshots: bool = False):
        """Destroy the sandbox.

        Args:
            delete_snapshots: also delete every snapshot this instance created.
        """
        if self.sandbox is None and not (delete_snapshots and self._snapshots):
            return

        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Cleaning up sandbox: {self.sandbox_id}{id_tag}")
        cleanup_start = time.time()

        if delete_snapshots and self._snapshots:
            Sandbox = _lazy_import_cubesandbox()
            cfg = self._build_config()
            for snap_id in self._snapshots:
                try:
                    Sandbox.delete_snapshot(snap_id, config=cfg)
                    self.logger.info(f"Deleted snapshot {snap_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete snapshot {snap_id}: {e}")
            self._snapshots.clear()

        if self.sandbox is not None:
            try:
                self.sandbox.kill()
            except Exception as e:
                self.logger.warning(f"Sandbox cleanup error for {self.sandbox_id}: {e}")
            finally:
                lifetime = f"{time.time() - self._sandbox_start_time:.1f}s" if self._sandbox_start_time else "unknown"
                cleanup_duration = time.time() - cleanup_start
                self.logger.info(
                    f"Sandbox {self.sandbox_id} cleaned up in {cleanup_duration:.1f}s (lifetime={lifetime})"
                )
                self.sandbox = None
                self.sandbox_id = None

    def _return_transport_error(self, output: str) -> dict[str, Any]:
        if self.config.raise_on_transport_error:
            raise TransportError(output)
        return {"output": output, "returncode": None, "reason": "transport_error"}

    def _build_env_prefix(self) -> str:
        """Build the ``export VAR=VAL ... && `` prefix that injects env into every execute()."""
        env_vars = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env_vars[key] = value
        env_vars.update(self.config.env)
        if not env_vars:
            return ""
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        return f"export {exports} && "
