"""Kubernetes environment for executing commands in Kubernetes pods."""

import hashlib
import io
import os
import random
import re
import shlex
import shutil
import stat
import tarfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import urllib3
from kubernetes import client, config
from kubernetes.stream import stream

# The k8s client talks to the API server with TLS verification disabled on
# some clusters, and every request then emits an InsecureRequestWarning.
# Silence it once at import.
urllib3.disable_warnings()

from mimoagent.environments import TransportError
from mimoagent.environments.detached import DetachedExecMixin
from mimoagent.utils.log import get_logger as _get_logger


def _default_namespace() -> str:
    return os.getenv("K8S_NAMESPACE", "default")


@dataclass
class KubernetesEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the pod."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the pod.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the pod."""
    kubeconfig: str | None = None
    """Path to a kubeconfig file. ``None`` uses the kubernetes client's default
    resolution (``$KUBECONFIG``, then ``~/.kube/config``), falling back to the
    in-cluster service account when the runner itself runs inside a pod."""
    namespace: str = field(default_factory=_default_namespace)
    """Kubernetes namespace to use (default: ``$K8S_NAMESPACE`` or ``default``)."""
    cpu_request: str = "0.5"
    """CPU request for the pod."""
    memory_request: str = "1Gi"
    """Memory request for the pod."""
    cpu_limit: str = "4"
    """CPU limit for the pod."""
    memory_limit: str = "8Gi"
    """Memory limit for the pod."""
    pod_timeout: str = "2h"
    """Max duration to keep pod running. Uses the same format as the sleep command."""
    labels: dict[str, str] = field(default_factory=lambda: {"app": "mimoagent"})
    """Labels to apply to the pod."""
    annotations: dict[str, str] = field(default_factory=dict)
    """Annotations to apply to the pod."""
    node_selector: dict[str, str] | None = None
    """``nodeSelector`` for the pod. Omitted from the pod spec when ``None``."""
    tolerations: list[dict[str, Any]] | None = None
    """Pod tolerations (raw Kubernetes dicts). Omitted from the spec when ``None``."""
    host_network: bool = False
    """Whether to run the pod with ``hostNetwork: true``."""
    image_pull_secrets: list[str] | None = None
    """Names of ``imagePullSecrets`` to attach, for private registries."""
    raise_on_transport_error: bool = False
    """Raise TransportError on transport failure instead of returning a dict.
    When True, any transport failure (exec stream open failure, stream closed
    without returncode) raises TransportError. The agent framework escalates
    this to InfraError, terminating the agent loop immediately."""
    answer_leak_blocklist: list[str] | None = None
    """Domains to blackhole (``0.0.0.0`` in ``/etc/hosts``) at the end of
    environment setup, so the agent cannot fetch reference solutions from the
    task's upstream PR / patch endpoints during the rollout. DNS-level only:
    it blocks access by hostname, not by raw IP, which is enough for git / curl
    / pip. Applied by ``DatasetEnvironment.setup_environment``."""
    exec_user: str | None = None
    """Run commands as this OS user via ``su``.  ``None`` = container default."""
    max_exec_budget: float = 0
    """Cumulative exec-seconds budget.  0 = unlimited."""
    no_binary: bool = False
    """Delete target binaries after pre-check (ARVO no-binary mode)."""
    max_submits: int = 0
    """Limit grading submissions per rollout.  0 = unlimited."""
    capabilities: list[str] = field(default_factory=list)
    """Linux capabilities to add to the pod security context."""
    skip_precheck: bool = False
    """Skip the dataset environment's pre-check step."""


class KubernetesEnvironment(DetachedExecMixin):
    # Some ingress setups make the k8s exec websocket unreliable for a single
    # stdin frame above roughly 33 MB. Route before that boundary, and before
    # copy_to builds an equally large in-memory tar archive.
    _COPY_TO_CHUNK_THRESHOLD_BYTES: int = 32 << 20

    # The exec websocket crosses a regional load balancer that hard-closes every
    # connection after ~4h. That is a lifetime cap, not an idle timeout, and an
    # exec session cannot be re-attached, so a command that outlives the
    # connection loses its exit status even though it keeps running in the pod.
    # execute_detached (DetachedExecMixin) launches the command in its own
    # session and polls an rc marker with short execs instead.
    _DETACHED_LOG_TAG = "pod_exec_detached"

    @property
    def logger(self):
        """Scoped instance logger, resolved on each access.

        Routes to the per-instance ``instance.log`` when an ``AgentLogContext``
        is bound for the current thread (batch mode), otherwise falls back to
        the module logger. Mirrors ``DatasetEnvironment.logger`` so pod setup /
        exec / cleanup events land in the instance's own log file instead of
        flooding the shared console / run-wide log.
        """
        return _get_logger()

    def __init__(self, *, config_class: type = KubernetesEnvironmentConfig, **kwargs):
        """
        This class executes bash commands in a Kubernetes pod.
        See `KubernetesEnvironmentConfig` for keyword arguments.
        """
        self.config = config_class(**kwargs)

        for k, v in self.config.labels.items():
            sanitized = self._sanitize_label_value(v)
            if sanitized != v:
                self.logger.warning(
                    "Label value for %r is not a valid k8s label (%r), sanitizing to %r",
                    k,
                    v,
                    sanitized,
                )
                self.config.labels[k] = sanitized

        self.pod_name: str | None = None
        self.pod_started: bool = False
        self.instance_id: str | None = None
        self._pod_start_time: float | None = None
        self._cumulative_exec_time: float = 0.0
        self._setup_kubernetes_client()

    @staticmethod
    def _sanitize_label_value(value: str) -> str:
        """Coerce a string into a valid k8s label value.

        A valid label value is at most 63 chars, and must be empty or consist of
        alphanumeric characters, '-', '_' or '.', starting and ending with an
        alphanumeric character. We replace illegal characters, truncate to 63
        chars, then strip any leading/trailing non-alphanumeric characters (which
        truncation itself may expose).
        """
        # Replace any character outside [A-Za-z0-9-_.] with '-'.
        value = re.sub(r"[^A-Za-z0-9\-_.]", "-", value)
        # Truncate to the 63-char limit.
        value = value[:63]
        # Strip leading/trailing chars that aren't alphanumeric.
        value = re.sub(r"^[^A-Za-z0-9]+", "", value)
        return re.sub(r"[^A-Za-z0-9]+$", "", value)

    def start(self):
        """Start the Kubernetes pod. Must be called before execute() or copy_to()."""
        self._start_pod()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _setup_kubernetes_client(self):
        """Setup Kubernetes client with a per-instance Configuration + ApiClient.

        The kubeconfig is loaded into an instance-held ``Configuration``
        (``client_configuration=``) instead of the library's process-global
        default. Multiple environments with different kubeconfigs
        (per-instance ``routing.kubeconfig``) can share one process; going
        through the global default, every ``load_kube_config`` would
        overwrite it and create/cleanup could silently target whichever
        cluster was loaded last.

        With no kubeconfig configured and none at the default location, the
        in-cluster service-account configuration is used so the runner can
        itself run as a pod.
        """
        self._kube_configuration = client.Configuration()
        kubeconfig = self.config.kubeconfig
        default_kubeconfig = os.getenv("KUBECONFIG") or os.path.expanduser("~/.kube/config")
        if kubeconfig is None and not os.path.exists(default_kubeconfig) and os.getenv("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config(client_configuration=self._kube_configuration)
        else:
            config.load_kube_config(
                config_file=kubeconfig,
                client_configuration=self._kube_configuration,
            )
        self._api_client = client.ApiClient(configuration=self._kube_configuration)
        self.v1_api = client.CoreV1Api(api_client=self._api_client)

    def _build_env_vars(self) -> dict[str, str]:
        """Resolve the pod's environment: forwarded host variables first, then
        the explicitly configured ``env`` (which wins on conflict)."""
        env_vars: dict[str, str] = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env_vars[key] = value
        env_vars.update(self.config.env)
        return env_vars

    def _build_pod_body(self, env_vars: dict[str, str]) -> dict[str, Any]:
        """Assemble the pod manifest from the config. Pure, so tests can assert
        on the exact spec without a cluster."""
        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        spec: dict[str, Any] = {
            "hostname": "container" + str(random.randint(1000, 9999)),
            "automountServiceAccountToken": False,
            "hostNetwork": self.config.host_network,
            "restartPolicy": "OnFailure",
            "containers": [
                {
                    "name": "main",
                    "image": self.config.image,
                    "command": ["/bin/bash", "-lc"],
                    "args": ["/bin/bash", "-lc"],
                    "stdin": True,
                    "tty": True,
                    "env": env_spec,
                    "imagePullPolicy": "IfNotPresent",
                    "resources": {
                        "requests": {"cpu": self.config.cpu_request, "memory": self.config.memory_request},
                        "limits": {"cpu": self.config.cpu_limit, "memory": self.config.memory_limit},
                    },
                }
            ],
        }
        if self.config.node_selector is not None:
            spec["nodeSelector"] = dict(self.config.node_selector)
        if self.config.tolerations is not None:
            spec["tolerations"] = list(self.config.tolerations)
        if self.config.image_pull_secrets:
            spec["imagePullSecrets"] = [{"name": name} for name in self.config.image_pull_secrets]
        if self.config.capabilities:
            spec["containers"][0]["securityContext"] = {
                "capabilities": {"add": list(self.config.capabilities)}
            }
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_name,
                "labels": self.config.labels,
                "annotations": self.config.annotations,
            },
            "spec": spec,
        }

    def _start_pod(self):
        """Start a Kubernetes pod and wait for it to be ready."""
        pod_start_time = time.time()
        self.pod_name = f"mimoagent-{uuid.uuid4().hex[:16]}"
        image_tag = self.config.image.split(":")[-1][:60] if ":" in self.config.image else self.config.image[-60:]
        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Starting pod: {self.config.namespace}/{self.pod_name} image={image_tag}{id_tag}")

        pod_body = self._build_pod_body(self._build_env_vars())

        # Create pod with retries and exponential backoff
        max_retries = 10
        base_backoff = 5

        # random sleep to stagger pod creation
        jitter = random.uniform(0, 10)
        time.sleep(jitter)

        create_start = time.time()
        for attempt in range(max_retries):
            try:
                pod = self.v1_api.create_namespaced_pod(
                    namespace=self.config.namespace, body=pod_body, _request_timeout=120
                )
                break
            except client.ApiException as e:
                if e.status in (406, 409, 429, 500, 503) and attempt < max_retries - 1:
                    backoff = base_backoff * (2**attempt) + random.uniform(0, 3)
                    self.logger.warning(
                        f"Create pod {self.pod_name} attempt {attempt + 1}/{max_retries} "
                        f"failed (status={e.status}), retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                self.logger.error(f"Failed to create pod {self.pod_name}: {e}")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = base_backoff * (2**attempt) + random.uniform(0, 3)
                    self.logger.warning(
                        f"Create pod {self.pod_name} attempt {attempt + 1}/{max_retries} "
                        f"failed ({type(e).__name__}), retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                self.logger.error(f"Failed to create pod {self.pod_name} after {max_retries} attempts: {e}")
                raise
        else:
            raise RuntimeError(f"Failed to create pod after {max_retries} attempts")

        self.pod_rv = pod.metadata.resource_version
        create_duration = time.time() - create_start
        self.logger.info(f"Pod {self.pod_name} created in {create_duration:.1f}s (attempts={attempt + 1})")

        # Wait for pod to be ready
        ready_start = time.time()
        self._wait_for_pod_ready()
        ready_duration = time.time() - ready_start
        self._pod_start_time = time.time()

        pod_duration = time.time() - pod_start_time
        self.logger.info(
            f"Pod {self.pod_name} ready | total={pod_duration:.1f}s "
            f"create={create_duration:.1f}s wait_ready={ready_duration:.1f}s"
        )

    def _wait_for_pod_ready(self, timeout: int = 1200):
        """Wait for the pod to be Running AND its container to be ready."""
        start_time = time.time()
        time.sleep(random.uniform(1, 7))
        while time.time() - start_time < timeout:
            try:
                pod = self.v1_api.read_namespaced_pod(
                    name=self.pod_name, namespace=self.config.namespace, _request_timeout=10
                )
                phase = pod.status.phase
                if phase in ["Failed", "Succeeded", "Unknown"]:
                    raise RuntimeError(f"Pod {self.pod_name} entered terminal phase: {phase}")
                if phase == "Running" and self._container_ready(pod):
                    elapsed = time.time() - start_time
                    node = pod.spec.node_name or "unknown"
                    self.logger.info(f"Pod {self.pod_name} running on {node} ({elapsed:.1f}s)")
                    self.pod_started = True
                    return
            except client.ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Pod {self.pod_name} not found while waiting for readiness") from e
                self.logger.debug(f"Read pod status ApiException: {e}, will retry")
            except RuntimeError:
                raise
            except Exception as e:
                self.logger.debug(f"Read pod error: {e}, checking pod status")
            time.sleep(4 + random.uniform(-1, 1))

        # Timeout reached
        pod = self.v1_api.read_namespaced_pod(name=self.pod_name, namespace=self.config.namespace, _request_timeout=60)
        error_msg = f"Pod {self.pod_name} is in state {pod.status.phase} after {timeout} seconds"
        reasons = []
        if pod.status.phase == "Pending":
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    if condition.status != "True":
                        reasons.append(f"{condition.type}: {condition.reason} - {condition.message}")
        if pod.status.container_statuses:
            for container_status in pod.status.container_statuses:
                if container_status.state.waiting:
                    reasons.append(
                        f"Container {container_status.name}: {container_status.state.waiting.reason} - {container_status.state.waiting.message}"
                    )
                elif not container_status.ready:
                    reasons.append(f"Container {container_status.name}: not ready")
        if reasons:
            error_msg += f". Reasons: {'; '.join(reasons)}"
        raise RuntimeError(error_msg)

    @staticmethod
    def _container_ready(pod) -> bool:
        """Check if the 'main' container in the pod is ready."""
        if not pod.status.container_statuses:
            return False
        for cs in pod.status.container_statuses:
            if cs.name == "main":
                return cs.ready is True
        return False

    def _return_transport_error(self, output: str) -> dict[str, Any]:
        """Build transport_error result, or raise TransportError if configured to abort."""
        if self.config.raise_on_transport_error:
            raise TransportError(output)
        return {"output": output, "returncode": None, "reason": "transport_error"}

    def _detached_assert_started(self) -> None:
        assert self.pod_name, "Pod not started"
        assert self.pod_started, "Pod not started"

    def _detached_target(self) -> str:
        return f"pod={self.pod_name}"

    # Maximum bytes to keep for command output per exec/copy call (50 MB).
    # Output beyond this is shed from the head so only the tail survives.
    _MAX_OUTPUT_BYTES: int = 50 * 1024 * 1024

    @staticmethod
    def _drain_ws_all_buffer(resp) -> None:
        """Clear the WSClient internal ``_all`` buffer to prevent unbounded
        memory growth.

        The kubernetes ``WSClient`` accumulates every channel's data into an
        internal ``_all`` StringIO regardless of whether it is consumed. We only
        ever read per-channel (``read_stdout`` / ``read_stderr``), so ``_all`` is
        never drained and would otherwise grow to tens of GB over long-running
        agent sessions. Truncate it on every read iteration.
        """
        buf = getattr(resp, "_all", None)
        if buf is not None:
            try:
                buf.truncate(0)
                buf.seek(0)
            except Exception:
                pass

    @staticmethod
    def _trim_chunks(chunks: list, total: int, max_bytes: int) -> tuple[list, int]:
        """Drop leading chunks so the joined size is at most ``max_bytes``.

        Returns the trimmed list and its new total size. ``total`` is the
        caller-tracked running size so we avoid re-summing every iteration.
        """
        while chunks and total > max_bytes:
            total -= len(chunks[0])
            chunks.pop(0)
        return chunks, total

    def execute(self, command: str, cwd: str = "", timeout: int = None, *, as_user: str | None = None) -> dict[str, Any]:
        """Execute a command in the Kubernetes pod and return the result as a dict.

        Returns {"output": str, "returncode": int | None, "reason": str} where
        reason ∈ {"ok", "pod_timeout", "client_timeout", "transport_error"}.
        Callers must NOT coerce returncode to 0; check reason first.
        """
        exec_t0 = time.monotonic()

        if self.config.max_exec_budget > 0 and self._cumulative_exec_time >= self.config.max_exec_budget:
            return {
                "output": f"Exec time budget exhausted ({self._cumulative_exec_time:.0f}s/{self.config.max_exec_budget:.0f}s)",
                "returncode": 1,
                "reason": "budget_exhausted",
            }

        if timeout is None:
            timeout = self.config.timeout

        cwd = cwd or self.config.cwd
        assert self.pod_name, "Pod not started"
        assert self.pod_started, "Pod not started"

        if cwd != self.config.cwd:
            full_command = f"cd {shlex.quote(cwd)} && {command}"
        else:
            # The pod's PWD is the image WORKDIR, which some images leave unset
            # (or set to /). A bare git command would then fail with "not a git
            # repository", so always cd into config.cwd explicitly.
            full_command = f"cd {shlex.quote(self.config.cwd)} && {command}"

        # Plain `timeout N cmd` — works on both GNU coreutils and BusyBox.
        # If the command ignores SIGTERM, the client deadline below is the
        # fallback (classified as client_timeout).
        if as_user:
            full_command = f"su {shlex.quote(as_user)} -s /bin/sh -c {shlex.quote(full_command)}"

        exec_argv = ["timeout", str(timeout), "/bin/bash", "-lc", full_command]

        # (1) Single monotonic deadline — every wait below is computed from it.
        grace = 5
        deadline = time.monotonic() + timeout + grace

        resp = None
        try:
            resp = stream(
                self.v1_api.connect_get_namespaced_pod_exec,
                self.pod_name,
                self.config.namespace,
                command=exec_argv,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
                _request_timeout=min(30, timeout + 3),
            )
        except Exception as e:
            self.logger.error(f"Error opening exec stream: {e}")
            exec_duration = time.monotonic() - exec_t0
            cmd_short = command[:200] + ("..." if len(command) > 200 else "")
            self.logger.info(
                f"[pod_exec] pod={self.pod_name} duration={exec_duration:.2f}s "
                f"rc=None reason=transport_error cmd={cmd_short!r}"
            )
            return self._return_transport_error(f"Error opening exec stream: {e}")

        output_chunks: list[str] = []
        output_size = 0
        done = threading.Event()

        def reader():
            nonlocal output_size
            try:
                while not done.is_set() and resp.is_open():
                    if time.monotonic() >= deadline:
                        break
                    try:
                        resp.update(timeout=1)
                        if resp.peek_stdout():
                            chunk = resp.read_stdout()
                            output_chunks.append(chunk)
                            output_size += len(chunk)
                        if resp.peek_stderr():
                            chunk = resp.read_stderr()
                            output_chunks.append(chunk)
                            output_size += len(chunk)
                        # Shed oldest output when it exceeds the cap so a command
                        # spewing huge output can't exhaust memory.
                        if output_size > self._MAX_OUTPUT_BYTES:
                            _, output_size = self._trim_chunks(output_chunks, output_size, self._MAX_OUTPUT_BYTES)
                        # Keep the WSClient._all accumulator from growing unbounded.
                        self._drain_ws_all_buffer(resp)
                    except Exception:
                        break
            finally:
                done.set()

        t = threading.Thread(target=reader, name=f"k8s-exec-{self.pod_name}", daemon=True)
        t.start()
        remaining = deadline - time.monotonic()
        done.wait(timeout=max(0.0, remaining))
        client_timed_out = not done.is_set()

        try:
            resp.close()
        except Exception as e:
            self.logger.debug(f"resp.close() after exec: {e}")

        t.join(timeout=2)

        raw = "".join(output_chunks)
        try:
            rc_stream = resp.returncode
        except Exception:
            rc_stream = None

        # (4) Classify from rc_stream. `timeout` returns 124 on expiry (covers
        # both SIGTERM and the --kill-after SIGKILL path). rc=None means the
        # exec channel never delivered an exit status -> transport failure.
        if client_timed_out:
            self.logger.error(f"Command timed out (client deadline) after {timeout}s")
            result = {
                "output": raw + f"\nCommand timed out (client) after {timeout}s",
                "returncode": None,
                "reason": "client_timeout",
            }
        elif rc_stream is None:
            self.logger.error(f"Exec stream closed without returncode; bytes={len(raw)}")
            result = self._return_transport_error(raw or "Exec stream closed without returncode")
        elif rc_stream == 124:
            result = {
                "output": raw + f"\nCommand timed out (pod) after {timeout}s",
                "returncode": 124,
                "reason": "pod_timeout",
            }
        else:
            result = {"output": raw, "returncode": rc_stream, "reason": "ok"}

        exec_duration = time.monotonic() - exec_t0
        self._cumulative_exec_time += exec_duration
        cmd_short = command[:200] + ("..." if len(command) > 200 else "")
        log_fn = self.logger.debug if result["reason"] == "ok" else self.logger.info
        log_fn(
            f"[pod_exec] pod={self.pod_name} duration={exec_duration:.2f}s "
            f"rc={result['returncode']} reason={result['reason']} cmd={cmd_short!r}"
        )
        return result

    def copy_to(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10, as_user: str | None = None) -> None:
        copy_t0 = time.time()
        assert self.pod_name and self.pod_started, "Pod not started"
        src_path = os.path.abspath(src_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"source not found: {src_path}")

        # A large regular file cannot safely fit in one exec websocket frame.
        # Keep directories and symlinks on the tar path so copy_to preserves
        # their existing layout and metadata semantics.
        chunk_candidate = os.path.isfile(src_path) and not os.path.islink(src_path) and os.path.basename(dest_path)
        size = os.path.getsize(src_path) if chunk_candidate else None
        if size is not None and size >= self._COPY_TO_CHUNK_THRESHOLD_BYTES:
            try:
                info = self.copy_file_chunked(
                    src_path,
                    dest_path,
                    timeout=timeout,
                    max_retries=max_retries,
                    as_user=as_user,
                )
            except Exception:
                copy_duration = time.time() - copy_t0
                self.logger.error(
                    f"[pod_copy] pod={self.pod_name} event=copy_to_failed "
                    f"mode=chunked src={src_path} dest={dest_path} bytes={size} "
                    f"duration={copy_duration:.2f}s"
                )
                raise
            copy_duration = time.time() - copy_t0
            self.logger.info(
                f"[pod_copy] pod={self.pod_name} event=copy_to "
                f"mode=chunked src={src_path} dest={dest_path} bytes={size} "
                f"chunks={info['chunks']} duration={copy_duration:.2f}s"
            )
            return

        dest_dir = os.path.dirname(dest_path) or "."
        arcname = os.path.basename(dest_path) or os.path.basename(src_path)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            tf.add(src_path, arcname=arcname, recursive=True)
        tar_bytes = buf.getvalue()

        # Bound the pod-side read to exactly the archive size with ``head -c``.
        # ``write_stdin("")`` does NOT close the exec stdin channel (it only
        # writes an empty websocket frame), so the pod never sees EOF on stdin.
        # GNU tar stops at the archive's end-of-archive zero blocks and exits
        # rc=0 regardless, but BusyBox tar (Alpine images, e.g. several
        # gravitational/teleport + protonmail/webclients swe-bench-pro
        # instances) keeps reading stdin waiting for more input, never exits,
        # and the exec returns rc=None — copy_to then burns the full timeout,
        # retries, and the agent eventually trips agent_timeout. ``head -c N``
        # closes the pipe after exactly N bytes, handing tar a real EOF on both
        # GNU and BusyBox. Belt-and-suspenders: keep ``tar xmf -`` reading the
        # same stream.
        untar_script = f"mkdir -p {shlex.quote(dest_dir)} && head -c {len(tar_bytes)} | tar xmf - -C {shlex.quote(dest_dir)}"
        if as_user:
            untar_script = f"su {shlex.quote(as_user)} -s /bin/sh -c {shlex.quote(untar_script)}"
        untar_cmd = ["/bin/sh", "-c", untar_script]

        last_err = None
        for attempt in range(1, max_retries + 1):
            resp = None
            try:
                resp = stream(
                    self.v1_api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.config.namespace,
                    command=untar_cmd,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                    _request_timeout=timeout + 5,
                )

                resp.write_stdin(tar_bytes)
                try:
                    resp.write_stdin("")
                except Exception:
                    pass

                out_chunks, err_chunks = [], []
                end_deadline = time.time() + timeout
                while resp.is_open() and time.time() < end_deadline:
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        out_chunks.append(resp.read_stdout())
                    if resp.peek_stderr():
                        err_chunks.append(resp.read_stderr())
                    self._drain_ws_all_buffer(resp)
                    if not resp.is_open():
                        break

                # rc=None means the exec channel never delivered an exit
                # status (transport interrupted mid-transfer) — that is a
                # failure, not a success; let the retry loop handle it.
                try:
                    rc = resp.returncode
                except Exception:
                    rc = None
                if rc != 0:
                    raise RuntimeError(f"untar failed with rc={rc}, stderr={''.join(err_chunks)[-400:]}")
                copy_duration = time.time() - copy_t0
                self.logger.info(
                    f"[pod_copy] pod={self.pod_name} event=copy_to "
                    f"src={src_path} dest={dest_path} "
                    f"duration={copy_duration:.2f}s attempts={attempt}"
                )
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(f"[copy_to_pod] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s")
                    time.sleep(5)
                    continue
                else:
                    copy_duration = time.time() - copy_t0
                    self.logger.error(
                        f"[pod_copy] pod={self.pod_name} event=copy_to_failed "
                        f"src={src_path} dest={dest_path} "
                        f"duration={copy_duration:.2f}s attempts={attempt}"
                    )
                    raise last_err
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception as close_err:
                        self.logger.debug(f"[copy_to_pod] resp.close(): {close_err}")

    def copy_file_chunked(
        self, src_path: str, dest_path: str, *, chunk_bytes: int = 16 << 20, timeout: int = 300, max_retries: int = 5, as_user: str | None = None
    ) -> dict:
        """Send one file to the pod in bounded chunks; verify by sha256.

        `copy_to` puts the whole payload in a single exec stream, and that stream
        dies above roughly 33 MB. Measured on large verifier assets
        are pushed at reward time: every task that graded successfully was
        <= 33.1 MB and every socket failure was >= 35.1 MB, and in the eval's own
        records all 101 `upload_failed` trials sit above 33 MB with none below --
        each of which scored 0.0, the same number a wrong submission gets.

        Only the framing changes; the bytes that land in the pod are identical,
        which the returned sha256 exists to prove. ``copy_to`` selects this path
        automatically for large regular files; callers may also use it directly.
        """
        size = os.path.getsize(src_path)
        quoted = shlex.quote(dest_path)
        dest_dir = os.path.dirname(dest_path) or "."
        r = self.execute(f"mkdir -p {shlex.quote(dest_dir)} && : > {quoted}")
        if r.get("returncode") != 0:
            raise TransportError(f"could not truncate {dest_path}: {r.get('output', '')[-200:]}")

        local = hashlib.sha256()
        t0 = time.time()
        idx = 0
        with open(src_path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_bytes)
                if not chunk:
                    break
                local.update(chunk)
                self._write_chunk(quoted, chunk, idx, chunk_bytes, timeout=timeout, max_retries=max_retries, as_user=as_user)
                idx += 1

        r = self.execute(f"sha256sum {quoted} | cut -d' ' -f1", timeout=600)
        out = (r.get("output") or "").strip()
        remote = out.splitlines()[-1] if out else ""
        if remote != local.hexdigest():
            # A short file left behind is worse than none: a consumer sees the path,
            # untars it silently wrong, and blames the submission. Removing it makes
            # the absence visible to whoever reads the assets.
            self.execute(f"rm -f {quoted}")
            raise TransportError(
                f"sha256 mismatch after upload of {dest_path}: "
                f"local {local.hexdigest()[:16]} vs pod {remote[:16]} (removed)"
            )

        mode = stat.S_IMODE(os.stat(src_path).st_mode)
        r = self.execute(f"chmod {mode:o} {quoted}")
        if r.get("returncode") != 0:
            self.execute(f"rm -f {quoted}")
            raise TransportError(
                f"could not preserve mode {mode:o} on {dest_path}: {r.get('output', '')[-200:]} (removed)"
            )
        return {"bytes": size, "chunks": idx, "seconds": round(time.time() - t0, 1), "sha256": remote}

    def _write_chunk(
        self, quoted_dest: str, chunk: bytes, idx: int, chunk_bytes: int, *, timeout: int, max_retries: int, as_user: str | None = None
    ) -> None:
        # Two properties are needed and neither is obvious.
        #
        # `head -c N` rather than `cat`: cat waits for EOF, this client cannot
        # half-close stdin, so every chunk burnt the full timeout (3.75 min per
        # 16 MB). But head also exits 0 on a SHORT read, so the size check below is
        # what turns a truncated websocket write into a non-zero exit.
        #
        # `dd seek=idx` rather than `>>`: the exec is at-least-once. When a chunk
        # landed but its return code was lost, a retry appended the same bytes
        # again -- caught only by the final sha256, after which the file had to be
        # thrown away. Writing at a fixed offset makes a retry overwrite instead.
        n = len(chunk)
        tmp = f"/tmp/_chunk_{os.getpid()}_{threading.get_ident()}"
        chunk_script = (
            f'head -c {n} > {tmp} && [ "$(stat -c %s {tmp})" = "{n}" ] '
            f"&& dd if={tmp} of={quoted_dest} bs={chunk_bytes} seek={idx} "
            f"conv=notrunc status=none && rm -f {tmp}"
        )
        if as_user:
            chunk_script = f"su {shlex.quote(as_user)} -s /bin/sh -c {shlex.quote(chunk_script)}"
        cmd = ["/bin/sh", "-c", chunk_script]
        for attempt in range(1, max_retries + 1):
            resp = None
            try:
                resp = stream(
                    self.v1_api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.config.namespace,
                    command=cmd,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                    _request_timeout=timeout + 5,
                )
                resp.write_stdin(chunk)
                try:
                    resp.write_stdin("")
                except Exception:  # noqa: BLE001
                    pass
                err_chunks: list[str] = []
                deadline = time.time() + timeout
                while resp.is_open() and time.time() < deadline:
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        resp.read_stdout()
                    if resp.peek_stderr():
                        err_chunks.append(resp.read_stderr())
                    self._drain_ws_all_buffer(resp)
                try:
                    rc = resp.returncode
                except Exception:  # noqa: BLE001
                    rc = None
                if rc != 0:
                    raise RuntimeError(f"chunk append rc={rc} stderr={''.join(err_chunks)[-300:]}")
                return
            except Exception as e:  # noqa: BLE001
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                raise TransportError(f"chunk append failed after {max_retries}: {e}")
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001
                        pass

    def copy_out(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10, as_user: str | None = None) -> None:
        """Copy a file or directory from the pod to the local filesystem.

        Streams `src_path` (a path inside the pod) out as a tar archive on the
        exec stdout channel — opened with `binary=True` so the tar bytes are
        not utf-8 mangled — then extracts it locally so that the entry lands at
        `dest_path` (mirrors `copy_to`: the remote basename is repacked under
        `dest_path`'s basename). Works for both files and directories.
        """
        assert self.pod_name and self.pod_started, "Pod not started"

        # Pack the remote entry by basename so the archive has a single
        # top-level member we can rename to dest_path on extraction.
        src_dir = os.path.dirname(src_path) or "."
        src_base = os.path.basename(src_path)
        if not src_base:
            raise ValueError(f"invalid src_path (no basename): {src_path!r}")

        dest_path = os.path.abspath(dest_path)
        dest_parent = os.path.dirname(dest_path) or "."
        dest_name = os.path.basename(dest_path) or src_base

        # `-` writes the archive to stdout. Verify existence first so a missing
        # source surfaces as a clear error instead of an empty/garbage stream.
        tar_script = f"test -e {shlex.quote(src_path)} && tar cf - -C {shlex.quote(src_dir)} {shlex.quote(src_base)}"
        if as_user:
            tar_script = f"su {shlex.quote(as_user)} -s /bin/sh -c {shlex.quote(tar_script)}"
        tar_cmd = ["/bin/sh", "-c", tar_script]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = stream(
                    self.v1_api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.config.namespace,
                    command=tar_cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    binary=True,
                    _preload_content=False,
                    _request_timeout=timeout + 5,
                )

                out_chunks: list[bytes] = []
                err_chunks: list[bytes] = []
                end_deadline = time.time() + timeout
                while resp.is_open() and time.time() < end_deadline:
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        out_chunks.append(resp.read_stdout())
                    if resp.peek_stderr():
                        err_chunks.append(resp.read_stderr())
                    # `out_chunks` holds the tar payload we must keep, but the
                    # WSClient._all mirror of it is dead weight — drop it.
                    self._drain_ws_all_buffer(resp)
                    if not resp.is_open():
                        break
                resp.close()

                # As in copy_to: rc=None (no exit status delivered) is a
                # transport failure and must go through the retry loop.
                try:
                    rc = resp.returncode
                except Exception:
                    rc = None
                stderr_text = b"".join(err_chunks).decode("utf-8", "replace")
                if rc != 0:
                    raise RuntimeError(f"tar failed with rc={rc}, stderr={stderr_text[-400:]}")

                tar_bytes = b"".join(out_chunks)
                if not tar_bytes:
                    raise RuntimeError(f"empty tar stream from pod, stderr={stderr_text[-400:]}")

                # Extract into a temp dir alongside dest, then atomically move
                # the single top-level member onto dest_path.
                os.makedirs(dest_parent, exist_ok=True)
                tmp_dir = os.path.join(dest_parent, f".copy_out_{uuid.uuid4().hex[:12]}")
                os.makedirs(tmp_dir, exist_ok=True)
                try:
                    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
                        tf.extractall(tmp_dir)
                    extracted = os.path.join(tmp_dir, src_base)
                    if not os.path.exists(extracted):
                        # Fall back to whatever single entry the archive held.
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
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self.logger.warning(f"[copy_out_pod] attempt {attempt}/{max_retries} failed: {e}; retrying in 5s")
                    time.sleep(5)
                    continue
                else:
                    self.logger.error(f"[copy_out_pod] failed after {max_retries} attempts: {e}")
                    raise last_err

    def cleanup(self):
        """Delete the Kubernetes pod and wait for it to disappear.

        Closes the per-instance stream ApiClient, then uses a dedicated
        ApiClient for the delete+poll cycle (the stream client's pool may
        hold half-closed connections that interfere with the delete request).
        Retries delete+poll until read_namespaced_pod returns 404.
        """
        pod_name = getattr(self, "pod_name", None)
        if pod_name is None:
            return

        id_tag = f" instance={self.instance_id}" if self.instance_id else ""
        self.logger.info(f"Cleaning up pod: {self.config.namespace}/{pod_name}{id_tag}")
        cleanup_start = time.time()

        if self._api_client is not None:
            try:
                self._api_client.close()
            except Exception:
                pass
            self._api_client = None

        namespace = self.config.namespace
        max_attempts = 5
        per_attempt_poll_s = 15

        # Same instance-held Configuration as the stream client — NOT the
        # library's global default, which may point at another environment's
        # cluster by now. A wrong-cluster delete would 404 and be mistaken
        # for "already gone", silently leaking the real pod.
        cleanup_api_client = client.ApiClient(configuration=self._kube_configuration)
        try:
            v1 = client.CoreV1Api(api_client=cleanup_api_client)

            for attempt in range(1, max_attempts + 1):
                propagation = "Foreground" if attempt == 1 else "Background"
                try:
                    v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions(
                            grace_period_seconds=0,
                            propagation_policy=propagation,
                        ),
                        _request_timeout=30,
                    )
                    self.logger.debug(
                        f"delete pod {pod_name} attempt={attempt}/{max_attempts} propagation={propagation} dispatched"
                    )
                except client.ApiException as e:
                    if e.status == 404:
                        cleanup_duration = time.time() - cleanup_start
                        lifetime_str = (
                            f"{time.time() - self._pod_start_time:.1f}s" if self._pod_start_time else "unknown"
                        )
                        self.logger.info(
                            f"Pod {pod_name} already gone (cleanup={cleanup_duration:.1f}s, lifetime={lifetime_str})"
                        )
                        self.pod_started = False
                        self.pod_name = None
                        return
                    # status=0 or any other non-404 — fall through to poll
                    # and retry rather than giving up.
                    self.logger.warning(
                        f"delete pod {pod_name} attempt={attempt} "
                        f"propagation={propagation} ApiException status={e.status}: {e}"
                    )
                except Exception as e:
                    self.logger.warning(f"delete pod {pod_name} attempt={attempt} propagation={propagation} error: {e}")

                # Poll until read returns 404 (authoritative proof of deletion).
                # deletion_timestamp alone isn't enough: pods can sit with it
                # set while finalizers stall removal.
                elapsed = 0
                while elapsed < per_attempt_poll_s:
                    try:
                        v1.read_namespaced_pod(name=pod_name, namespace=namespace, _request_timeout=10)
                    except client.ApiException as e:
                        if e.status == 404:
                            cleanup_duration = time.time() - cleanup_start
                            lifetime_str = (
                                f"{time.time() - self._pod_start_time:.1f}s" if self._pod_start_time else "unknown"
                            )
                            self.logger.info(
                                f"Pod {pod_name} deleted in {cleanup_duration:.1f}s "
                                f"(attempt={attempt}, poll={elapsed}s, lifetime={lifetime_str})"
                            )
                            self.pod_started = False
                            self.pod_name = None
                            return
                        self.logger.debug(f"read_namespaced_pod poll for {pod_name}: status={e.status}")
                    except Exception as e:
                        self.logger.debug(f"read_namespaced_pod poll for {pod_name}: {e}")
                    time.sleep(1)
                    elapsed += 1

                self.logger.warning(
                    f"delete pod {pod_name} attempt={attempt} poll timed out "
                    f"after {per_attempt_poll_s}s; retrying delete"
                )

            self.logger.error(
                f"Pod {pod_name} still present after {max_attempts} delete+poll cycles "
                f"(~{max_attempts * per_attempt_poll_s}s) — giving up in cleanup()"
            )
        finally:
            try:
                cleanup_api_client.close()
            except Exception as e:
                self.logger.debug(f"cleanup_api_client.close() for {pod_name}: {e}")
