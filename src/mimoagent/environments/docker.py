import logging
import os
import posixpath
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MIMOAGENT_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=list)
    """Additional arguments to pass to the docker/container executable."""


class DockerEnvironment:
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logging.getLogger("mimoagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)

    def start(self):
        """Start the Docker container. Must be called before execute() or copy_to()."""
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"mimoagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            "infinity",
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # docker pull might take a while
            check=True,
        )
        self.logger.debug(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "", timeout: int = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-c", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def execute_detached(self, command: str, cwd: str = "", timeout: int = None, **_ignored) -> dict[str, Any]:
        """``docker exec`` is a local socket with no connection lifetime cap: a plain execute."""
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(
        self,
        src_path: str,
        dest_path: str,
        *,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        """Copy files from host into the container."""
        assert self.container_id, "Container not started"

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path not found: {src_path}")

        abs_src = os.path.abspath(src_path)
        target = f"{self.container_id}:{dest_path}"

        target_dir = None
        if dest_path.endswith("/"):
            target_dir = dest_path
        else:
            target_dir = posixpath.dirname(dest_path)
            if target_dir in ("", "."):
                target_dir = None

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                if target_dir:
                    mkdir_cmd = f"mkdir -p {shlex.quote(target_dir)}"
                    mkdir_res = self.execute(mkdir_cmd)
                    if mkdir_res.get("returncode", 1) != 0:
                        raise RuntimeError(mkdir_res.get("output", ""))

                cp_cmd = [self.config.executable, "cp", abs_src, target]
                result = subprocess.run(
                    cp_cmd,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stdout.strip())
                return
            except Exception as exc:  # noqa: PERF203 (attempt-based retry)
                last_err = exc
                if attempt == max_retries:
                    break
        if last_err:
            raise last_err

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"{self.config.executable} rm -f {self.container_id} >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
