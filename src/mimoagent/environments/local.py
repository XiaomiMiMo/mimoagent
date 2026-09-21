import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def start(self):
        """No-op for local environment."""
        pass

    def execute(self, command: str, cwd: str = "", timeout: int = None):
        """Execute a command in the local environment and return the result as a dict."""
        if timeout is None:
            timeout = self.config.timeout
        cwd = cwd or self.config.cwd or os.getcwd()
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=cwd,
            env=os.environ | self.config.env,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def execute_detached(self, command: str, cwd: str = "", timeout: int = None, **_ignored) -> dict[str, Any]:
        """No transport to detach from locally: a plain execute."""
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None:
        """Copy a file or directory into the "environment" (plain local copy)."""
        src_path = os.path.abspath(src_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"source not found: {src_path}")
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dest_path)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict() | os.environ
