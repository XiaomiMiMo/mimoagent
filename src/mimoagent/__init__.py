"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mimoagent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "0.1.0"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir

from mimoagent.utils.log import logger

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MIMOAGENT_GLOBAL_CONFIG_DIR") or user_config_dir("mimoagent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "", timeout: int = None) -> dict[str, str]: ...

    def execute_detached(
        self,
        command: str,
        cwd: str = "",
        timeout: int = None,
        *,
        idle_files: list[str] | tuple[str, ...] = (),
        idle_timeout: int = 0,
    ) -> dict[str, str]:
        """Like ``execute`` for runs that may outlive a single transport
        connection (hours-long blackbox harness sessions). Backends without a
        long-connection problem simply delegate to ``execute``."""
        ...

    def copy_to(self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 10) -> None: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...

    def get_model_query_kwargs(self) -> dict: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
