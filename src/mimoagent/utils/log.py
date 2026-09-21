"""Logging utilities.

Three independent streams, each with a clear scope:

1. **Module logger** (``logger``) — process-wide, prints to stderr via Rich.
   Used before any instance context is active.
2. **Run-wide file** — ``add_file_handler()`` teees the module logger to a
   second file (e.g. ``outputs/<run>/mimoagent.log``).
3. **Per-instance** — inside an :class:`AgentLogContext`:
   * ``instance.log`` — orchestration events: env setup, test results,
     subagent breadcrumbs, tracebacks. Written via ``get_logger()``.
   * ``agent_msgs/<name>.log`` — one file per agent/subagent, holding only
     that agent's conversation messages. Each agent writes to its OWN file
     via :class:`BaseAgent.msg_path`; there is no stack or contextvar for it.

``AgentLogContext.next_subagent_path(type)`` atomically numbers subagent
files (``explore_1.log``, ``plan_1.log``, …) so parallel ``agent`` tool calls
don't collide.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from rich.logging import RichHandler

# --- Module logger -----------------------------------------------------------


def _setup_root_logger() -> None:
    lg = logging.getLogger("mimoagent")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.handlers.clear()
    handler = RichHandler(show_path=False, show_time=False, show_level=False, markup=True)
    handler.setFormatter(logging.Formatter("%(name)s: %(levelname)s: %(message)s"))
    lg.addHandler(handler)


_setup_root_logger()
logger = logging.getLogger("mimoagent")


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    """Tee the module logger to ``path`` (run-wide log file)."""
    h = logging.FileHandler(path)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    if print_path:
        print(f"Logging to '{path}'")


# --- Per-instance file loggers ----------------------------------------------


def make_file_logger(name: str, path: Path | str) -> logging.Logger:
    """Info-style file logger with timestamp prefix.

    Constructs the Logger directly instead of going through
    ``logging.getLogger``: the global logger registry never releases entries,
    so registering one logger (and one open FileHandler) per instance leaks
    file descriptors across a large batch. The caller owns the handler's
    lifetime — close it via :meth:`AgentLogContext.close` when the instance
    finishes.
    """
    lg = logging.Logger(name, level=logging.DEBUG)
    lg.propagate = False
    h = logging.FileHandler(path, mode="w")
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d %H:%M:%S"))
    lg.addHandler(h)
    return lg


AGENT_MSGS_DIRNAME = "agent_msgs"
INSTANCE_LOG_NAME = "instance.log"


@dataclass
class AgentLogContext:
    """Per-instance logging + trajectory-collection scope.

    * ``info``: the ``instance.log`` file logger for orchestration events.
    * ``agent_msgs_dir``: directory where each agent's message file lives.
      Agents are given a path into this directory at construction time and
      write to it themselves — no stack, no per-message routing.
    * ``trajectories``: a ``{agent_name: agent}`` dict holding live references
      to every registered agent. Saving the trajectory at the end of the run
      pulls each agent's messages and tool definitions for export. The
      registrations are locked so parallel ``agent`` tool calls can't race.
    """

    dir: Path
    instance_id: str = ""
    info: logging.Logger = field(init=False)
    trajectories: dict[str, object] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, dir: Path, instance_id: str = "") -> AgentLogContext:
        dir.mkdir(parents=True, exist_ok=True)
        (dir / AGENT_MSGS_DIRNAME).mkdir(exist_ok=True)
        ctx = cls(dir=dir, instance_id=instance_id)
        name = f"instance.{instance_id}.info" if instance_id else "instance.info"
        ctx.info = make_file_logger(name, dir / INSTANCE_LOG_NAME)
        return ctx

    @property
    def agent_msgs_dir(self) -> Path:
        return self.dir / AGENT_MSGS_DIRNAME

    def agent_msg_path(self, name: str) -> Path:
        """Path for a named agent's message file (e.g. ``"main"``)."""
        return self.agent_msgs_dir / f"{name}.log"

    def next_subagent_path(self, subagent_type: str) -> tuple[Path, str]:
        """Reserve the next auto-numbered subagent file and return ``(path, stem)``."""
        with self._lock:
            n = self._counts.get(subagent_type, 0) + 1
            self._counts[subagent_type] = n
        stem = f"{subagent_type}_{n}"
        return self.agent_msg_path(stem), stem

    def register_agent(self, name: str, agent) -> None:
        """Record ``agent`` under ``name`` for later trajectory export.

        The stored reference is *live*: any ``add_message`` call on the agent
        after registration is visible when the trajectory is serialized.
        """
        with self._lock:
            self.trajectories[name] = agent

    def close(self) -> None:
        """Close the instance.log file handler (releases its fd)."""
        for h in self.info.handlers:
            try:
                h.close()
            except Exception:
                pass
        self.info.handlers.clear()


# --- ContextVar + accessors --------------------------------------------------

_ctx: ContextVar[AgentLogContext | None] = ContextVar("mswea_log_ctx", default=None)


def get_logger() -> logging.Logger:
    """Info/event logger — ``instance.log`` in batch mode, module logger otherwise."""
    ctx = _ctx.get()
    return ctx.info if ctx is not None else logger


def get_log_context() -> AgentLogContext | None:
    return _ctx.get()


@contextmanager
def use_log_context(ctx: AgentLogContext | None) -> Iterator[AgentLogContext | None]:
    """Bind ``ctx`` in the log-context ContextVar for this block.

    Safe for concurrent use across threads: each worker can call this to
    inherit a parent-thread context without sharing a ``contextvars.Context``
    object (which is not re-entrant).
    """
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


__all__ = [
    "logger",
    "add_file_handler",
    "make_file_logger",
    "get_logger",
    "get_log_context",
    "use_log_context",
    "AgentLogContext",
]
