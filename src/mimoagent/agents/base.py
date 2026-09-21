"""Core agent loop.

`BaseAgent` owns the loop skeleton: system/instance priming, model query, step
dispatch, and termination handling. Concrete agents plug in how a single step
is performed by overriding ``step()``.

Exit semantics
--------------
``run()`` returns a ``(status, message)`` pair. There are three possible statuses:

* ``"Idle"`` — normal exit. The last assistant turn issued no tool call, so the
  loop has nothing to do and hands control back to the caller. The agent is
  resumable: call ``run(next_task)`` on the same instance and the conversation
  (messages + environment) continues from where it left off.
* ``"LimitsExceeded"`` — step or cost budget was hit before any exit.
* ``"ModelQueryError"`` — model query failed unrecoverably.

Only the latter two are exceptional; the idle exit is a plain control-flow
return (``step()`` returning ``None``), not an exception.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from logging import Logger
from pathlib import Path
from typing import Any

from jinja2 import Template

from mimoagent import Environment, Model
from mimoagent.models.utils.content import content_text
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs
from mimoagent.utils.log import get_logger


@dataclass
class AgentConfig:
    """Base config shared by every agent. Subclasses extend with tool-specific fields."""

    system_template: str = "You are a helpful assistant."
    instance_template: str = "Your task: {{task}}"
    step_limit: int = 0


# NonTerminating: surfaced back to the model as a user message, then the loop continues.
# Terminating:    abnormal termination; returned as (exit_status, message).
# The *normal* end of a run (no tool call issued) is NOT an exception: the loop
# detects it from ``step()`` returning ``None`` and returns ("Idle", text).


class NonTerminatingException(Exception):
    """Recoverable condition — the loop continues after showing the message to the model."""


class FormatError(NonTerminatingException):
    """Model output did not match the expected format."""


class TerminatingException(Exception):
    """Abnormal termination — ends the run with a non-``Idle`` status."""


class LimitsExceeded(TerminatingException):
    """Step or cost limit reached."""


class ModelQueryError(TerminatingException):
    """Model query failed unrecoverably."""


class InfraError(TerminatingException):
    """Unrecoverable infrastructure failure (e.g., pod connection timeout).

    Raised by execute_action() when a tool call triggers TransportError from
    the environment layer. Terminates the agent loop; exit_status = "InfraError".
    """


def _model_io_tokens(model: Any) -> tuple[int, int]:
    """Cumulative input/output from ``model.token_stats``, or (0, 0) if absent.

    ``token_stats`` is a running total. Callers snapshot before/after ``query``
    to recover this request's usage.
    """
    stats = getattr(model, "token_stats", None)
    if stats is None:
        return 0, 0
    return int(getattr(stats, "input_tokens", 0) or 0), int(getattr(stats, "output_tokens", 0) or 0)


def _redact_media_data(content):
    """Copy of multimodal content with inline base64 payloads shortened for the msg-file dump."""
    if not isinstance(content, list):
        return content

    def shorten(url):
        if isinstance(url, str) and url.startswith("data:"):
            return f"{url.split(',', 1)[0]},<{len(url)} chars omitted>"
        return url

    redacted = []
    for piece in content:
        if isinstance(piece, dict):
            if piece.get("type") in ("image_url", "video_url"):
                key = piece["type"]
                inner = piece.get(key)
                url = inner.get("url") if isinstance(inner, dict) else inner
                if shorten(url) != url:
                    piece = {**piece, key: {"url": shorten(url)}}
            elif piece.get("type") == "input_audio":
                inner = piece.get("input_audio") or {}
                data = inner.get("data")
                # base64 audio carries no data: prefix; shorten anything that is not a URL
                if isinstance(data, str) and len(data) > 256 and not data.startswith(("http://", "https://")):
                    piece = {**piece, "input_audio": {**inner, "data": f"<{len(data)} chars omitted>"}}
        redacted.append(piece)
    return redacted


class BaseAgent(ABC):
    """Abstract agent: runs the loop, defers the per-step work to subclasses."""

    IDLE_STATUS = "Idle"
    # Trajectory files this agent writes *inside the environment*, if any.
    # Blackbox scaffolds run in the pod and drop their raw session log there;
    # they override this with the in-pod path(s). Native agents write their
    # trajectory only on the host (``msg_path``), so the default is empty.
    # Consumed by sidecars (e.g. the user-agent) that want to read the
    # trajectory from inside the environment without copying.
    ENV_TRAJECTORY_FILES: list[str] = []

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = AgentConfig,
        logger: Logger | None = None,
        msg_path: Path | str | None = None,
        **kwargs,
    ):
        # Unknown yaml keys (e.g. legacy ``cost_limit``) are warned about and
        # dropped instead of raising TypeError.
        self.config = config_class(**filter_dataclass_kwargs(config_class, kwargs))
        self.model = model
        self.env = env
        self.messages: list[dict] = []
        self.extra_template_vars: dict[str, Any] = {}
        # Per-agent step counter. ``model.n_calls`` is a *global* counter
        # shared across a parent agent and any subagents it spawns; enforcing
        # ``step_limit`` against that counter would trip a subagent on its very
        # first query whenever the parent has already used the budget.
        self._steps_taken: int = 0
        # Last ``model.query`` occupancy: that call's input_tokens + output_tokens
        # from the usage block (not cumulative ``token_stats``, not a char heuristic).
        self._last_request_tokens: int = 0
        # One bool per assistant turn taken inside ``step()``: True when the
        # turn produced a tool-call format error (unknown tool / bad JSON / no
        # function name); False otherwise (including idle turns with no tool
        # calls). Populated by concrete agents — base loop never appends.
        self.tool_call_errors: list[bool] = []
        # Subagents this agent has spawned (e.g. via the ``agent`` tool). The
        # ``agent`` tool appends to this list when a subagent is created so
        # callers can walk the parent → children tree without going through
        # the log context. Subagents themselves never spawn further subagents
        # (their tool sets exclude ``agent``), so this list stays one level
        # deep.
        self.subagents: list[BaseAgent] = []
        # Explicit override wins; otherwise ``self.logger`` resolves through
        # the active log context on each access (see the ``logger`` property).
        self._logger_override: Logger | None = logger
        # Optional file that receives a formatted dump of every message this
        # agent sees — one file per agent. Subagents get their own path so
        # parent/child conversations don't mix. ``None`` disables file dump
        # (messages are still appended to ``self.messages`` in memory and
        # announced on ``self.logger`` as a short line).
        self.msg_path: Path | None = Path(msg_path) if msg_path else None
        if self.msg_path:
            self.msg_path.parent.mkdir(parents=True, exist_ok=True)
            self.msg_path.write_text("")  # truncate

    @property
    def logger(self) -> Logger:
        """Info logger for this agent — explicit override, else the current scope."""
        return self._logger_override if self._logger_override is not None else get_logger()

    def run(self, task: str | dict, **kwargs) -> tuple[str, str]:
        """Drive the loop until the agent goes idle or hits an error.

        First call seeds the conversation with the system template and the rendered
        instance template. Subsequent calls on the same instance (``self.messages``
        non-empty) append ``task`` as a new user message, enabling multi-turn usage.

        ``task`` is normally the task text. It may instead be a full
        user-message payload — a dict with ``content`` (which may carry
        multimodal parts: image_url / input_audio / video_url) and optional
        extra fields. A payload is appended verbatim as the user message; the
        instance template is bypassed (a text template can't wrap structured
        content), while the ``{{task}}`` template variable still gets the
        payload's text.

        Returns ``(status, message)``; see module docstring for status semantics.
        """
        first_turn = not self.messages
        task_text = content_text(task.get("content")) if isinstance(task, dict) else task
        self.extra_template_vars |= {"task": task_text, **kwargs}
        if first_turn:
            self.add_message("system", self.render_template(self.config.system_template))
        if isinstance(task, dict):
            self.add_message(**{"role": "user", **task})
        elif first_turn:
            self.add_message("user", self.render_template(self.config.instance_template))
        else:
            self.add_message("user", task)
        while True:
            try:
                if self.step() is None:
                    return self.IDLE_STATUS, self._last_assistant_text()
                self.after_step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
                self.after_step()
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    @abstractmethod
    def step(self) -> dict | None:
        """One iteration.

        Return a dict for a regular step. Return ``None`` to signal that the
        agent has nothing more to do and ``run()`` should exit normally (``Idle``).
        """

    def after_step(self) -> None:
        """Hook after a non-idle step (including recoverable FormatError turns).

        ``DefaultAgent`` uses this for the optional wrap-up hint. Idle and
        terminating exits skip it — the run is already ending.
        """
        return

    def query(self) -> dict:
        """Query the model. Appends the assistant message and returns the response."""
        if 0 < self.config.step_limit <= self._steps_taken:
            raise LimitsExceeded()
        prev_in, prev_out = _model_io_tokens(self.model)
        try:
            response = self.model.query(self.messages, **self.get_model_query_kwargs())
        except Exception as e:
            raise ModelQueryError(f"{e}\n{traceback.format_exc()}") from e
        now_in, now_out = _model_io_tokens(self.model)
        self._last_request_tokens = max(0, (now_in - prev_in) + (now_out - prev_out))
        if not (response.get("content") or response.get("reasoning_content") or response.get("tool_calls")):
            raise LimitsExceeded("Empty assistant response")
        self._steps_taken += 1
        self.add_message("assistant", **response)
        return response

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.extra_template_vars
        if self.model:
            template_vars |= self.model.get_template_vars()
        return Template(template).render(**kwargs, **template_vars)

    def add_message(self, role: str, content: str, **kwargs) -> None:
        msg = {"role": role, "content": content, **kwargs}
        self.messages.append(msg)
        if self.msg_path:
            self._append_msg_to_file(msg)

    def _append_msg_to_file(self, msg: dict) -> None:
        role = msg.get("role", "?")
        content = _redact_media_data(msg.get("content", ""))
        extras = {k: v for k, v in msg.items() if k not in ("role", "content")}
        ts = time.strftime("%m-%d %H:%M:%S")
        header = f"[{ts}] #{len(self.messages):03d} {role.upper()}"
        if role == "tool" and "name" in extras:
            header += f"  tool={extras['name']}"
        if "tool_call_id" in extras:
            header += f"  call_id={extras['tool_call_id']}"
        lines = ["=" * 80, header, "-" * 80]
        if content:
            lines.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2))
        if extras.get("tool_calls"):
            lines.append("")
            lines.append("tool_calls:")
            for i, tc in enumerate(extras["tool_calls"], 1):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                lines.append(f"  [{i}] {name}  id={tc_id}")
                if args:
                    if isinstance(args, str):
                        try:
                            args = json.dumps(json.loads(args), ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    else:
                        args = json.dumps(args, ensure_ascii=False, indent=2)
                    for line in str(args).splitlines():
                        lines.append(f"      {line}")
        leftover = {k: v for k, v in extras.items() if k not in ("name", "tool_call_id", "tool_calls")}
        if leftover:
            lines.append("")
            lines.append(f"extras: {leftover}")
        lines.append("")
        with self.msg_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def get_model_query_kwargs(self) -> dict:
        """Hook for subclasses to inject extra kwargs into `model.query()`."""
        return {}

    def _last_assistant_text(self) -> str:
        """Return the plain-text content of the most recent assistant message, if any."""
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for piece in content:
                    if isinstance(piece, dict):
                        parts.append(piece.get("text") or piece.get("content") or "")
                    else:
                        parts.append(str(piece))
                return "".join(parts)
            return content or ""
        return ""
