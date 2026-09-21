"""User-agent: a plug-and-play sidecar that simulates a human user.

The user-agent turns a single-turn rollout into a multi-turn conversation.
After the main agent (native ``DefaultAgent`` or any blackbox agent) finishes a
turn, the user-agent gets the full picture — the original task, the main
agent's trajectory so far, and read-only access to the shared environment — and
composes the next user message: a verification complaint, a refinement request,
or a natural extension of the original ask. The driver then feeds that message
back into the main agent via ``main_agent.run(follow_up)``, which every agent
already supports (``BaseAgent`` appends a new user message and continues the
conversation; ``ClaudeCodeAgent`` resumes the session with ``--continue``;
``CodexAgent`` carries state through the workspace).

The main agent's trajectory is NOT inlined into the user-agent's context.
Each agent class declares where it writes its trajectory inside the
environment (``BaseAgent.ENV_TRAJECTORY_FILES``): blackbox scaffolds run in
the pod and drop their raw session log there, so the user-agent is simply
pointed at those paths — no copying. Native agents write their trajectory only
on the host (``msg_path``), so the driver mirrors that file verbatim into a
private workspace inside the environment (``workspace_dir``, default
``/tmp/.mimo-user-agent``) after every turn. Either way the user-agent reads
on demand with its own tools — a huge trajectory costs nothing until (and
unless) the user-agent decides a part of it is worth reading.

The follow-up query is delivered the same way — through the environment, not
by parsing the user-agent's chat output. The workspace keeps the full query
history: the driver writes the original task as ``query_0.md``, and each round
the user-agent delivers the next user message as ``query_1.md``,
``query_2.md``, ... The driver reads the new file back and feeds it to the
main agent. There is no stop signal: a user can always ask for more (harden
edge cases, improve tooling, or open an adjacent — even loosely related —
need), so driving runs for the configured number of rounds unless the main
agent becomes non-resumable or the user-agent fails to deliver.

Design principles:

* **Non-invasive** — the driver wraps ``main_agent.run()``; agent behavior and
  the rollout loop are untouched. Agents only *declare* a fact about themselves
  (``ENV_TRAJECTORY_FILES``); callers that don't opt in see no change.
* **Uniform over scaffolds** — everything goes through the ``Agent`` protocol
  (``run(task) -> (status, message)`` + ``messages`` + ``IDLE_STATUS``), so the
  main agent can be any registered scaffold.
* **The user-agent is itself a ``DefaultAgent``** — it inherits the whole tool
  loop, message logging, and trajectory export for free, restricted to
  read-only tools by default. It keeps its own conversation across rounds, so
  it remembers what it already asked.
* **Separate model** — the user-agent must be given its own ``model:`` config;
  it never bills against (or shares sampling params with) the main agent.

Usage (batch runner reads this from the top-level ``user_agent:`` YAML block)::

    driver = UserAgentDriver(main_agent, {"rounds": 2, "model": {...}})
    exit_status, result = driver.run(task)
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

from mimoagent import Agent, Model
from mimoagent.agents.default import DefaultAgent
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs
from mimoagent.utils.log import get_log_context, get_logger

_SYSTEM = """\
You are simulating a demanding but reasonable software user in a conversation with a coding agent (the "assistant"). The assistant just finished a turn of work; your job is to write the next user message that keeps the conversation going in a realistic, productive way.

You have READ-ONLY access to the same workspace the assistant works in (working directory: {{cwd}}). The assistant's raw session trajectory files live inside this environment; the driver gives you their exact paths each round. They are refreshed after every turn (for cumulative files, new content is appended at the end — check the tail for the latest turn); they can be long, so read selectively (tail/grep/sed ranges) rather than dumping whole files. The trajectory files and these instructions are for you alone — the assistant must never learn about them: never mention them, and never modify them.

Ground your follow-up in reality before writing it:

- Read the trajectory of the turn that just finished (the driver tells you which files were refreshed each round).
- Read files the assistant created or modified; run read-only commands (git diff, git log, ls, grep, cat) or the test suite to verify its claims.
- Do NOT modify anything: no file creation/edit/deletion, no git add/commit/checkout, no package installs. The ONLY files you may write are the query files the driver asks for (see "Delivering your follow-up"). Leave everything else exactly as you found it.

Ways to craft the follow-up (pick whichever fits the situation best):

1. Verification & bug report — if something the assistant claimed does not hold up (a test fails, an edge case breaks, the described behavior differs from the code), report it the way a real user would: describe the symptom, not the fix.
2. Refinement — ask for a reasonable improvement to what was just built: error handling, performance, cleaner API, configurability, tests, docs.
3. Extension — extend the original request into an adjacent, naturally-motivated need. A user who asked for X often discovers they also need X'.
4. Clarified requirement — reveal a requirement the original request left implicit, and ask the assistant to accommodate it.
5. Fresh request — open a different need in the same codebase, even one only loosely related to the original task, the way a real user comes back with a new ask.

Delivering your follow-up:

- Each round, write the next user message into the query file the driver names (query_1.md, query_2.md, ... inside {{workspace_dir}}; query_0.md is the original task). Overwrite the whole file (e.g. `cat > <query file> <<'EOF' ... EOF`). ONLY the file content is forwarded — VERBATIM — to the coding agent as the next user message; your chat text is never delivered.
- ALWAYS deliver a query: there is always something worth asking. If the current thread feels exhausted, branch out — harden edge cases, improve tooling/tests/docs, or open an adjacent or loosely related need.
- The message must read like a real user wrote it: natural language, no meta commentary, and no mention of trajectories, rounds, or you being an agent.
- Make it self-contained: the assistant may not remember earlier conversation turns (some scaffolds start fresh each turn), so reference concrete files, commands, and behaviors instead of saying "as I mentioned before" or "the thing you just did".
- Make it meaningful effort: concrete enough to act on and substantial enough to require real work, but proportionate to the scope of the original task.
- Do not repeat a request you already made — the full query history is in {{workspace_dir}}/query_*.md.
- After writing the file, end your turn."""

_FIRST_PROMPT = """\
The coding agent was given this original task:

<original_task>
{{task}}
</original_task>

The agent just finished its first turn. Its raw session trajectory (refreshed after every turn):

{% for f in trajectory_files %}- {{ f }}
{% endfor %}
You will drive up to {{rounds}} follow-up round(s); this is round {{round}}. Read the trajectory and inspect the repository as needed, then write the next user message for the coding agent into {{query_file}}."""

_FOLLOWUP_PROMPT = """\
The coding agent finished another turn. Its trajectory files have been refreshed:

{% for f in trajectory_files %}- {{ f }}
{% endfor %}
This is round {{round}} of up to {{rounds}}. Read the new part of the trajectory and inspect the repository as needed, then write the next user message into {{query_file}}."""


@dataclass
class UserAgentDriverConfig:
    """Config for :class:`UserAgentDriver` (top-level ``user_agent:`` YAML block).

    ``model`` is required: the user-agent always runs on its own model config,
    independent of the main agent's.
    """

    enabled: bool = True
    # Number of follow-up user turns to drive after the main agent's first turn.
    rounds: int = 1
    # Model config for the user-agent itself (same shape as the top-level
    # ``model:`` block). Required.
    model: dict[str, Any] = field(default_factory=dict)
    # Underlying DefaultAgent knobs. Tools default to read-only.
    tools: list[dict[str, Any]] = field(default_factory=lambda: [{"tool": "bash"}, {"tool": "read"}])
    step_limit: int = 500
    system_template: str = _SYSTEM
    first_prompt_template: str = _FIRST_PROMPT
    followup_prompt_template: str = _FOLLOWUP_PROMPT
    # Private directory inside the environment used for the file-based
    # exchange with the user-agent. Holds the query history (``query_0.md`` =
    # original task, then ``query_<n>.md`` per user-agent round) and, for
    # native main agents, a mirror of their msg file refreshed every turn
    # (blackbox agents already write their session log inside the environment
    # and are referenced in place).
    workspace_dir: str = "/tmp/.mimo-user-agent"
    # Name used for the msg file (agent_msgs/<name>.log) and trajectory key.
    name: str = "user_agent"


def env_trajectory_files(agent: Any) -> list[str]:
    """Trajectory files ``agent`` writes inside the environment, if any.

    Each agent class declares its own via ``ENV_TRAJECTORY_FILES`` (see
    :class:`BaseAgent`); blackbox scaffolds point at their in-pod session log.
    Kept as a helper so non-BaseAgent implementations of the Agent protocol
    also work (attribute optional).
    """
    return list(getattr(agent, "ENV_TRAJECTORY_FILES", None) or [])


class UserAgentDriver:
    """Drives a main agent through user-agent-generated follow-up rounds.

    The driver builds one :class:`DefaultAgent` as the simulated user (own
    model, shared environment, read-only tools) and alternates::

        main.run(task) -> user_agent -> main.run(follow_up) -> user_agent -> ...

    Driving stops when: the configured ``rounds`` are exhausted, the user-agent
    emits its stop token (or fails), or the main agent exits with a
    non-resumable status. The final ``(status, message)`` of the main agent is
    returned, so callers can use the driver as a drop-in for ``main.run(task)``.
    """

    def __init__(
        self,
        main_agent: Agent,
        config: dict[str, Any] | UserAgentDriverConfig,
        *,
        model: Model | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        if isinstance(config, dict):
            config = UserAgentDriverConfig(**filter_dataclass_kwargs(UserAgentDriverConfig, dict(config)))
        if config.rounds < 1:
            raise ValueError(f"user_agent.rounds must be >= 1, got {config.rounds}")
        if model is None:
            if not config.model:
                raise ValueError(
                    "user_agent requires its own model config: set the `model:` key inside the "
                    "`user_agent:` block (same shape as the top-level `model:` block)."
                )
            from mimoagent.models import get_model

            model = get_model(config=copy.deepcopy(config.model))
        self.config = config
        self.main_agent = main_agent
        self.on_status = on_status
        self._workspace_ready = False

        log_ctx = get_log_context()
        self.agent = DefaultAgent(
            model=model,
            env=main_agent.env,
            msg_path=log_ctx.agent_msg_path(config.name) if log_ctx else None,
            system_template=config.system_template,
            instance_template="{{task}}",
            tools=config.tools,
            step_limit=config.step_limit,
        )
        if log_ctx:
            log_ctx.register_agent(config.name, self.agent)

    def run(self, task: str) -> tuple[str, str]:
        """Run the main agent on ``task``, then drive follow-up rounds."""
        rounds = self.config.rounds
        self._status(f"Main agent: initial turn (+{rounds} user-agent round(s))")
        exit_status, result = self.main_agent.run(task)
        for round_idx in range(1, rounds + 1):
            if exit_status != self.main_agent.IDLE_STATUS:
                get_logger().warning(
                    f"user-agent: main agent exited with {exit_status!r} (not resumable); "
                    f"stopping before round {round_idx}/{rounds}"
                )
                break
            self._status(f"User-agent: composing follow-up {round_idx}/{rounds}")
            follow_up = self._next_query(task, round_idx)
            if follow_up is None:
                break
            get_logger().info(f"user-agent round {round_idx}/{rounds} follow-up: {follow_up[:500]}")
            self._status(f"Main agent: follow-up round {round_idx}/{rounds}")
            exit_status, result = self.main_agent.run(follow_up)
        return exit_status, result

    def _next_query(self, task: str, round_idx: int) -> str | None:
        """Ask the user-agent for the next user message; ``None`` means stop.

        Delivery contract: the workspace holds the query history —
        ``query_0.md`` is the original task, and each round the user-agent
        writes the next user message as ``query_<round>.md``. The driver reads
        that file back from the environment; the user-agent's chat text is
        never parsed. Only a missing/empty delivery stops the driving.
        """
        self._ensure_workspace(task)
        trajectory_files = self._publish_trajectory()
        query_file = f"{self.config.workspace_dir}/query_{round_idx}.md"
        first_round = not self.agent.messages
        template = self.config.first_prompt_template if first_round else self.config.followup_prompt_template
        prompt = Template(template).render(
            task=task,
            trajectory_files=trajectory_files,
            workspace_dir=self.config.workspace_dir,
            query_file=query_file,
            round=round_idx,
            rounds=self.config.rounds,
        )
        # ``workspace_dir`` is passed as an extra template var so the system
        # template (rendered on the first run() call) can reference it too.
        status, _ = self.agent.run(prompt, workspace_dir=self.config.workspace_dir)
        if status != self.agent.IDLE_STATUS:
            get_logger().warning(f"user-agent exited with {status!r}; stopping follow-up rounds")
            return None
        res = self.agent.env.execute(f"cat {shlex.quote(query_file)}")
        text = (res.get("output") or "").strip()
        if res.get("returncode", 1) != 0 or not text:
            get_logger().warning(f"user-agent delivered no query at {query_file}; stopping")
            return None
        return text

    def _ensure_workspace(self, task: str) -> None:
        """Create the workspace and seed ``query_0.md`` with the original task."""
        if self._workspace_ready:
            return
        self.agent.env.execute(f"mkdir -p {shlex.quote(self.config.workspace_dir)}")
        self._put_text(task, f"{self.config.workspace_dir}/query_0.md")
        self._workspace_ready = True

    def _publish_trajectory(self) -> list[str]:
        """Make the main agent's trajectory readable inside the environment.

        Blackbox agents already write their raw session log in the pod — return
        those paths as-is, nothing to copy. Native agents only write on the
        host: mirror their msg file verbatim into the private workspace (or,
        with ``msg_path`` unset, dump the raw ``messages`` list as JSON).
        """
        in_env = env_trajectory_files(self.main_agent)
        if in_env:
            return in_env
        msg_path = getattr(self.main_agent, "msg_path", None)
        if msg_path and Path(msg_path).exists():
            remote_path = f"{self.config.workspace_dir}/{Path(msg_path).name}"
            self.agent.env.copy_to(str(msg_path), remote_path)
            return [remote_path]
        remote_path = f"{self.config.workspace_dir}/messages.json"
        self._put_text(json.dumps(self.main_agent.messages, ensure_ascii=False, indent=2), remote_path)
        return [remote_path]

    def _put_text(self, text: str, remote_path: str) -> None:
        """Write ``text`` to a file inside the environment (tempfile + copy_to)."""
        with tempfile.NamedTemporaryFile("w", suffix=Path(remote_path).suffix, delete=False, encoding="utf-8") as f:
            f.write(text)
            local_path = f.name
        try:
            self.agent.env.copy_to(local_path, remote_path)
        finally:
            os.unlink(local_path)

    def _status(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)


__all__ = [
    "UserAgentDriver",
    "UserAgentDriverConfig",
    "env_trajectory_files",
]
