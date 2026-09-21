"""Optional synchronous MiMo ``actor.run`` tool.

This is the deliberately small RL-safe subset of MiMo-Code's Actor tool. It
supports one blocking child turn and does not expose the background lifecycle
(``spawn``, ``status``, ``wait``, ``cancel``, or ``send``).
"""

from __future__ import annotations

from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.tools.mimocode.state import derive_child_state
from mimoagent.utils.log import get_log_context


class ActorTool(BaseTool):
    """Run a constrained child agent and return its final answer inline."""

    @property
    def name(self) -> str:
        return "actor"

    @property
    def description(self) -> str:
        from mimoagent.agents.mimocode import MIMOCODE_SUBAGENT_PRESETS

        types = "\n".join(
            f"- **{name}** — {preset['description']} Tools: {', '.join(spec['tool'] for spec in preset['tools'])}."
            for name, preset in MIMOCODE_SUBAGENT_PRESETS.items()
        )
        return f"""Launch a new actor (subagent) to handle a focused task and return its final answer.

JSON calls wrap the action payload inside an `operation` object; `action` is the discriminator field.

`run` is the only supported operation in this rollout. It launches the subagent
and BLOCKS the conversation until the subagent finishes; the result is returned
inline.

## When to delegate

Delegate work that is separable from what you are doing now:
- Work that could run alongside your current line of work rather than after it.
  Hand each such piece to an actor instead of interleaving it yourself.
- A focused lookup whose answer you need before the current turn can continue.
- Investigation that would otherwise flood your context with file contents and
  command output you do not need to keep — the child reads all of it and returns
  only the conclusion.

Keep tightly coupled work in your own loop; a child cannot see your conversation.

## Operations (the `operation.action` field selects one)

- run:    required: `subagent_type`, `description`, `prompt`
          returns the subagent's final answer inline

Example:
{{"operation":{{"action":"run","subagent_type":"explore","description":"Find error recovery","prompt":"<full task>"}}}}

## Available subagent types

{types}

The child sees neither the parent conversation nor its reasoning, so the prompt
must be self-contained. Subagents cannot launch another actor."""

    def get_function_parameters(self) -> dict[str, Any]:
        from mimoagent.agents.mimocode import available_mimocode_subagent_types

        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["run"],
                            "description": "Run the child synchronously and return its final answer inline.",
                        },
                        "subagent_type": {
                            "type": "string",
                            "enum": available_mimocode_subagent_types(),
                            "description": "The specialized child preset to use.",
                        },
                        "description": {
                            "type": "string",
                            "description": "A short (3-5 word) description of the task.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": (
                                "The self-contained task for the child. It does not see the parent conversation."
                            ),
                        },
                    },
                    "required": ["action", "subagent_type", "description", "prompt"],
                    "additionalProperties": False,
                }
            },
            "required": ["operation"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        operation = self._parse_operation(params)
        subagent_type = operation["subagent_type"]

        context = context or {}
        env = context.get("env")
        model = context.get("model")
        if env is None:
            raise ToolException("Actor tool requires 'env' in context")
        if model is None:
            raise ToolException(
                "Actor tool requires 'model' in context. The parent agent must pass it "
                "via execute_action's context dict."
            )
        # Gate on the parent's per-agent semaphore so concurrent actor.run
        # calls (tool_parallel_workers > 1) cannot exceed actor_max_concurrency.
        # Missing parent/semaphore (direct construction) runs ungated.
        semaphore = getattr(context.get("agent"), "actor_semaphore", None)
        if semaphore is None:
            return self._execute_child(operation, context, env, model, subagent_type)
        with semaphore:
            return self._execute_child(operation, context, env, model, subagent_type)

    def _execute_child(
        self,
        operation: dict[str, str],
        context: dict[str, Any],
        env: Any,
        model: Any,
        subagent_type: str,
    ) -> ToolOutput:
        # Lazy imports break the tools <-> agents cycle: the mimocode package
        # imports this module at package load, but the child is only needed when
        # actor.run is invoked.
        from mimoagent.agents.mimocode import MimocodeAgent, mimocode_subagent_kwargs

        try:
            kwargs = mimocode_subagent_kwargs(subagent_type)
        except KeyError as exc:
            raise ToolException(str(exc)) from exc

        log_ctx = get_log_context()
        msg_path, stem = (None, None)
        if log_ctx is not None:
            msg_path, stem = log_ctx.next_subagent_path(subagent_type)
            log_ctx.info.info(f"Running actor -> agent_msgs/{stem}.log")

        parent = context.get("agent")
        if parent is not None:
            # The anti-hack guard has to reach the child too. A child built from
            # preset kwargs alone gets a default config with the guard off, which
            # makes actor.run a one-step route to the eval artifacts the parent is
            # blocked from reading.
            kwargs["antihack"] = parent.config.antihack

        child = MimocodeAgent(model=model, env=env, msg_path=msg_path, **kwargs)
        parent_state = context.get("state")
        if isinstance(parent_state, dict):
            child.tool_state = derive_child_state(parent_state)
        if log_ctx is not None and stem is not None:
            log_ctx.register_agent(stem, child)

        if parent is not None:
            parent.subagents.append(child)

        exit_status, message = child.run(operation["prompt"])
        if exit_status == "InfraError":
            raise TransportError(message)

        return ToolOutput(
            output=message,
            success=exit_status == child.IDLE_STATUS,
            metadata={
                "action": "run",
                "subagent_type": subagent_type,
                "description": operation["description"],
                "exit_status": exit_status,
                "steps": child._steps_taken,
                **({"log_file": f"{stem}.log"} if stem else {}),
            },
        )

    @staticmethod
    def _parse_operation(params: Any) -> dict[str, str]:
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")
        operation = params.get("operation")
        if not isinstance(operation, dict):
            raise ToolException("Missing required object parameter: operation")
        if operation.get("action") != "run":
            raise ToolException("Actor only supports operation.action='run'")
        for field_name in ("subagent_type", "description", "prompt"):
            value = operation.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ToolException(f"Missing required parameter: operation.{field_name}")
        return operation
