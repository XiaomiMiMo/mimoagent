"""Agent tool: spawn a focused subagent (explore, plan, ...) and return its answer."""

from __future__ import annotations

from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput
from mimoagent.utils.log import get_log_context


class AgentTool(BaseTool):
    """Spawn a subagent to handle a focused task and return its final answer.

    The subagent runs its own tool-calling loop with a restricted tool set,
    shares this session's ``env`` and ``model``, and returns a single text
    answer. Its internal conversation is NOT appended to the parent agent's
    messages — only the final answer reaches the parent as a tool response.

    Tool context requirements: the parent agent must pass both ``env`` and
    ``model`` in the tool context dict (see ``DefaultAgent.execute_action``).
    """

    @property
    def name(self) -> str:
        return "agent"

    @property
    def description(self) -> str:
        # Imported lazily to avoid a circular import at module load time
        # (tools.registry imports this module, and agents.subagents is part of
        # the agents package that depends on tools via DefaultAgent).
        from mimoagent.agents.subagents import SUBAGENT_PRESETS

        # One bullet per preset, straight from its ``description`` — adding a
        # preset updates this tool description without touching this file.
        bullets = "\n".join(f"- `{name}`: {preset['description']}" for name, preset in SUBAGENT_PRESETS.items())
        return f"""Spawn a subagent to handle a focused task and return its final answer.

Available subagent types and when to use them:
{bullets}

Rules:
- Give a complete, self-contained prompt. The subagent does NOT see your conversation history — include the question and all necessary context.
- The subagent shares this session's filesystem: files it reads are the files you see, and any modification it makes (via its tools or bash side-effects) persists after it returns.
- Multiple `agent` calls issued in the same turn run in parallel.
- For simple one-shot lookups (e.g. read one known file), use the direct tool instead — spawning a subagent adds latency and token cost.
- Subagents cannot spawn further subagents; do not ask one to delegate."""

    def get_function_parameters(self) -> dict[str, Any]:
        from mimoagent.agents.subagents import available_subagent_types

        return {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "enum": available_subagent_types(),
                    "description": "Which subagent preset to spawn.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Complete, self-contained task description. The subagent does not see the parent conversation — include all relevant context here.",
                },
            },
            "required": ["subagent_type", "prompt"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        subagent_type = params.get("subagent_type")
        if not subagent_type:
            raise ToolException("Missing required parameter: subagent_type")
        prompt = params.get("prompt")
        if not prompt:
            raise ToolException("Missing required parameter: prompt")

        context = context or {}
        env = context.get("env")
        model = context.get("model")
        if env is None:
            raise ToolException("Agent tool requires 'env' in context")
        if model is None:
            raise ToolException(
                "Agent tool requires 'model' in context. The parent agent must pass it "
                "via execute_action's context dict."
            )

        # Lazy imports break the tools <-> agents cycle: tools.registry imports
        # this module at package load, but we only need agents.default once the
        # tool is actually invoked.
        from mimoagent.agents.default import DefaultAgent
        from mimoagent.agents.subagents import subagent_kwargs

        try:
            kwargs = subagent_kwargs(subagent_type)
        except KeyError as e:
            raise ToolException(str(e))

        # When a log context is active (batch mode), reserve a fresh
        # ``agent_msgs/<type>_<N>.log`` for this subagent so its conversation
        # doesn't mix with the parent's, and register the agent so its
        # messages are collected into the final ``traj.json``. Outside that
        # context, the subagent has no ``msg_path`` and no registration.
        log_ctx = get_log_context()
        msg_path, stem = (None, None)
        if log_ctx is not None:
            msg_path, stem = log_ctx.next_subagent_path(subagent_type)
            log_ctx.info.info(f"Spawning subagent -> agent_msgs/{stem}.log")

        subagent = DefaultAgent(model=model, env=env, msg_path=msg_path, **kwargs)
        if log_ctx is not None and stem is not None:
            log_ctx.register_agent(stem, subagent)
        # Register on the parent agent so callers can walk the parent →
        # subagents tree directly without depending on the log context
        # (which doesn't exist outside batch mode).
        parent = context.get("agent")
        if parent is not None:
            parent.subagents.append(subagent)
        exit_status, message = subagent.run(prompt)

        # If subagent terminated due to infra failure, propagate to parent
        # so it also aborts immediately (no wasted model query).
        if exit_status == "InfraError":
            raise TransportError(message)

        return ToolOutput(
            output=message,
            success=exit_status == subagent.IDLE_STATUS,
            metadata={
                "subagent_type": subagent_type,
                "exit_status": exit_status,
                "steps": subagent._steps_taken,
                **({"log_file": f"{stem}.log"} if stem else {}),
            },
        )
