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
        return "Agent"

    @property
    def description(self) -> str:
        # Imported lazily to avoid a circular import at module load time
        # (tools.registry imports this module, and agents.subagents is part of
        # the agents package that depends on tools via DefaultAgent).
        from mimoagent.agents.subagents import available_subagent_types

        types = available_subagent_types()
        bullets = "\n".join(f"- {t}" for t in types)
        return f"""Launch a new agent to handle complex, multi-step tasks. Each agent type has specific capabilities and tools available to it.

Available agent types:
{bullets}

When using the Agent tool, specify a subagent_type parameter to select which agent type to use. If omitted, defaults to a general-purpose explorer.

## When to use
- Open-ended questions that span the codebase, or tasks that match an available agent type.

## When not to use
- If the target is already known, use the direct tool: Read for a known path, the Grep tool for a specific symbol or string. Reserve this tool for open-ended questions.

## Usage notes
- Always include a short `description` summarizing what the agent will do.
- When you launch multiple agents for independent work, send them in a single message with multiple tool uses so they run concurrently.
- When the agent is done, it will return a single message back to you.
- Briefly describe the task and what form of answer you expect.
- Subagents cannot spawn further subagents; do not ask one to delegate.

## Writing the prompt
Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so.

Terse command-style prompts produce shallow, generic work."""

    def get_function_parameters(self) -> dict[str, Any]:
        from mimoagent.agents.subagents import available_subagent_types

        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) description of the task",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task for the agent to perform. Self-contained: the subagent does not see the parent conversation — include all relevant context here.",
                },
                "subagent_type": {
                    "type": "string",
                    "enum": available_subagent_types(),
                    "description": "The type of specialized agent to use for this task. If omitted, a general-purpose explorer is used.",
                },
            },
            "required": ["description", "prompt"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        prompt = params.get("prompt")
        if not prompt:
            raise ToolException("Missing required parameter: prompt")
        description = params.get("description") or ""

        # subagent_type is optional in the schema; pick a sensible default.
        subagent_type = params.get("subagent_type")
        if not subagent_type:
            from mimoagent.agents.subagents import available_subagent_types

            available = available_subagent_types()
            # Prefer "explore" if present; otherwise fall back to the first registered type.
            subagent_type = "explore" if "explore" in available else available[0]

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

        # Lazy imports break the tools <-> agents cycle: the cc package imports
        # this module at package load, but we only need agents.cc.cc_agent once
        # the tool is actually invoked.
        from mimoagent.agents.cc.cc_agent import CCAgent
        from mimoagent.agents.subagents import subagent_kwargs

        try:
            kwargs = subagent_kwargs(subagent_type)
        except KeyError as e:
            raise ToolException(str(e))

        # The shared presets name tools for the original catalogue (lowercase,
        # bash/read only) — CCToolRegistry resolves those names case-
        # insensitively, so they map onto the CC schemas unchanged. On top of
        # that, give the read-only presets the CC-only search tools they are
        # prompted to prefer over raw bash grep/find.
        if subagent_type in ("explore", "plan"):
            names = {spec.get("tool", "").lower() for spec in kwargs.get("tools", [])}
            kwargs["tools"] = list(kwargs.get("tools", [])) + [
                {"tool": t} for t in ("Grep", "Glob") if t.lower() not in names
            ]

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

        parent = context.get("agent")
        parent_antihack = getattr(getattr(parent, "config", None), "antihack", None)
        if parent_antihack is not None:
            # The anti-hack guard has to reach the child too. A child built from
            # preset kwargs alone gets a default config with the guard off, which
            # makes the Agent tool a one-step route to the eval artifacts the
            # parent is blocked from reading.
            kwargs["antihack"] = parent_antihack

        subagent = CCAgent(model=model, env=env, msg_path=msg_path, **kwargs)
        if log_ctx is not None and stem is not None:
            log_ctx.register_agent(stem, subagent)
        # Register on the parent agent so callers can walk the parent →
        # subagents tree directly without depending on the log context
        # (which doesn't exist outside batch mode).
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
                "description": description,
                "exit_status": exit_status,
                "steps": subagent._steps_taken,
                **({"log_file": f"{stem}.log"} if stem else {}),
            },
        )
