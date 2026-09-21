#!/usr/bin/env python3
"""Run Claude Code via the Python Agent SDK. (SERVER SIDE ONLY — runs inside the pod)

Uses the Agent SDK (not ``claude
--print``) so interactive tool calls — AskUserQuestion / ExitPlanMode — that
would block in headless mode are auto-handled programmatically.

The CLI binary is a *pinned standalone* Claude Code executable laid down by
``install-claude-code.sh`` (default ``/opt/mimo-claude/claude``), passed via
``--cli-path``. We deliberately do NOT use the CLI bundled inside the
``claude-agent-sdk`` wheel, so the scaffold version is reproducible.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def _prompt_stream(text):
    """Wrap a string as an AsyncIterable (required when can_use_tool is set)."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": text},
    }


async def dummy_hook(input_data, tool_use_id, context):
    """Dummy PreToolUse hook required to keep the stream open for can_use_tool."""
    return {"continue_": True}


async def auto_permission_handler(tool_name, input_data, context):
    """Auto-approve all tool calls, with special handling for interactive ones."""
    from claude_agent_sdk.types import PermissionResultAllow

    if tool_name == "AskUserQuestion":
        # Auto-select the first option for each question
        questions = input_data.get("questions", [])
        answers = {}
        for q in questions:
            question_text = q.get("question", "")
            options = q.get("options", [])
            if options:
                answers[question_text] = options[0].get("label", "")
            else:
                answers[question_text] = ""
        return PermissionResultAllow(
            updated_input={
                "questions": questions,
                "answers": answers,
            }
        )

    # ExitPlanMode and all other tools: allow as-is
    return PermissionResultAllow(updated_input=input_data)


def _build_options(
    model,
    cwd,
    max_turns,
    stderr_callback=None,
    *,
    continue_session=False,
    effort=None,
    cli_path=None,
    append_system_prompt=None,
    cli_args=None,
    disallowed_tools=None,
):
    """Build ClaudeAgentOptions with can_use_tool and required hooks.

    ``cli_path`` pins the standalone Claude Code binary (see module docstring).
    When ``None`` the SDK falls back to the wheel-bundled CLI.

    ``append_system_prompt`` appends extra instructions on top of the built-in
    ``claude_code`` preset (the preset is preserved, not replaced).
    """
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    if cli_args is not None and not isinstance(cli_args, dict):
        raise TypeError("cli_args must be an object")
    extra_args = dict(cli_args or {})
    for key, value in extra_args.items():
        if not isinstance(key, str) or not key or key.startswith("--"):
            raise ValueError("cli_args keys must be non-empty strings without leading '--'")
        if value is not None and not isinstance(value, str):
            raise TypeError(f"cli_args[{key!r}] must be a string or null")
    if stderr_callback is not None:
        extra_args["debug-to-stderr"] = None

    if disallowed_tools is None:
        disallowed_tools = ["WebSearch"]
    if not isinstance(disallowed_tools, list) or not all(isinstance(tool, str) and tool for tool in disallowed_tools):
        raise TypeError("disallowed_tools must be a list of non-empty strings")

    kwargs = {}
    kwargs["effort"] = effort or "high"
    if cli_path:
        kwargs["cli_path"] = cli_path

    # Raise the SDK transport read buffer. claude-agent-sdk defaults to 1MB
    # (_DEFAULT_MAX_BUFFER_SIZE in _internal/transport/subprocess_cli.py); a
    # single stream-json message larger than that aborts the whole run with
    #   Fatal error in message reader: Failed to decode JSON: JSON message
    #   exceeded maximum buffer size of 1048576 bytes
    # which surfaces as rc=1 / ClaudeCodeError with an unparseable submission.
    # Multimodal tasks blow past 1MB routinely: one Read of a full-page render
    # is a multi-MB base64 tool_result. Measured on an image-heavy SVG design
    # rollout: 1167 of 1490 finished trajectories (81%) died this way.
    # 256MB is a safety valve, not a target — the pod has 8Gi and only one such
    # buffer is live at a time. Override with MIMO_CLAUDE_MAX_BUFFER if needed.
    kwargs["max_buffer_size"] = int(os.environ.get("MIMO_CLAUDE_MAX_BUFFER", 256 * 1024 * 1024))

    system_prompt = {"type": "preset", "preset": "claude_code"}
    if append_system_prompt:
        system_prompt["append"] = append_system_prompt

    def _make_options():
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
            disallowed_tools=disallowed_tools,
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            can_use_tool=auto_permission_handler,
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[dummy_hook])],
            },
            extra_args=extra_args,
            stderr=stderr_callback,
            continue_conversation=continue_session,
            **kwargs,
        )

    try:
        return _make_options()
    except TypeError:
        kwargs.pop("max_buffer_size", None)
        return _make_options()


def _log_message(message, log_file):
    """Append one SDK message to the session log.

    Never echo to stdout: stdout here is the k8s-exec pipe and it is
    O_NONBLOCK, so a full pipe made the previous synchronous retry loop spin
    inside this process's event loop. That froze the loop, the CLI issued no
    further requests, and the rollout was scored as an agent timeout.
    The harness reads this file instead (claude_code.py: _copy_log_out /
    _collect), same as codex does with codex.txt.
    """
    try:
        if hasattr(message, "__dict__"):
            line = json.dumps(message.__dict__, default=str, ensure_ascii=False)
        else:
            line = json.dumps({"message": str(message)}, ensure_ascii=False)
    except (TypeError, ValueError):
        line = json.dumps({"message": str(message)}, ensure_ascii=False)
    log_file.write(line + "\n")
    log_file.flush()


def _extract_result(message):
    """Extract result data from a ResultMessage."""
    return {
        "subtype": getattr(message, "subtype", None),
        "duration_ms": getattr(message, "duration_ms", None),
        "duration_api_ms": getattr(message, "duration_api_ms", None),
        "is_error": getattr(message, "is_error", False),
        "num_turns": getattr(message, "num_turns", None),
        "session_id": getattr(message, "session_id", None),
        "stop_reason": getattr(message, "stop_reason", None),
        "total_cost_usd": getattr(message, "total_cost_usd", None),
        "usage": getattr(message, "usage", None),
        "result": getattr(message, "result", None),
    }


async def _keepalive_stream():
    """An async iterable that never yields, keeping stdin open indefinitely.

    When passed as the prompt to ClaudeSDKClient.connect(), this prevents the
    SDK's stream_input() background task from exhausting the iterator and
    entering its 60-second stdin-close timeout. Without this, _empty_stream()
    (used when prompt=None) returns immediately, triggering the timeout that
    closes stdin while API calls may still be in-flight — causing the CLI to
    abort with "[Request interrupted by user]".

    IMPORTANT: Do NOT catch CancelledError here. Letting it propagate cancels
    stream_input() itself so it never reaches the timeout code path.
    """
    await asyncio.get_running_loop().create_future()  # block forever
    yield {}  # make this an async generator; never reached


async def run(
    instructions,
    model,
    cwd,
    max_turns,
    logs_dir,
    continue_session=False,
    effort=None,
    cli_path=None,
    append_system_prompt=None,
    cli_args=None,
    disallowed_tools=None,
):
    """Run one or more turns via the SDK client."""
    from claude_agent_sdk import ClaudeSDKClient, ResultMessage

    log_path = Path(logs_dir) / "claude-code.txt"
    debug_enabled = os.environ.get("CLAUDE_CODE_DEBUG", "").lower() in ("1", "true")

    debug_file = None
    stderr_callback = None
    if debug_enabled:
        debug_path = Path(logs_dir) / "claude-code-debug.log"
        debug_file = open(debug_path, "a" if continue_session else "w")

        def stderr_callback(line: str) -> None:
            debug_file.write(line + "\n")
            debug_file.flush()

    options = _build_options(
        model,
        cwd,
        max_turns,
        stderr_callback,
        continue_session=continue_session,
        effort=effort,
        cli_path=cli_path,
        append_system_prompt=append_system_prompt,
        cli_args=cli_args,
        disallowed_tools=disallowed_tools,
    )
    result_data = None

    open_mode = "a" if continue_session else "w"

    with open(log_path, open_mode) as log_file:
        client = ClaudeSDKClient(options=options)
        await client.connect(prompt=_keepalive_stream())
        try:
            for instruction in instructions:
                await client.query(_prompt_stream(instruction))

                async for message in client.receive_response():
                    _log_message(message, log_file)
                    if isinstance(message, ResultMessage):
                        result_data = _extract_result(message)
        finally:
            await client.disconnect()

    if debug_file:
        debug_file.close()

    if result_data:
        result_path = Path(logs_dir) / "sdk_result.json"
        result_path.write_text(json.dumps(result_data, default=str, ensure_ascii=False))

    return result_data


def main():
    parser = argparse.ArgumentParser(description="Run Claude Code via Agent SDK")
    parser.add_argument("--instruction", type=str, help="Single-turn instruction")
    parser.add_argument(
        "--instructions-file",
        type=str,
        help="Path to JSON file with multi-turn instructions (list of strings)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cwd", type=str, default="/testbed")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--logs-dir", type=str, default="/logs/agent")
    parser.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        default=False,
        help="Resume the most recent session instead of starting a new one",
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level (low/medium/high/xhigh/max)",
    )
    parser.add_argument(
        "--cli-path",
        type=str,
        default=None,
        help="Path to the pinned standalone Claude Code CLI binary. If omitted, the SDK uses its wheel-bundled CLI.",
    )
    parser.add_argument(
        "--append-system-prompt-file",
        type=str,
        default=None,
        help="Path to a file whose contents are appended to the claude_code "
        "system prompt preset (preset preserved, not replaced).",
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default=None,
        help="JSON object containing cli_args and/or disallowed_tools.",
    )
    args = parser.parse_args()

    if not args.instruction and not args.instructions_file:
        print("Error: must provide --instruction or --instructions-file", file=sys.stderr)
        sys.exit(1)

    if args.instructions_file:
        instructions = json.loads(Path(args.instructions_file).read_text())
    else:
        instructions = [args.instruction]

    append_system_prompt = None
    if args.append_system_prompt_file:
        append_system_prompt = Path(args.append_system_prompt_file).read_text()

    options = {}
    if args.options_file:
        options = json.loads(Path(args.options_file).read_text())
        if not isinstance(options, dict):
            raise TypeError("options file must contain a JSON object")

    result = asyncio.run(
        run(
            instructions,
            args.model,
            args.cwd,
            args.max_turns,
            args.logs_dir,
            continue_session=args.continue_session,
            effort=args.effort,
            cli_path=args.cli_path,
            append_system_prompt=append_system_prompt,
            cli_args=options.get("cli_args"),
            disallowed_tools=options.get("disallowed_tools"),
        )
    )

    if result and result.get("is_error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
