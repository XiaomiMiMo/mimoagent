#!/usr/bin/env python3
"""Collect OpenAI Responses API trajectories (from ``OpenAIResponsesModel`` runs)
into a flat SFT-style jsonl, e.g.::

    python scripts/collect_responses_trajs.py outputs/mimoagent-grok-0709-full
    # -> .cache/trajs/mimoagent-grok-0709-full.jsonl

Input layout: ``<output_dir>/<instance_id>/<instance_id>.traj.json`` where each
traj.json has ``trajs: {main: {messages, tools, ...}, <subagent>: {...}}``.
Every traj (main *and* subagents) becomes one jsonl line: ``{"messages", "tools"}``.

Assistant messages from the responses model carry ``responses_items`` (the raw
output items, incl. reasoning). Those are flattened into fields parallel to
``content`` / ``reasoning_content``:

* reasoning ``summary[].text``      -> ``summarized_reasoning_content``
* reasoning ``encrypted_content``   -> ``signature`` (str; list if >1 reasoning item)

``reasoning_content`` is dropped when it merely duplicates the joined summary
(the responses model derives it from the summary), and kept otherwise.
Tool calls are flattened to ``{"name", "arguments": <parsed dict>}`` to match
the existing ``.cache/trajs/*.jsonl`` format.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import typer
from tqdm import tqdm


def convert_tool_call(tc: dict) -> dict:
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            pass  # keep raw string rather than dropping the trajectory
    return {"name": fn.get("name"), "arguments": args}


def convert_assistant(msg: dict) -> dict:
    out: dict = {"role": "assistant", "content": msg.get("content") or ""}

    summaries: list[str] = []
    signatures: list[str] = []
    for item in msg.get("responses_items") or []:
        if item.get("type") != "reasoning":
            continue
        for s in item.get("summary") or []:
            if s.get("text"):
                summaries.append(s["text"])
        if item.get("encrypted_content"):
            signatures.append(item["encrypted_content"])

    summarized = "\n\n".join(summaries)
    reasoning = msg.get("reasoning_content") or ""
    if reasoning and reasoning != summarized:
        out["reasoning_content"] = reasoning
    if summarized:
        out["summarized_reasoning_content"] = summarized
    if signatures:
        out["signature"] = signatures[0] if len(signatures) == 1 else signatures
    if msg.get("tool_calls"):
        out["tool_calls"] = [convert_tool_call(tc) for tc in msg["tool_calls"]]
    return out


def convert_message(msg: dict) -> dict:
    role = msg.get("role")
    if role == "assistant":
        return convert_assistant(msg)
    out = {"role": role, "content": msg.get("content") or ""}
    if role == "tool" and msg.get("tool_call_id"):
        out["tool_call_id"] = msg["tool_call_id"]
    return out


def process_instance(traj_json: Path) -> tuple[int, int, list[str], bool]:
    """Parse one traj.json -> (n_subagents, n_sig_msgs, serialized jsonl lines, skipped_error)."""
    try:
        data = json.loads(traj_json.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0, 0, [], False
    exit_status = data.get("info", {}).get("exit_status") or ""
    if "error" in exit_status.lower():
        return 0, 0, [], True
    n_sub = n_sig_msgs = 0
    lines: list[str] = []
    for name, entry in (data.get("trajs") or {}).items():
        messages = entry.get("messages") or []
        if not messages:
            continue
        if name != "main":
            n_sub += 1
        converted = [convert_message(m) for m in messages]
        n_sig_msgs += sum(1 for m in converted if m.get("signature"))
        lines.append(json.dumps({"messages": converted, "tools": entry.get("tools") or []}, ensure_ascii=False))
    return n_sub, n_sig_msgs, lines, False


def main(
    output_dir: Path = typer.Argument(..., help="Run outputs directory, e.g. outputs/mimoagent-grok-0709-full"),
    output_file: Path = typer.Option(
        None, "-o", "--output", help="Output jsonl path (default: .cache/trajs/<dirname>.jsonl)"
    ),
    workers: int = typer.Option(None, "-j", "--workers", help="Parallel workers (default: cpu count)"),
) -> None:
    if output_file is None:
        output_file = Path(".cache/trajs") / f"{output_dir.name}.jsonl"

    names = [
        e.name for e in tqdm(os.scandir(output_dir), desc="Scanning", unit=" dirs") if e.is_dir(follow_symlinks=False)
    ]
    names.sort()
    traj_files = [output_dir / name / f"{name}.traj.json" for name in names]

    workers = workers or os.cpu_count() or 4
    n_instances = n_trajs = n_sub = n_sig_msgs = n_skipped = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f, ProcessPoolExecutor(max_workers=workers) as pool:
        for sub, sig, lines, skipped in tqdm(
            pool.map(process_instance, traj_files, chunksize=4),
            total=len(traj_files),
            desc="Collecting",
        ):
            if skipped:
                n_skipped += 1
            if lines:
                n_instances += 1
            n_trajs += len(lines)
            n_sub += sub
            n_sig_msgs += sig
            for line in lines:
                f.write(line + "\n")

    typer.echo(
        f"instances={n_instances}  trajs={n_trajs} (subagents={n_sub})  "
        f"assistant msgs with signature={n_sig_msgs}  skipped(error exit_status)={n_skipped}"
    )
    typer.echo(f"Saved {n_trajs} trajectories to {output_file}")


if __name__ == "__main__":
    typer.run(main)
