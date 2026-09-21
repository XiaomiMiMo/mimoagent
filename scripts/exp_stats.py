#!/usr/bin/env python3
"""
Experiment stats calculator for traj.json files.

Supports the current traj.json shape (``trajs: {name: {messages, tools, ...}}``
+ ``info.tool_calls: {name: {tool: count}}``) and falls back gracefully to the
legacy ``traj.messages`` layout.

Usage:
  python scripts/exp_stats.py <prefix>

Example:
  python scripts/exp_stats.py 0909_qwen3
  python scripts/exp_stats.py outputs/manual-dp-test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

OUTPUTS_ROOT = Path("outputs")
console = Console()


@dataclass
class TrajResult:
    instance: str
    passed: bool
    exit_status: str = ""
    # Rollout index when the run used ``--num-rollouts`` (nested
    # ``<instance>/rollout_<k>/`` layout); ``None`` for the flat one-per-instance
    # layout.
    rollout: int | None = None
    api_calls: int = 0  # total model calls across parent + all subagents
    # Per-agent message counts and per-agent steps (assistant turns).
    agent_steps: dict[str, int] = field(default_factory=dict)
    # Per-agent assistant turns that emitted no tool_calls and no final answer
    # => genuine tool-call format errors. (Empty tool_calls on the *final*
    # assistant turn is the normal "Idle" exit and must not be counted.)
    agent_tc_errors: dict[str, int] = field(default_factory=dict)
    # Per-agent per-tool invocation counts from ``info.tool_calls``.
    agent_tool_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        """Total assistant turns across all agents in this trajectory."""
        return sum(self.agent_steps.values())

    @property
    def tool_call_errors(self) -> int:
        """Total format errors across all agents."""
        return sum(self.agent_tc_errors.values())


def _messages_by_agent(data: dict) -> dict[str, list[dict]]:
    """Return ``{agent_name: messages}``, handling both new and legacy formats."""
    trajs = data.get("trajs")
    if isinstance(trajs, dict) and trajs:
        return {name: (t.get("messages") or []) for name, t in trajs.items()}
    # Legacy: single top-level "traj.messages" was the parent agent.
    legacy = data.get("traj", {}).get("messages") or []
    return {"main": legacy} if legacy else {}


def _base_name(agent_key: str) -> str:
    """Strip numeric suffix so ``explore_1``/``explore_2`` collapse to ``explore``.

    ``main`` (no underscore) and non-numeric suffixes pass through unchanged.
    """
    if "_" not in agent_key:
        return agent_key
    stem, suffix = agent_key.rsplit("_", 1)
    return stem if suffix.isdigit() else agent_key


# Signatures of messages that land in the ``tool`` role when a tool call is
# malformed or rejected. These all originate in ``DefaultAgent._parse_tool_call``
# (FormatError) or ``execute_action`` (wrapped ToolException / NonTerminating).
# Counting these in ``tool`` messages is a direct measure of how often the
# model emitted a malformed tool call.
_TOOL_ERROR_PREFIXES: tuple[str, ...] = (
    "Unknown tool ",
    "Invalid JSON arguments for tool ",
    "Tool call missing function name",
    "Tool arguments for '",
    "Unexpected error executing tool ",
)


def _count_tool_call_errors(messages: list[dict]) -> int:
    """Count ``tool`` messages whose body is a framework-level tool-call error.

    Matches the exception strings raised by ``_parse_tool_call`` /
    ``execute_action`` — these are cases where the model's tool call was
    structurally invalid (unknown tool, bad JSON args, missing name, …) and
    got bounced back. Normal tool runtime errors (e.g. "File not found") are
    NOT counted — those are legitimate tool feedback, not format errors.
    """
    errors = 0
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if any(content.startswith(p) for p in _TOOL_ERROR_PREFIXES):
            errors += 1
    return errors


def read_traj(traj_path: Path) -> TrajResult | None:
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except Exception:
        return None

    info = data.get("info", {}) or {}
    status = info.get("exit_status") or ""
    passed = isinstance(status, str) and "pass" in status.lower()
    api_calls = (info.get("model_stats") or {}).get("api_calls") or 0

    per_agent_msgs = _messages_by_agent(data)
    agent_steps = {n: sum(1 for m in msgs if m.get("role") == "assistant") for n, msgs in per_agent_msgs.items()}
    agent_tc_errors = {n: _count_tool_call_errors(msgs) for n, msgs in per_agent_msgs.items()}
    agent_tool_counts = info.get("tool_calls") or {}
    if not isinstance(agent_tool_counts, dict):
        agent_tool_counts = {}

    # Layout detection:
    #   flat:   <exp>/<instance>/<instance>.traj.json        -> parent = instance
    #   nested: <exp>/<instance>/rollout_<k>/<instance>.traj.json
    #             -> parent = rollout_<k>, grandparent = instance
    parent = traj_path.parent.name
    rollout: int | None = None
    instance = parent
    if parent.startswith("rollout_") and parent[len("rollout_") :].isdigit():
        rollout = int(parent[len("rollout_") :])
        instance = traj_path.parent.parent.name

    return TrajResult(
        instance=instance,
        passed=passed,
        exit_status=status,
        rollout=rollout,
        api_calls=api_calls,
        agent_steps=agent_steps,
        agent_tc_errors=agent_tc_errors,
        agent_tool_counts=agent_tool_counts,
    )


def collect_exp_results(exp_path: Path) -> list[TrajResult]:
    # ``*/*.traj.json`` catches the flat layout; ``*/rollout_*/*.traj.json`` the
    # nested multi-rollout layout. A run dir has one or the other.
    traj_files = list(exp_path.glob("*/*.traj.json")) + list(exp_path.glob("*/rollout_*/*.traj.json"))
    if not traj_files:
        return []
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(read_traj, traj_files))
    return [r for r in results if r is not None]


def find_experiments(prefix: str) -> list[Path]:
    if prefix.startswith("outputs/"):
        prefix = prefix[8:]
    if not OUTPUTS_ROOT.exists():
        return []
    return sorted(d for d in OUTPUTS_ROOT.iterdir() if d.is_dir() and d.name.startswith(prefix))


# --- aggregations ------------------------------------------------------------


def _aggregate_tool_counts(results: list[TrajResult]) -> dict[str, dict[str, int]]:
    """Sum ``agent_tool_counts`` across all results, merging numbered subagents.

    ``explore_1`` / ``explore_2`` / ``explore_3`` → ``explore``. ``main`` stays
    as ``main``. The returned dict maps ``base_agent_name -> {tool: total}``.
    """
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for agent, counts in r.agent_tool_counts.items():
            base = _base_name(agent)
            for tool, n in counts.items():
                agg[base][tool] += n
    return {a: dict(c) for a, c in agg.items()}


def _subagent_invocations(results: list[TrajResult]) -> Counter[str]:
    """Total subagent spawns per base type across all trajectories.

    Each ``<type>_<N>`` key in ``agent_tool_counts`` represents one spawn.
    """
    counter: Counter[str] = Counter()
    for r in results:
        for name in r.agent_tool_counts:
            if name == "main":
                continue
            counter[_base_name(name)] += 1
    return counter


def _error_counts_by_base(results: list[TrajResult]) -> dict[str, int]:
    """Total tool-call errors per base agent name, merging numbered subagents."""
    out: dict[str, int] = defaultdict(int)
    for r in results:
        for agent, n in r.agent_tc_errors.items():
            out[_base_name(agent)] += n
    return dict(out)


# --- rendering ---------------------------------------------------------------


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _render_tool_usage(results: list[TrajResult], title_suffix: str = "") -> Table:
    """Per-agent × per-tool totals + totals, avg/traj, and format-error counts.

    Numbered subagents (``explore_1`` / ``explore_2`` / …) are merged into a
    single row under the base type (``explore``).
    """
    agg = _aggregate_tool_counts(results)
    errors = _error_counts_by_base(results)
    n = len(results)
    all_tools = sorted({t for counts in agg.values() for t in counts})
    agents = sorted(agg, key=lambda a: (a != "main", a))  # main first

    table = Table(title=f"Tool Usage{title_suffix}", show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    for tool in all_tools:
        table.add_column(tool, justify="right")
    table.add_column("total", justify="right", style="bold")
    table.add_column("avg/traj", justify="right")
    table.add_column("tc_errors", justify="right", style="red")

    for agent in agents:
        counts = agg[agent]
        row = [agent]
        row_total = 0
        for tool in all_tools:
            c = counts.get(tool, 0)
            row.append(str(c) if c else "-")
            row_total += c
        row.append(str(row_total))
        row.append(f"{row_total / n:.1f}" if n else "-")
        err = errors.get(agent, 0)
        row.append(str(err) if err else "-")
        table.add_row(*row)
    return table


def _is_error_status(status: str) -> bool:
    """True for statuses that are neither a pass nor a normal test fail.

    ``batch.py`` records exceptions as the bare class name (``TimeoutError``,
    ``RuntimeError``, …) and uncaught/unknown cases as ``Unknown``; none contain
    "pass"/"fail". A "- Fail" is a legitimate non-resolving run, NOT an error.
    """
    s = (status or "").lower()
    return "pass" not in s and "fail" not in s


def _render_status_breakdown(results: list[TrajResult], title: str = "Exit Status Breakdown") -> Table:
    """Pooled exit-status histogram. Errors are highlighted red so a glance at
    the table shows how much of the run died on exceptions vs. ran to a verdict.
    """
    total = len(results)
    counter: Counter[str] = Counter(r.exit_status or "unknown" for r in results)
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Status", width=50)
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    for status, count in sorted(counter.items(), key=lambda x: -x[1]):
        if "pass" in status.lower():
            style = "green"
        elif _is_error_status(status):
            style = "red"
        else:
            style = ""  # "- Fail": ran to a verdict, just didn't resolve
        table.add_row(status[:50], str(count), f"{_pct(count, total):.1f}%", style=style)
    return table


def _render_subagent_summary(results: list[TrajResult]) -> Table | None:
    counter = _subagent_invocations(results)
    if not counter:
        return None
    n = len(results)
    table = Table(title="Subagent Spawns", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("avg/traj", justify="right")
    for typ, count in counter.most_common():
        table.add_row(typ, str(count), f"{count / n:.2f}" if n else "-")
    return table


# --- single / multi views ---------------------------------------------------


def print_single_exp(results: list[TrajResult], exp_name: str) -> None:
    total = len(results)
    passes = sum(1 for r in results if r.passed)
    pass_rate = _pct(passes, total)

    all_steps = [r.steps for r in results if r.steps > 0]
    avg_steps = sum(all_steps) / len(all_steps) if all_steps else 0
    avg_api_calls = sum(r.api_calls for r in results) / total if total else 0

    trajs_with_tc_errors = sum(1 for r in results if r.tool_call_errors > 0)
    total_tc_errors = sum(r.tool_call_errors for r in results)
    error_trajs = sum(1 for r in results if _is_error_status(r.exit_status))

    summary = Text()
    summary.append("Experiment: ", style="bold")
    summary.append(f"{exp_name}\n", style="cyan")
    summary.append("Total instances: ", style="bold")
    summary.append(f"{total}\n")
    summary.append("Passed: ", style="bold")
    summary.append(f"{passes}", style="green")
    summary.append(f" / {total} (")
    summary.append(f"{pass_rate:.2f}%", style="green bold")
    summary.append(")\n")
    summary.append("Errored (exception/unknown): ", style="bold")
    summary.append(f"{error_trajs}", style="red" if error_trajs else "")
    summary.append(f" / {total} ({_pct(error_trajs, total):.2f}%)\n")
    summary.append("Avg assistant turns (all agents): ", style="bold")
    summary.append(f"{avg_steps:.2f}\n")
    summary.append("Avg api_calls (parent + subagents): ", style="bold")
    summary.append(f"{avg_api_calls:.2f}\n")
    summary.append("Trajs with TC format errors: ", style="bold")
    summary.append(f"{trajs_with_tc_errors}/{total} ({_pct(trajs_with_tc_errors, total):.2f}%)  ")
    summary.append(f"[total errors: {total_tc_errors}]", style="dim")
    console.print(Panel(summary, title="[bold]Summary[/bold]", border_style="blue"))

    console.print(_render_status_breakdown(results))

    # Tool usage + subagent spawns
    console.print(_render_tool_usage(results))
    if (sub_table := _render_subagent_summary(results)) is not None:
        console.print(sub_table)

    console.print()
    console.print(f"[bold]pass@1[/bold]: [green bold]{pass_rate:.2f}%[/green bold]")


def print_multi_exp(
    all_results: list[list[TrajResult]],
    exp_names: list[str],
    prefix: str,
    *,
    unit_label: str = "Experiment",
) -> None:
    """Render pass@k / avg@k across ``k`` sample groups.

    ``unit_label`` names what each group is — "Experiment" when the groups are
    separate run dirs (the classic cross-dir view), "Rollout" when they are the
    rollouts of one ``--num-rollouts`` run.
    """
    k = len(all_results)

    instance_passes: dict[str, int] = defaultdict(int)
    for results in all_results:
        for r in results:
            if r.passed:
                instance_passes[r.instance] += 1
            else:
                instance_passes.setdefault(r.instance, 0)

    total_instances = len(instance_passes)
    if total_instances == 0:
        console.print("[red]No instances found.[/red]")
        return

    instances_with_any_pass = sum(1 for p in instance_passes.values() if p > 0)
    exp_pass_rates = [_pct(sum(1 for r in rs if r.passed), len(rs)) for rs in all_results]
    avg_at_k = sum(exp_pass_rates) / len(exp_pass_rates) if exp_pass_rates else 0
    pass_at_k = _pct(instances_with_any_pass, total_instances)

    all_results_flat = [r for rs in all_results for r in rs]
    avg_steps = sum(r.steps for r in all_results_flat) / len(all_results_flat) if all_results_flat else 0

    trajs_with_tc_errors = sum(1 for r in all_results_flat if r.tool_call_errors > 0)
    total_trajs = len(all_results_flat)
    error_trajs = sum(1 for r in all_results_flat if _is_error_status(r.exit_status))

    summary = Text()
    summary.append(f"{'Run' if unit_label == 'Rollout' else 'Prefix'}: ", style="bold")
    summary.append(f"{prefix}\n", style="cyan")
    summary.append(f"{unit_label}s: ", style="bold")
    summary.append(f"{k}\n")
    summary.append("Unique instances: ", style="bold")
    summary.append(f"{total_instances}\n")
    summary.append("Instances with any pass: ", style="bold")
    summary.append(f"{instances_with_any_pass}\n")
    summary.append("Errored trajs (exception/unknown): ", style="bold")
    summary.append(f"{error_trajs}", style="red" if error_trajs else "")
    summary.append(f" / {total_trajs} ({_pct(error_trajs, total_trajs):.2f}%)\n")
    summary.append("Avg assistant turns per traj: ", style="bold")
    summary.append(f"{avg_steps:.2f}\n")
    summary.append("Trajs with TC format errors: ", style="bold")
    summary.append(f"{trajs_with_tc_errors}/{total_trajs} ({_pct(trajs_with_tc_errors, total_trajs):.2f}%)")
    console.print(Panel(summary, title="[bold]Summary[/bold]", border_style="blue"))

    # Pooled exit-status breakdown across all rollouts/experiments.
    console.print(_render_status_breakdown(all_results_flat, title="Exit Status Breakdown (pooled)"))

    # Per-experiment stats
    exp_table = Table(title=f"Per-{unit_label} Stats", show_header=True, header_style="bold magenta")
    exp_table.add_column(unit_label, max_width=55)
    exp_table.add_column("Pass", justify="right")
    exp_table.add_column("Total", justify="right")
    exp_table.add_column("Rate", justify="right")
    exp_table.add_column("Avg Steps", justify="right")
    exp_table.add_column("TC Err %", justify="right")
    for results, name in zip(all_results, exp_names):
        passes = sum(1 for r in results if r.passed)
        total = len(results)
        rate = _pct(passes, total)
        steps_mean = sum(r.steps for r in results) / total if total else 0
        tc_err = sum(1 for r in results if r.tool_call_errors > 0)
        exp_table.add_row(
            name,
            str(passes),
            str(total),
            f"[green]{rate:.2f}%[/green]",
            f"{steps_mean:.1f}",
            f"{_pct(tc_err, total):.1f}%",
        )
    console.print(exp_table)

    # Progressive metrics
    prog_table = Table(title="Progressive Metrics", show_header=True, header_style="bold magenta")
    prog_table.add_column("k", justify="center")
    prog_table.add_column("avg@k", justify="right")
    prog_table.add_column("pass@k", justify="right")
    cumul_instances: set[str] = set()
    cumul_passed: set[str] = set()
    cumul_rates: list[float] = []
    for i, results in enumerate(all_results):
        for r in results:
            cumul_instances.add(r.instance)
            if r.passed:
                cumul_passed.add(r.instance)
        total = len(results)
        passes = sum(1 for r in results if r.passed)
        cumul_rates.append(_pct(passes, total))
        avg_i = sum(cumul_rates) / len(cumul_rates)
        pass_i = _pct(len(cumul_passed), len(cumul_instances))
        prog_table.add_row(str(i + 1), f"[green]{avg_i:.2f}%[/green]", f"[green bold]{pass_i:.2f}%[/green bold]")
    console.print(prog_table)

    # Tool usage pooled across all experiments
    console.print(_render_tool_usage(all_results_flat, title_suffix=" (pooled)"))
    if (sub_table := _render_subagent_summary(all_results_flat)) is not None:
        console.print(sub_table)

    console.print()
    console.print(f"[bold]pass@{k}[/bold]: [green bold]{pass_at_k:.2f}%[/green bold]")
    console.print(f"[bold]avg@{k}[/bold]: [green bold]{avg_at_k:.2f}%[/green bold]")


def group_by_rollout(results: list[TrajResult]) -> tuple[list[list[TrajResult]], list[str]]:
    """Split a single run dir's results into one group per rollout index.

    Returns ``(groups, names)`` ordered by rollout index, so the existing
    pass@k / avg@k machinery treats "rollout k" as the k-th sample. Trajs
    without a rollout marker (shouldn't happen in a nested run) fall under a
    trailing ``flat`` group.
    """
    groups: dict[int | None, list[TrajResult]] = defaultdict(list)
    for r in results:
        groups[r.rollout].append(r)
    keys = sorted(groups, key=lambda k: (k is None, k))
    names = [(f"rollout_{k}" if k is not None else "flat") for k in keys]
    return [groups[k] for k in keys], names


def main():
    parser = argparse.ArgumentParser(description="Compute stats from traj.json files.")
    parser.add_argument("prefix", help="Experiment name prefix (e.g., 0909_qwen3)")
    args = parser.parse_args()

    exp_dirs = find_experiments(args.prefix)
    if not exp_dirs:
        console.print(f"[red]No experiments found matching prefix: {args.prefix}[/red]")
        return

    console.print(f"[dim]Found {len(exp_dirs)} experiment(s), loading...[/dim]")
    all_results, exp_names = [], []
    for exp_path in exp_dirs:
        results = collect_exp_results(exp_path)
        if results:
            all_results.append(results)
            exp_names.append(exp_path.name)
    if not all_results:
        console.print("[red]No valid traj.json files found.[/red]")
        return

    console.print()
    # A single run dir produced with --num-rollouts: the rollouts ARE the pass@k
    # sample axis. Regroup by rollout index and reuse the multi-exp view.
    if len(all_results) == 1 and any(r.rollout is not None for r in all_results[0]):
        groups, names = group_by_rollout(all_results[0])
        if len(groups) == 1:
            print_single_exp(groups[0], exp_names[0])
        else:
            print_multi_exp(groups, names, exp_names[0], unit_label="Rollout")
    elif len(all_results) == 1:
        print_single_exp(all_results[0], exp_names[0])
    else:
        print_multi_exp(all_results, exp_names, args.prefix)


if __name__ == "__main__":
    main()
