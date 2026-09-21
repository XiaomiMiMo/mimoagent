#!/usr/bin/env python3

"""Run mimoagent on SWE-bench instances in batch mode."""

import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
import traceback
from collections import Counter
from itertools import cycle
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from mimoagent.agents.default import DefaultAgent
from mimoagent.agents.factory import make_agent
from mimoagent.agents.user_agent import UserAgentDriver
from mimoagent.config import expand_env_vars, get_config_path
from mimoagent.environments.utils import make_dataset_env
from mimoagent.models import GlobalModelStats, get_model
from mimoagent.models.utils.content import content_text
from mimoagent.run.extra.utils.batch_progress import RunBatchProgressManager
from mimoagent.run.utils.instance_logger import start_instance_logging
from mimoagent.run.utils.save import save_traj
from mimoagent.utils.log import add_file_handler, logger
from mimoagent.utils.tool_call_errors import collect_tool_call_errors

_HELP_TEXT = """Run mimoagent on a dataset of software-engineering tasks.

[not dim]
Documentation: [bold green]https://github.com/XiaomiMiMo/mimoagent[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()
_API_KEY_CYCLE_LOCK = threading.Lock()
# One round-robin cycle per distinct api_key_pool (keyed by the pool tuple), so
# in random_config mode each config's pool rotates independently.
_API_KEY_CYCLES: dict[tuple[str, ...], "cycle[str]"] = {}


def get_next_api_key(pool: list[str] | None) -> str | None:
    """Get the next API key from ``pool`` using round-robin (one shared cycle per pool)."""
    if not pool:
        return None
    with _API_KEY_CYCLE_LOCK:
        key = tuple(pool)
        if key not in _API_KEY_CYCLES:
            _API_KEY_CYCLES[key] = cycle(pool)
        return next(_API_KEY_CYCLES[key])


def seed_api_key_cycle(pool: list[str] | None, offset: int = 0) -> None:
    """Pre-seed ``pool``'s round-robin cycle starting ``offset`` keys in.

    Used by Ray actors (separate processes, each with its own module globals)
    so different actors don't all start on the same key.
    """
    if not pool:
        return
    off = offset % len(pool)
    with _API_KEY_CYCLE_LOCK:
        _API_KEY_CYCLES[tuple(pool)] = cycle(pool[off:] + pool[:off])


def _close_agent(agent, log) -> None:
    close = getattr(agent, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as error:
        log.warning(f"Agent runtime cleanup failed: {error}")


class ProgressTrackingAgent(DefaultAgent):
    """DefaultAgent with per-step progress reporting."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict | None:
        tokens = self.model.token_stats
        status = (
            f"Step {self.model.n_calls + 1:3d} "
            f"(in:{tokens.input_tokens // 1000}k out:{tokens.output_tokens // 1000}k "
            f"cache:{tokens.cache_read_tokens // 1000}k)"
        )
        self.progress_manager.update_instance_status(self.instance_id, status)
        return super().step()


def append_result(
    output_path: Path,
    instance_id: str,
    model_name: str,
    exit_status: str,
    *,
    unit_id: str | None = None,
    rollout_idx: int | None = None,
    config_name: str | None = None,
) -> None:
    """Append one work unit's result as a JSONL line (atomic, no full-file rewrite).

    For multi-rollout runs ``unit_id`` is the per-rollout key
    (``<instance_id>/rollout_<k>``) and is what dedup/skip logic keys on, while
    ``instance_id`` stays the real dataset id and ``rollout_idx`` records which
    rollout this was. Single-rollout runs omit both for byte-identical output.
    ``config_name`` records which config a random_config run assigned to this
    unit; single-config runs omit it.
    """
    record = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "exit_status": exit_status,
    }
    if unit_id is not None:
        record["unit_id"] = unit_id
    if rollout_idx is not None:
        record["rollout"] = rollout_idx
    if config_name is not None:
        record["config"] = config_name
    with _OUTPUT_FILE_LOCK:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_results(path: Path) -> dict[str, dict]:
    """Read a results file into ``{unit_id: record}``.

    The map key is the record's ``unit_id`` when present (multi-rollout runs,
    ``<instance_id>/rollout_<k>``) and falls back to ``instance_id`` otherwise,
    so single-rollout and legacy files behave exactly as before.

    Primary format is JSONL (one record per line, later lines win, corrupt
    lines skipped). Legacy whole-file JSON dicts (results.json from older
    runs) are still readable so ``--recalc-input`` works on old run dirs.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        # Legacy results.json is {instance_id: {record}}. A single-line JSONL
        # file also parses as one dict, but its values are scalars — only
        # treat dict-valued mappings as the legacy format.
        if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
            return data
    results: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("instance_id"):
            key = record.get("unit_id") or record["instance_id"]
            results[key] = record
    return results


def load_jsonl_instances(path: str | Path) -> list[dict]:
    """Load local JSONL without coercing heterogeneous rows into one schema."""
    path = Path(path)
    instances: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                instance = json.loads(line)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(
                    f"Invalid JSON in {path} at line {line_number}: {e.msg}",
                    param_hint="--dataset",
                ) from e
            if not isinstance(instance, dict):
                raise typer.BadParameter(
                    f"Expected a JSON object in {path} at line {line_number}, got {type(instance).__name__}",
                    param_hint="--dataset",
                )
            instances.append(instance)
    return instances


def compact_results(path: Path, drop_instance_ids: set[str]) -> None:
    """Rewrite the JSONL results file: dedup (last wins) and drop the given ids.

    Called once at startup for the instances about to (re)run, so a crash
    mid-instance can't leave a stale 'done' record that skips it next time.
    """
    if not path.exists():
        return
    results = read_results(path)
    kept = [rec for iid, rec in results.items() if iid not in drop_instance_ids]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def process_instance(
    instance: dict,
    output_dir: Path,
    results_file: str,
    config: dict,
    progress_manager: RunBatchProgressManager,
    recalc_input: "Path | str | None" = None,
    recalc_model_names: dict[str, str] | None = None,
    unit_id: str | None = None,
    rollout_idx: int | None = None,
    skip_reward: bool = False,
    model_stats: GlobalModelStats | None = None,
    config_name: str | None = None,
) -> None:
    """Process a single agent work unit.

    A "work unit" is one rollout of one instance. ``unit_id`` is the output
    sub-path / progress key (``<instance_id>/rollout_<k>`` for multi-rollout
    runs); it defaults to ``instance_id`` so single-rollout runs are unchanged.
    ``rollout_idx`` is recorded in the results line when set.

    ``config_name`` identifies which config this unit was assigned in
    random_config mode (multiple ``-c``); it is logged, recorded in the results
    line, and saved into the trajectory's ``extra_info``. ``None`` for
    single-config runs keeps the output byte-identical.

    If ``recalc_input`` is provided, skip the agent rollout and instead apply a
    patch before calling ``calculate_reward``:

    * ``recalc_input == "gt"``: apply the ground-truth ``patch`` field from the
      dataset row (useful for validating that the harness itself is sound — a
      ground-truth run should resolve every instance).
    * otherwise: ``recalc_input`` is treated as a directory of a previous run;
      apply the ``model_patch`` saved under
      ``<recalc_input>/<instance_id>/reward_extra_info.json``. Used to re-score
      old runs without re-invoking the model.
    """
    # Deep copy: worker threads must not share nested dicts (model_kwargs is
    # mutated per instance to inject the per-worker api_key).
    config = copy.deepcopy(config)
    instance_id = instance["instance_id"]
    # ``unit_id`` is the output sub-path and progress/dedup key. It equals
    # ``instance_id`` for single-rollout runs (output layout unchanged) and
    # ``<instance_id>/rollout_<k>`` when running multiple rollouts.
    unit_id = unit_id or instance_id
    instance_dir = output_dir / unit_id
    recalc_mode = recalc_input is not None
    recalc_gt_mode = isinstance(recalc_input, str) and recalc_input == "gt"

    # In recalc mode, load the saved patch up front; skip the instance if missing.
    saved_patch = ""
    recalc_model_name = "recalculate"
    if recalc_gt_mode:
        saved_patch = instance.get("patch", "") or ""
        recalc_model_name = "groundtruth"
        if not saved_patch.strip():
            logger.warning(f"Skipping {instance_id}: dataset has no ground-truth patch")
            return
    elif recalc_mode:
        src_info = recalc_input / instance_id / "reward_extra_info.json"
        if not src_info.exists():
            logger.warning(f"Skipping {instance_id}: no saved reward_extra_info at {src_info}")
            return
        try:
            saved = json.loads(src_info.read_text())
        except Exception as e:
            logger.warning(f"Skipping {instance_id}: cannot read saved reward_extra_info ({e})")
            return
        saved_patch = saved.get("model_patch", "") or ""
        if recalc_model_names and instance_id in recalc_model_names:
            recalc_model_name = recalc_model_names[instance_id]

    # avoid inconsistent state if something here fails and there's leftover previous files
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # Opens ``<instance>/instance.log`` + ``<instance>/agent_msgs/``. Info events
    # (env setup, subagent breadcrumbs, test results) go to instance.log via
    # ``get_logger()``; each agent writes its own conversation to
    # ``agent_msgs/<name>.log`` through its ``msg_path`` attribute.
    # NOTE: ``log`` is deliberately NOT named ``logger`` — assigning to
    # ``logger`` here would make it function-local and break the module-logger
    # uses in the skip paths above (UnboundLocalError).
    with start_instance_logging(output_dir, instance_id, rel_dir=unit_id) as log_ctx:
        log = log_ctx.info
        model = None
        task = None
        if config_name is not None:
            log.info(f"Config (random_config mode): {config_name}")
        if not recalc_mode:
            # Get model config and apply api_key from this config's pool if
            # configured (pool round-robin is shared across threads per pool).
            model_config = config.get("model", {}).copy()
            api_key = get_next_api_key(model_config.pop("api_key_pool", None))
            if api_key:
                model_config.setdefault("model_kwargs", {})["api_key"] = api_key
                log.info(f"Using API key: {api_key[:20]}...")
            model = get_model(config=model_config, shared_stats=model_stats)
            # Optional multi-turn: an instance may carry a ``queries`` list, fed
            # to the agent one after another in a single continuing context
            # (every agent's ``run()`` appends follow-ups to the same session).
            # ``user_message`` instead carries the FIRST user turn as a full
            # message payload (dict with ``content``, which may hold multimodal
            # parts — image_url / input_audio / video_url). Any one of
            # user_message / queries / problem_statement is sufficient.
            task = instance.get("problem_statement") or ""
            user_message = instance.get("user_message")
            queries = [q for q in (instance.get("queries") or []) if q]
            if user_message:
                queries = [user_message, *queries]
            elif not queries:
                queries = [task]
            if not any(queries):
                raise ValueError(f"{instance_id}: instance has none of 'user_message', 'queries', 'problem_statement'")
            # ``task`` still seeds the user-agent driver when present.
            if not task:
                task = queries[0] if isinstance(queries[0], str) else content_text(queries[0].get("content"))

        progress_manager.on_instance_start(unit_id)
        progress_manager.update_instance_status(
            unit_id, f"Pulling/starting {config.get('environment', {}).get('environment_class', 'kubernetes')}"
        )

        agent = None
        extra_info = None
        env = None
        exit_status, result = "Unknown", ""

        try:
            env_config = config.get("environment", {}).copy()
            # anti_hack_cleanup / git_leak_prevention / judge_agent / reward_mode
            # live in the yaml ``environment`` block. make_dataset_env captures
            # them as keyword parameters so they never reach env_kwargs (where
            # filter_env_kwargs would drop them); when absent, class defaults apply.
            env = make_dataset_env(instance, **env_config)
            log.info(f"Dataset Environment: {type(env)}")
            log.info(f"Environment Config: {env.env.config}")
            progress_manager.update_instance_status(unit_id, "Setting up environment")
            env.setup_environment()
            # Log the created pod name so instances can be traced / cleaned up by hand
            logger.info(f"Pod name: {getattr(getattr(env, 'env', None), 'pod_name', None)}")

            if recalc_mode:
                progress_manager.update_instance_status(unit_id, "Applying saved model_patch")
                log.info(f"Recalculate mode: applying saved model_patch ({len(saved_patch)} bytes)")
                if saved_patch.strip():
                    # copy_to (tar-stream) instead of a heredoc: large patches
                    # (>~128KB) blow past the kernel's per-argv limit when the
                    # whole command is passed to `bash -lc` ("argument list too
                    # long"), and heredocs mangle non-utf8 bytes in binary diffs.
                    remote_patch_path = "/tmp/recalc_model.patch"
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".patch") as patch_f:
                        patch_f.write(saved_patch)
                        local_patch_path = patch_f.name
                    try:
                        env.env.copy_to(local_patch_path, remote_patch_path)
                    finally:
                        os.unlink(local_patch_path)
                    apply_res = env.execute(f"git apply --verbose --reject {remote_patch_path}", cwd=env.repo_path)
                    env.execute(f"rm -f {remote_patch_path}")
                    if apply_res.get("returncode", 1) != 0:
                        log.warning(
                            f"{instance_id}: git apply returned rc={apply_res.get('returncode')}; "
                            f"proceeding anyway. Output head: {apply_res.get('output', '')[:2000]}"
                        )
                else:
                    log.info("Saved model_patch is empty; proceeding with no patch applied.")
                exit_status, result = "Recalculated", ""
            else:
                agent_config = dict(config.get("agent", {}))
                # ``agent.type`` selects the scaffold (default: the native
                # tool-calling agent). Popped so it never reaches an agent
                # config dataclass. Configs without it are unchanged.
                agent_type = agent_config.pop("type", "default")
                if agent_type == "default":
                    agent = ProgressTrackingAgent(
                        model,
                        env.env,  # base environment
                        progress_manager=progress_manager,
                        instance_id=unit_id,
                        msg_path=log_ctx.agent_msg_path("main"),
                        **agent_config,
                    )
                else:
                    # Blackbox / third-party agents (e.g. Claude Code) run a
                    # self-contained scaffold inside the pod. They share the
                    # Agent protocol but don't take the progress-manager kwargs.
                    progress_manager.update_instance_status(unit_id, f"Running {agent_type}")
                    # Per-instance scaffold context file (project CLAUDE.md).
                    # Dataclass kwarg filtering drops it for agents that don't
                    # declare the field (codex, mimocode).
                    if instance.get("claudemd"):
                        agent_config.setdefault("claudemd", instance["claudemd"])
                    agent = make_agent(
                        agent_type,
                        model,
                        env.env,  # base environment
                        msg_path=log_ctx.agent_msg_path("main"),
                        **agent_config,
                    )
                log_ctx.register_agent("main", agent)
                # Optional user-agent sidecar: after the main agent's turn, a
                # separate simulated-user agent composes follow-up queries and
                # drives further rounds. Opt-in via a top-level ``user_agent:``
                # config block (requires its own ``model:``).
                user_agent_config = config.get("user_agent") or {}
                if user_agent_config and user_agent_config.get("enabled", True):
                    driver = UserAgentDriver(
                        agent,
                        user_agent_config,
                        on_status=lambda s: progress_manager.update_instance_status(unit_id, s),
                    )
                    exit_status, result = driver.run(task)
                else:
                    # Feed each query in turn; the agent continues the same
                    # session, so later queries build on earlier ones. The last
                    # turn's (status, result) is the run outcome.
                    for turn_idx, query in enumerate(queries):
                        if len(queries) > 1:
                            progress_manager.update_instance_status(
                                unit_id, f"Query turn {turn_idx + 1}/{len(queries)}"
                            )
                            log.info(f"Query turn {turn_idx + 1}/{len(queries)}")
                        exit_status, result = agent.run(query)

                # Code mode owns a persistent sandbox host so store/load works
                # across query turns. All turns are complete now; stop it before
                # reward code reads the final workspace.
                _close_agent(agent, log)

            if skip_reward:
                log.info(f"Instance {instance_id} exit status: {exit_status} (reward calculation skipped)")
            else:
                # The rubric judge (any dataset whose instance carries
                # rubric.rubrics + yaml judge_agent) needs the finished
                # rollout's context — doer agent handle, task, final response —
                # which calculate_reward() deliberately doesn't take. In recalc
                # mode agent/task are None; attach_rollout accepts that (the
                # judge then works from task + patch + repo state alone). A
                # no-op when no judge runs.
                env.attach_rollout(agent=agent, task=task or "", result=result)
                progress_manager.update_instance_status(unit_id, "Calculating reward")
                start_time = time.time()
                reward, test_output, reward_extra_info = env.calculate_reward()
                # Inject tool_call_errors from the agent tree (main + its
                # subagents). In recalc mode no agent was created, so leave the
                # field empty rather than touching ``agent.subagents`` on None.
                agents_for_tce = [agent, *agent.subagents] if agent is not None else []
                reward_extra_info["tool_call_errors"] = collect_tool_call_errors(agents_for_tce)
                (instance_dir / "reward_extra_info.json").write_text(json.dumps(reward_extra_info, indent=4))
                duration = time.time() - start_time
                extra_info = extra_info or {}
                extra_info["reward"] = reward
                extra_info["test_output"] = test_output[-5000:] if len(test_output) > 5000 else test_output
                extra_info["test_duration"] = int(duration)
                log.info(f"Instance {instance_id} reward: {reward}")

                # Log test results to instance log
                log.info("-" * 80)
                log.info("TEST RESULTS:")
                log.info(f"Reward: {reward}")
                log.info(f"Test duration: {duration:.2f} seconds")
                log.info("Test output:")
                log.info(test_output)

                if reward_extra_info.get("error_category"):
                    # Reward-time infra fault (e.g. rubric judge failed to
                    # deliver, testbed corrupted): the 0 is a false reward, not
                    # a real fail. The "Error"/"Exception" substring also keeps
                    # the resume filter from treating the unit as done, so a
                    # re-run retries it instead of skipping.
                    exit_status += " - RewardError"
                elif reward == 0:
                    exit_status += " - Fail"
                else:
                    exit_status += " - Pass"

        except Exception as e:
            log.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}

            # Log error to instance log
            log.error("-" * 80)
            log.error(f"EXCEPTION: {type(e).__name__}")
            log.error(f"Message: {str(e)}")
            log.error(f"Traceback:\n{traceback.format_exc()}")
        finally:
            # Cleanup must run even if saving results raises — otherwise the
            # pod leaks. Hence the inner try/finally.
            try:
                if config_name is not None:
                    extra_info = extra_info or {}
                    extra_info["config"] = config_name
                if not recalc_mode:
                    save_traj(
                        agent,
                        instance_dir / f"{instance_id}.traj.json",
                        exit_status=exit_status,
                        result=result,
                        extra_info=extra_info,
                        instance_id=instance_id,
                        log_context=log_ctx,
                        print_fct=log.info,
                    )
                model_name = model.config.model_name if model is not None else recalc_model_name
                append_result(
                    output_dir / results_file,
                    instance_id,
                    model_name,
                    exit_status,
                    unit_id=unit_id if unit_id != instance_id else None,
                    rollout_idx=rollout_idx,
                    config_name=config_name,
                )
                progress_manager.on_instance_end(unit_id, exit_status)
            finally:
                if agent is not None:
                    _close_agent(agent, log)
                if env:
                    env.cleanup()


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of agent instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != after_filter:
            logger.info(f"Instance slice: {after_filter} -> {after_slice} instances")
    return instances


def deep_update(original: dict, override: dict) -> dict:
    """Recursively update dict `original` with values from `override`."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(original.get(k), dict):
            original[k] = deep_update(original.get(k, {}), v)
        else:
            original[k] = v
    return original


def load_configs(
    config_specs: list[Path],
    *,
    environment_class: str | None = None,
    model: str | None = None,
    override_config: str | None = None,
) -> list[tuple[str, dict]]:
    """Load one or more config files, applying the same CLI overrides to each.

    Returns ``[(config_name, config_dict), ...]``. ``config_name`` is the
    file's stem, deduplicated with ``#2``/``#3`` suffixes when several files
    share one stem — it labels results/trajectories in random_config mode.
    """
    stem_counts: Counter[str] = Counter()
    configs: list[tuple[str, dict]] = []
    for spec in config_specs:
        path = get_config_path(spec)
        config = yaml.safe_load(path.read_text())
        if environment_class is not None:
            config.setdefault("environment", {})["environment_class"] = environment_class
        if model is not None:
            config.setdefault("model", {})["model_name"] = model
        if override_config:
            config = deep_update(config, json.loads(override_config))
        # ${VAR} / ${VAR:-default} in string values: credentials and endpoints
        # come from the environment instead of being written into configs.
        config = expand_env_vars(config)
        stem_counts[path.stem] += 1
        name = path.stem if stem_counts[path.stem] == 1 else f"{path.stem}#{stem_counts[path.stem]}"
        configs.append((name, config))
    return configs


def assign_config_idx(unit_id: str, n_configs: int, seed: int = 42) -> int:
    """Deterministically pick a config index for a work unit (random_config mode).

    Keyed on a stable hash of ``unit_id`` (not submission order, not Python's
    salted ``hash()``), so a resumed run deals the same config to a unit no
    matter how the remaining work is filtered or reordered. Roughly uniform
    across configs.
    """
    digest = hashlib.sha256(f"{seed}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_configs


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option("dataset", "--dataset", help="agent dataset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    results_file: str = typer.Option("results.jsonl", "--results-file", help="Results file name (JSONL, one record per instance)", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    num_rollouts: int = typer.Option(1, "--num-rollouts", help="Run each instance this many times. Output goes to <instance_id>/rollout_<k>/. Default 1 keeps the flat <instance_id>/ layout.", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_specs: list[Path] = typer.Option(..., "-c", "--config", help="Path to a config file. Pass multiple times for random_config mode: each work unit is randomly assigned one of the configs (different agents/models per unit).", rich_help_panel="Basic"),
    config_seed: int = typer.Option(42, "--config-seed", help="Seed for the per-unit config assignment in random_config mode (multiple -c). Same seed + same unit id => same config, so resumed runs keep their assignment.", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use (kubernetes, docker, local, cube, modal)", rich_help_panel="Advanced"),
    override_config: str | None = typer.Option(None, "--override-config", help="JSON string to override config fields", rich_help_panel="Advanced"),
    recalc_input: str | None = typer.Option(None, "--recalc-input", help="Re-calculate rewards by applying a patch before running tests; skips agent rollout. Pass a directory to replay each instance's saved model_patch from <dir>/<iid>/reward_extra_info.json, or the literal string 'gt' to apply the dataset's ground-truth patch field.", rich_help_panel="Advanced"),
    skip_reward: bool = typer.Option(False, "--skip-reward", help="Skip reward calculation; only record the agent's exit status", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "mimoagent.log")

    if dataset.endswith(".parquet"):
        instances = list(load_dataset("parquet", data_files=dataset, split="train"))
    elif dataset.endswith(".jsonl"):
        instances = load_jsonl_instances(dataset)
    else:
        instances = list(load_dataset(dataset, split=split))

    # RL-wrapper parquets nest the task under
    # extra_info.interaction_kwargs.instance (flat rows are the batch.py
    # native format) — unwrap so the same parquet is runnable here.
    if instances and "instance_id" not in instances[0] and "extra_info" in instances[0]:
        instances = [inst["extra_info"]["interaction_kwargs"]["instance"] for inst in instances]
        logger.info(f"Unwrapped {len(instances)} blackbox-schema rows (extra_info.interaction_kwargs.instance)")

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    recalc_input_arg: Path | str | None = None
    recalc_model_names: dict[str, str] = {}
    if recalc_input:
        if recalc_input == "gt":
            # Ground-truth mode: use each row's ``patch`` field directly.
            recalc_input_arg = "gt"
            before = len(instances)
            instances = [inst for inst in instances if (inst.get("patch") or "").strip()]
            logger.info(f"Recalc mode (ground-truth): {before} -> {len(instances)} instances have a ground-truth patch")
        else:
            recalc_input_path = Path(recalc_input)
            if not recalc_input_path.exists():
                raise typer.BadParameter(f"--recalc-input directory does not exist: {recalc_input_path}")
            recalc_input_arg = recalc_input_path
            before = len(instances)
            instances = [
                inst for inst in instances
                if (recalc_input_path / inst["instance_id"] / "reward_extra_info.json").exists()
            ]
            logger.info(f"Recalc mode: {before} -> {len(instances)} instances have saved model_patch in {recalc_input_path}")
            # Old run dirs may carry legacy results.json; try the configured
            # name first, then the legacy default.
            for candidate in (recalc_input_path / results_file, recalc_input_path / "results.json"):
                if not candidate.exists():
                    continue
                try:
                    recalc_model_names = {
                        iid: rec.get("model_name_or_path", "recalculate")
                        for iid, rec in read_results(candidate).items()
                    }
                except Exception as e:
                    logger.warning(f"Could not read model names from {candidate}: {e}")
                break

    if num_rollouts < 1:
        raise typer.BadParameter("--num-rollouts must be >= 1")
    if num_rollouts > 1 and recalc_input:
        raise typer.BadParameter("--num-rollouts > 1 is not supported together with --recalc-input")
    if skip_reward and recalc_input:
        raise typer.BadParameter("--skip-reward makes no sense together with --recalc-input")

    # Expand each instance into ``num_rollouts`` work units. A unit's ``unit_id``
    # is its output sub-path and dedup/skip key: ``instance_id`` for single
    # rollout (flat layout preserved), ``<instance_id>/rollout_<k>`` otherwise.
    work_units: list[tuple[dict, str, int | None]] = []
    for instance in instances:
        iid = instance["instance_id"]
        if num_rollouts == 1:
            work_units.append((instance, iid, None))
        else:
            for k in range(num_rollouts):
                work_units.append((instance, f"{iid}/rollout_{k}", k))

    results_path = output_path / results_file
    if not redo_existing and results_path.exists():
        existing_results = read_results(results_path)
        existing_units = {
            uid for uid, rec in existing_results.items()
            if "exit_status" in rec
            and all(
                marker not in str(rec["exit_status"]).lower()
                for marker in ("error", "exception")
            )
        }
        before = len(work_units)
        work_units = [wu for wu in work_units if wu[1] not in existing_units]
        logger.info(f"Skipping {before - len(work_units)} existing work units")
    # Drop stale records for the units about to (re)run, so a crash mid-unit
    # can't leave behind a 'done' record from a previous run.
    compact_results(results_path, {wu[1] for wu in work_units})
    logger.info(f"Running on {len(work_units)} work units ({num_rollouts} rollout(s) each)...")

    # Load config(s). One -c: classic single-config run (byte-identical output).
    # Several -c: random_config mode — each work unit is dealt one config
    # (deterministically by unit id, see assign_config_idx).
    configs = load_configs(
        config_specs, environment_class=environment_class, model=model, override_config=override_config
    )
    random_config_mode = len(configs) > 1
    if random_config_mode and recalc_input:
        raise typer.BadParameter("multiple -c (random_config mode) makes no sense with --recalc-input")
    for name, cfg in configs:
        prefix = f"[{name}] " if random_config_mode else ""
        logger.info(f"{prefix}Model: [bold]{cfg.get('model', {}).get('model_name', 'NOT SET')}[/bold]")
        api_key_pool = cfg.get("model", {}).get("api_key_pool")
        if api_key_pool:
            logger.info(f"{prefix}API key pool: {len(api_key_pool)} keys configured")

    # Deal a config to every work unit. Single-config runs pass name=None so
    # results/trajectories stay byte-identical to before.
    unit_configs: list[tuple[str | None, dict]] = []
    for _, unit_id, _ in work_units:
        if random_config_mode:
            unit_configs.append(configs[assign_config_idx(unit_id, len(configs), config_seed)])
        else:
            unit_configs.append((None, configs[0][1]))
    if random_config_mode:
        dist = Counter(name for name, _ in unit_configs)
        logger.info("random_config mode: " + ", ".join(f"{n}={c}" for n, c in sorted(dist.items())))

    # Run-level stats: injected into every model instance and read by the
    # progress bar's token counter. Owned by this run, not a process global.
    run_model_stats = GlobalModelStats()
    progress_manager = RunBatchProgressManager(len(work_units), model_stats=run_model_stats)

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                unit_id = futures[future]
                logger.error(f"Error in future for work unit {unit_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(unit_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    results_file,
                    unit_config,
                    progress_manager,
                    recalc_input_arg,
                    recalc_model_names,
                    unit_id,
                    rollout_idx,
                    skip_reward,
                    run_model_stats,
                    config_name,
                ): unit_id
                for (instance, unit_id, rollout_idx), (config_name, unit_config) in zip(work_units, unit_configs)
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
