#!/usr/bin/env python3

"""Run mimoagent on SWE-bench instances in batch mode, distributed with Ray.

This is a Ray-based sibling of ``batch.py``. Where ``batch.py`` fans work out to
a single-process ``ThreadPoolExecutor``, this script fans out to N Ray *actors*
that each run their own ``ThreadPoolExecutor``. The actors are scheduled
``SPREAD`` across a Ray cluster, so total concurrency is
``--num-actors * --workers`` and can span many nodes — far more than a single
machine can drive.

It is meant to be launched on the **head node** of an already-running Ray actor
cluster (``ray start`` on every node first). It connects with
``ray.init(address="auto")``.

Two hard requirements for a distributed run:
  * ``--output`` must live on a **shared filesystem** (NFS, a mounted object
    store, ...) visible to every node — trajectories and per-instance logs
    are written by whichever node ran the instance.
  * mimoagent (and its deps) must be importable on every node, since each actor
    imports and calls ``process_instance`` locally.

The actual per-instance work is reused verbatim from ``batch.py``
(:func:`process_instance`), so rollout / reward / recalc / skip-reward behaviour
is identical; only the orchestration layer differs.

Logging:
  * Head node: ``<output>/ray_batch.log`` (+ periodic progress to stdout).
  * Each actor: ``<output>/ray_worker_<id>.log``.
  * Each instance: ``<output>/<unit_id>/instance.log`` + ``agent_msgs/`` (same as
    ``batch.py``, written on whichever node ran it).

Interrupt (^C): worker actors are killed, then every pod this run created is
deleted by a run-scoped label selector (``mimoagent-run=rb-...``) across all
clusters the instances were routed to — Ray SIGKILLs actors, so per-instance
``env.cleanup()`` cannot be relied on and pods would otherwise leak.

Example:
    # on the head node
    python -m mimoagent.run.extra.ray_batch \\
        --dataset data/instances.jsonl \\
        --config configs/example.yaml \\
        -o /shared/runs/exp1 \\
        --num-actors 16 --workers 8
    # => up to 128 instances in flight across the cluster
"""

import json
import os
import random
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ray
import typer
from datasets import load_dataset

# Reuse the work unit + helpers verbatim so behaviour matches batch.py exactly.
from mimoagent.run.extra.batch import (
    assign_config_idx,
    filter_instances,
    load_configs,
    load_jsonl_instances,
    process_instance,
    read_results,
    seed_api_key_cycle,
)
from mimoagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mimoagent on agent instances, distributed across a Ray cluster.

[not dim]
Launch on the Ray head node. Total concurrency = --num-actors * --workers.
--output must be on a shared filesystem visible to every node.
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ============================================================================
# Shared progress tracker (one Ray actor for the whole cluster)
# ============================================================================


@ray.remote
class ProgressTracker:
    """Cluster-wide progress counters, polled by the head node.

    ``report_end`` is idempotent per ``unit_id`` (guarded by ``_ended``) so a
    belt-and-suspenders error report from a worker can't double-count a unit
    that ``process_instance`` already finished in its ``finally`` block.
    """

    def __init__(self, total: int):
        self.total = total
        self.started = 0
        self.done = 0
        self.exit_counts: dict[str, int] = {}  # exit_status -> count
        self.statuses: dict[str, str] = {}  # unit_id -> latest status (active only)
        self._ended: set[str] = set()
        self.start_time = time.time()

    def report_start(self, unit_id: str):
        self.started += 1
        self.statuses[unit_id] = "started"

    def report_status(self, unit_id: str, status: str):
        # Only track units we haven't finished, to bound the dict.
        if unit_id not in self._ended:
            self.statuses[unit_id] = status

    def report_end(self, unit_id: str, exit_status: str):
        if unit_id in self._ended:
            return
        self._ended.add(unit_id)
        self.done += 1
        self.statuses.pop(unit_id, None)
        es = str(exit_status)
        self.exit_counts[es] = self.exit_counts.get(es, 0) + 1

    def get_stats(self) -> dict:
        return {
            "total": self.total,
            "started": self.started,
            "done": self.done,
            "active": max(0, self.started - self.done),
            "exit_counts": dict(self.exit_counts),
        }

    def sample_statuses(self, limit: int = 8) -> list[str]:
        items = list(self.statuses.items())[:limit]
        return [f"{uid}: {st}" for uid, st in items]


# ============================================================================
# Progress-manager shim: satisfies the interface process_instance expects,
# forwarding to the shared ProgressTracker actor. Lives inside each worker.
# ============================================================================


class RayProgressManager:
    """Drop-in for ``RunBatchProgressManager`` as far as ``process_instance`` and
    ``ProgressTrackingAgent`` are concerned. They only call three methods; each
    forwards fire-and-forget to the shared tracker actor (the returned ObjectRef
    is intentionally dropped — same pattern as the reference ultra_fast script).
    """

    def __init__(self, tracker):
        self.tracker = tracker

    def on_instance_start(self, unit_id: str):
        self.tracker.report_start.remote(unit_id)

    def update_instance_status(self, unit_id: str, status: str):
        self.tracker.report_status.remote(unit_id, status)

    def on_instance_end(self, unit_id: str, exit_status: str):
        self.tracker.report_end.remote(unit_id, exit_status)


# ============================================================================
# Worker actor: runs a ThreadPoolExecutor over its slice of work units.
# ============================================================================


@ray.remote
class BatchWorker:
    def __init__(
        self,
        actor_id: int,
        configs: list,
        output_dir: str,
        results_file: str,
        tracker,
        per_actor_workers: int,
        recalc_input_arg,
        recalc_model_names: dict,
        skip_reward: bool,
    ):
        self.actor_id = actor_id
        # ``configs`` is the shared [(config_name, config_dict), ...] list from
        # load_configs; work units carry an index into it. config_name is None
        # for single-config runs (keeps results byte-identical to before).
        self.configs = configs
        self.output_dir = Path(output_dir)
        # Per-actor shard so cross-node writes never contend on one file; the
        # threading.Lock inside append_result still serialises this actor's own
        # threads writing to the shard. Merged into the main file at the end.
        self.results_file = f"{results_file}.shard_{actor_id:03d}"
        self.tracker = tracker
        self.per_actor_workers = per_actor_workers
        self.recalc_input_arg = recalc_input_arg
        self.recalc_model_names = recalc_model_names
        self.skip_reward = skip_reward
        self.pm = RayProgressManager(tracker)

        # Each actor is its own process, so batch.py's module-global api-key
        # cycles must be seeded here. Offset each pool by actor_id so actors
        # don't all start on the same key.
        for _, cfg in configs:
            seed_api_key_cycle(cfg.get("model", {}).get("api_key_pool"), offset=actor_id)

        # Per-actor log file on the shared FS.
        add_file_handler(self.output_dir / f"ray_worker_{actor_id:03d}.log")
        logger.info(f"[actor-{actor_id:03d}] up: {per_actor_workers} workers, shard={self.results_file}")

    def process_batch(self, work_units: list) -> str:
        """Run this actor's assigned work units concurrently. Returns shard name."""
        logger.info(f"[actor-{self.actor_id:03d}] processing {len(work_units)} work units")
        with ThreadPoolExecutor(max_workers=self.per_actor_workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    self.output_dir,
                    self.results_file,
                    self.configs[config_idx][1],
                    self.pm,
                    self.recalc_input_arg,
                    self.recalc_model_names,
                    unit_id,
                    rollout_idx,
                    self.skip_reward,
                    None,  # model_stats
                    self.configs[config_idx][0],  # config_name
                ): unit_id
                for instance, unit_id, rollout_idx, config_idx in work_units
            }
            for future in as_completed(futures):
                unit_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    # process_instance handles its own exceptions and reports
                    # on_instance_end in a finally, so this only fires for
                    # failures around start_instance_logging itself. report_end
                    # is idempotent, so a redundant call here is harmless.
                    logger.error(f"[actor-{self.actor_id:03d}] uncaught error in {unit_id}: {e}", exc_info=True)
                    self.tracker.report_end.remote(unit_id, type(e).__name__)
        logger.info(f"[actor-{self.actor_id:03d}] done")
        return self.results_file


# ============================================================================
# Shard merge / resume helpers
# ============================================================================


def _shard_paths(output_path: Path, results_file: str) -> list[Path]:
    return sorted(output_path.glob(f"{results_file}.shard_*"))


def merge_shards_into_main(output_path: Path, results_file: str) -> int:
    """Fold all shard files into the main results file (last-wins dedup by the
    key ``read_results`` uses: unit_id, else instance_id), then remove shards.
    Returns the number of shard files merged.
    """
    shards = _shard_paths(output_path, results_file)
    if not shards:
        return 0
    main_path = output_path / results_file

    combined: dict[str, dict] = {}
    if main_path.exists():
        combined.update(read_results(main_path))
    for sp in shards:
        # Shard records are from this/last run — let them win over main.
        combined.update(read_results(sp))

    tmp = main_path.with_suffix(main_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in combined.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(main_path)

    for sp in shards:
        sp.unlink(missing_ok=True)
    return len(shards)


def load_done_units(output_path: Path, results_file: str, no_retry: list[str] | None = None) -> set[str]:
    """Unit ids to skip on resume.

    A unit is skipped if it finished without error or exception, OR if its exit_status matches
    one of the ``no_retry`` substrings (case-insensitive) — the latter lets you
    treat known-hopeless errors (e.g. "RuntimeError" from failed git clones) as
    terminal so they aren't retried, while other errors (e.g. timeouts) still are.
    """
    main_path = output_path / results_file
    if not main_path.exists():
        return set()
    no_retry_lc = [s.lower() for s in (no_retry or [])]
    done: set[str] = set()
    for uid, rec in read_results(main_path).items():
        if "exit_status" not in rec:
            continue
        es = str(rec["exit_status"]).lower()
        if ("error" not in es and "exception" not in es) or any(tok in es for tok in no_retry_lc):
            done.add(uid)
    return done


# ============================================================================
# Run-scoped pod sweep
# ============================================================================


def sweep_run_pods(kubeconfigs: set[str | None], namespaces: set[str], label_selector: str) -> None:
    """Best-effort delete of every pod this run created (matched by the
    run-scoped label), across all clusters/namespaces instances were routed to.

    Needed on interrupt: Ray SIGKILLs the worker actors, so the per-instance
    ``finally: env.cleanup()`` never runs and in-flight pods would leak. Also
    run after a normal finish as a safety net for actors that died mid-run.
    """
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    for kubeconfig in sorted(kubeconfigs, key=lambda k: k or ""):
        tag = Path(kubeconfig).name if kubeconfig else "default-kubeconfig"
        try:
            api_client = k8s_config.new_client_from_config(config_file=kubeconfig)
        except Exception as e:
            logger.error(f"[pod-sweep {tag}] cannot load kubeconfig: {e}")
            continue
        try:
            v1 = k8s_client.CoreV1Api(api_client=api_client)
            for namespace in sorted(namespaces):
                _sweep_namespace(v1, kubeconfig, tag, namespace, label_selector)
        finally:
            try:
                api_client.close()
            except Exception:
                pass


def _sweep_namespace(v1, kubeconfig: str | None, tag: str, namespace: str, label_selector: str) -> None:
    try:
        remaining = 0
        for _ in range(3):
            pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector, _request_timeout=60).items
            # Pods already Terminating don't need another delete.
            pods = [p for p in pods if p.metadata.deletion_timestamp is None]
            remaining = len(pods)
            if not pods:
                break
            logger.info(f"[pod-sweep {tag}/{namespace}] deleting {remaining} pod(s) matching {label_selector}")
            v1.delete_collection_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                grace_period_seconds=0,
                propagation_policy="Background",
                _request_timeout=120,
            )
            # Catch pods whose create was still in flight when actors died.
            time.sleep(5)
        if remaining:
            logger.warning(
                f"[pod-sweep {tag}/{namespace}] {remaining} pod(s) may remain; clean up manually with: "
                f"kubectl{' --kubeconfig ' + kubeconfig if kubeconfig else ''} -n {namespace} "
                f"delete pods -l {label_selector}"
            )
        else:
            logger.info(f"[pod-sweep {tag}/{namespace}] no pods left matching {label_selector}")
    except Exception as e:
        logger.error(f"[pod-sweep {tag}/{namespace}] sweep failed: {e}")


# ============================================================================
# Main
# ============================================================================


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option("dataset", "--dataset", help="agent dataset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory (MUST be on a shared filesystem)", rich_help_panel="Basic"),
    results_file: str = typer.Option("results.jsonl", "--results-file", help="Results file name (JSONL). Actors write <name>.shard_NNN, merged into <name> at the end.", rich_help_panel="Basic"),
    num_actors: int = typer.Option(8, "--num-actors", help="Number of Ray actors to spread across the cluster", rich_help_panel="Ray"),
    workers: int = typer.Option(4, "-w", "--workers", help="Worker threads PER actor. Total concurrency = num-actors * workers.", rich_help_panel="Ray"),
    ray_address: str = typer.Option("auto", "--ray-address", help="Ray cluster address. Default 'auto' connects to the Ray cluster running on this machine. Pass a head-node address to use a remote cluster (a bare IP gets ':6379' appended).", rich_help_panel="Ray"),
    actor_num_cpus: float = typer.Option(1.0, "--actor-num-cpus", help="CPUs to reserve per actor (controls packing/spread)", rich_help_panel="Ray"),
    num_rollouts: int = typer.Option(1, "--num-rollouts", help="Run each instance this many times. Output goes to <instance_id>/rollout_<k>/. Default 1 keeps the flat <instance_id>/ layout.", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    no_retry: str = typer.Option("", "--no-retry", help="Comma-separated exit_status substrings to treat as terminal on resume (skip instead of retry). By default error and exception statuses are retried. e.g. 'RuntimeError' to not retry failed git clones.", rich_help_panel="Data selection"),
    config_specs: list[Path] = typer.Option(..., "-c", "--config", help="Path to a config file. Pass multiple times for random_config mode: each work unit is randomly assigned one of the configs (different agents/models per unit).", rich_help_panel="Basic"),
    config_seed: int = typer.Option(42, "--config-seed", help="Seed for the per-unit config assignment in random_config mode (multiple -c). Same seed + same unit id => same config, so resumed runs keep their assignment.", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use (kubernetes, docker, local, cube, modal)", rich_help_panel="Advanced"),
    override_config: str | None = typer.Option(None, "--override-config", help="JSON string to override config fields", rich_help_panel="Advanced"),
    recalc_input: str | None = typer.Option(None, "--recalc-input", help="Re-calculate rewards by applying a patch before running tests; skips agent rollout. Pass a directory to replay each instance's saved model_patch from <dir>/<iid>/reward_extra_info.json, or the literal string 'gt' to apply the dataset's ground-truth patch field.", rich_help_panel="Advanced"),
    skip_reward: bool = typer.Option(False, "--skip-reward", help="Skip reward calculation; only record the agent's exit status", rich_help_panel="Advanced"),
    log_interval: int = typer.Option(30, "--log-interval", help="Seconds between head-node progress lines", rich_help_panel="Ray"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "ray_batch.log")
    logger.info(f"Results will be saved to {output_path}")

    if dataset.endswith(".parquet"):
        instances = list(load_dataset("parquet", data_files=dataset, split="train"))
    elif dataset.endswith(".jsonl"):
        instances = load_jsonl_instances(dataset)
    else:
        instances = list(load_dataset(dataset, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    recalc_input_arg: Path | str | None = None
    recalc_model_names: dict[str, str] = {}
    if recalc_input:
        if recalc_input == "gt":
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
    if num_actors < 1 or workers < 1:
        raise typer.BadParameter("--num-actors and --workers must both be >= 1")

    # Expand instances into work units (unit_id is the output sub-path / dedup key).
    work_units: list[tuple[dict, str, int | None]] = []
    for instance in instances:
        iid = instance["instance_id"]
        if num_rollouts == 1:
            work_units.append((instance, iid, None))
        else:
            for k in range(num_rollouts):
                work_units.append((instance, f"{iid}/rollout_{k}", k))

    # Fold any leftover shards from a previous (possibly crashed) run into the
    # main results file first, so resume sees a single source of truth.
    merged = merge_shards_into_main(output_path, results_file)
    if merged:
        logger.info(f"Merged {merged} leftover shard file(s) from a previous run")

    if not redo_existing:
        no_retry_list = [s.strip() for s in no_retry.split(",") if s.strip()]
        if no_retry_list:
            logger.info(f"Not retrying exit statuses matching: {no_retry_list}")
        done_units = load_done_units(output_path, results_file, no_retry_list)
        if done_units:
            before = len(work_units)
            work_units = [wu for wu in work_units if wu[1] not in done_units]
            logger.info(f"Skipping {before - len(work_units)} existing work units")

    if not work_units:
        logger.info("Nothing to process. All work units already completed.")
        return

    logger.info(f"Running on {len(work_units)} work units ({num_rollouts} rollout(s) each)...")

    # ---- config(s) ----
    # One -c: classic single-config run. Several -c: random_config mode — each
    # work unit is dealt one config (deterministically by unit id, see
    # assign_config_idx), so agents/models can be mixed within one run.
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
            logger.info(f"{prefix}API key pool: {len(api_key_pool)} keys configured (distributed across actors)")
    if not random_config_mode:
        # Single-config: pass name=None so results stay byte-identical to before.
        configs = [(None, configs[0][1])]

    # Deal a config to every work unit (index into the shared ``configs`` list,
    # so actors don't each carry N config copies per unit).
    work_units = [
        (instance, unit_id, rollout_idx, assign_config_idx(unit_id, len(configs), config_seed))
        for instance, unit_id, rollout_idx in work_units
    ]
    if random_config_mode:
        dist = Counter(configs[idx][0] for _, _, _, idx in work_units)
        logger.info("random_config mode: " + ", ".join(f"{n}={c}" for n, c in sorted(dist.items())))

    # ---- run-scoped pod label (for sweep-on-interrupt) ----
    # Tag every pod this run creates with a unique label so the head node can
    # delete them by label selector when actors are killed before their
    # per-instance env.cleanup() could run. Injected into every config's
    # environment block (any of them may be kubernetes-backed).
    run_pod_label: str | None = None
    sweep_kubeconfigs: set[str | None] = set()
    pod_namespaces: set[str] = set()
    run_id = f"rb-{uuid.uuid4().hex[:12]}"
    for _, cfg in configs:
        env_block = cfg.setdefault("environment", {})
        if (env_block.get("environment_class") or "kubernetes") != "kubernetes":
            continue
        # Default mirrors KubernetesEnvironmentConfig.labels so injecting the
        # run label doesn't drop the stock "app: mimoagent" label.
        labels = env_block.setdefault("labels", {"app": "mimoagent"})
        labels["mimoagent-run"] = run_id
        run_pod_label = f"mimoagent-run={run_id}"
        pod_namespaces.add(env_block.get("namespace") or os.getenv("K8S_NAMESPACE", "default"))
        # None = the kubernetes client's default kubeconfig resolution.
        sweep_kubeconfigs.add(env_block.get("kubeconfig") or None)
    if run_pod_label:
        # Instances can be routed to other clusters via routing.kubeconfig;
        # sweep every cluster this run may touch.
        sweep_kubeconfigs |= {
            rk for inst, _, _, _ in work_units if (rk := (inst.get("routing") or {}).get("kubeconfig"))
        }
        logger.info(
            f"Pods labelled {run_pod_label} (namespaces={sorted(pod_namespaces)}, "
            f"{len(sweep_kubeconfigs)} cluster(s)); swept on interrupt/exit"
        )

    # ---- distribute ----
    # Shuffle so long instances don't cluster on one actor (esp. on resume).
    random.shuffle(work_units)
    actor_batches: list[list] = [[] for _ in range(num_actors)]
    for i, wu in enumerate(work_units):
        actor_batches[i % num_actors].append(wu)

    # A bare head-node IP/hostname means "use that remote Ray cluster"; Ray
    # itself wants host:port, so default to the GCS port. "auto" (the default)
    # keeps the original behaviour of joining the cluster on this machine.
    if ray_address not in ("auto", "local") and "://" not in ray_address and ":" not in ray_address:
        ray_address = f"{ray_address}:6379"

    logger.info(
        f"Ray: {num_actors} actors x {workers} workers = {num_actors * workers} max concurrency "
        f"(address={ray_address})"
    )

    # ---- Ray ----
    ray.init(address=ray_address, ignore_reinit_error=True)
    tracker = ProgressTracker.remote(len(work_units))

    actors = [
        BatchWorker.options(
            num_cpus=actor_num_cpus,
            scheduling_strategy="SPREAD",  # spread actors across cluster nodes
        ).remote(
            actor_id=i,
            configs=configs,
            output_dir=str(output_path),
            results_file=results_file,
            tracker=tracker,
            per_actor_workers=workers,
            recalc_input_arg=recalc_input_arg,
            recalc_model_names=recalc_model_names,
            skip_reward=skip_reward,
        )
        for i in range(num_actors)
    ]

    logger.info(f"Submitting batches to {num_actors} actors...")
    start_time = time.time()
    futures = [
        actor.process_batch.remote(batch_units)
        for actor, batch_units in zip(actors, actor_batches)
        if batch_units  # don't submit to actors with no work
    ]

    # ---- poll ----
    last_log = 0.0
    try:
        while True:
            ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=5.0)
            now = time.time()
            if now - last_log >= log_interval or not not_ready:
                stats = ray.get(tracker.get_stats.remote())
                elapsed = now - start_time
                rate = stats["done"] / max(elapsed / 60, 0.01)
                remaining = stats["total"] - stats["done"]
                eta = remaining / max(rate, 0.01)
                pct = stats["done"] * 100 // max(stats["total"], 1)
                exit_str = " ".join(
                    f"{k}={v}" for k, v in sorted(stats["exit_counts"].items(), key=lambda kv: -kv[1])
                ) or "-"
                logger.info(
                    f"progress={stats['done']}/{stats['total']} ({pct}%) | "
                    f"active={stats['active']} | {exit_str} | "
                    f"rate={rate:.1f}/min | elapsed={elapsed/60:.1f}min ETA={eta:.0f}min"
                )
                last_log = now
            if not not_ready:
                break
    except KeyboardInterrupt:
        logger.info(
            "Interrupted — killing actors, sweeping this run's pods, merging shards "
            "(press ^C again to skip cleanup)."
        )
        # Kill workers first so no new pods get created while we sweep. The
        # tracker actor stays alive for the final stats below.
        for actor in actors:
            ray.kill(actor)

    # ---- finish ----
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    stats = ray.get(tracker.get_stats.remote())
    logger.info("Final stats:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Done:  {stats['done']}")
    logger.info("  Exit status breakdown:")
    for k, v in sorted(stats["exit_counts"].items(), key=lambda kv: -kv[1]):
        logger.info(f"    {v:6d}  {k}")

    if run_pod_label:
        sweep_run_pods(sweep_kubeconfigs, pod_namespaces, run_pod_label)

    merged = merge_shards_into_main(output_path, results_file)
    logger.info(f"Merged {merged} shard file(s) into {output_path / results_file}")

    ray.shutdown()


if __name__ == "__main__":
    app()
