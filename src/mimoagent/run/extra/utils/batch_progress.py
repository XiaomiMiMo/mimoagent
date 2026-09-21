"""This module contains an auxiliary class for rendering progress of a batch run.
It's identical to the one used in swe-agent.
"""

import collections
import logging
import time
from datetime import timedelta
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

import yaml
from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mimoagent.models import GlobalModelStats


def _shorten_str(s: str, max_len: int, shorten_left=False) -> str:
    if not shorten_left:
        s = s[: max_len - 3] + "..." if len(s) > max_len else s
    else:
        s = "..." + s[-max_len + 3 :] if len(s) > max_len else s
    return f"{s:<{max_len}}"


class RunBatchProgressManager:
    def __init__(
        self,
        num_instances: int,
        yaml_report_path: Path | None = None,
        model_stats: GlobalModelStats | None = None,
    ):
        """This class manages a progress bar/UI for run-batch

        Args:
            num_instances: Number of task instances
            yaml_report_path: Path to save a yaml report of the instances and their exit statuses
            model_stats: Run-level stats aggregator for the token counter in the
                main progress bar. Inject the same instance into every model of
                the run (``get_model(shared_stats=...)``). ``None`` hides tokens.
        """

        self._model_stats = model_stats
        self._spinner_tasks: dict[str, TaskID] = {}
        """We need to map instance ID to the task ID that is used by the rich progress bar."""

        self._lock = Lock()
        self._start_time = time.time()
        self._total_instances = num_instances

        self._instances_by_exit_status = collections.defaultdict(list)
        self._main_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("[progress.description]{task.description} ({task.fields[token_info]})"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[cyan]{task.fields[eta]}[/cyan]"),
            # Wait 5 min before estimating speed
            speed_estimate_period=60 * 5,
        )
        self._task_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("{task.fields[instance_id]}"),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
        )
        """Task progress bar for individual instances. There's only one progress bar
        with one task for each instance.
        """

        self._main_task_id = self._main_progress_bar.add_task(
            "[cyan]Overall Progress", total=num_instances, token_info="in:0 out:0 cache:0", eta=""
        )

        self.render_group = Group(Table(), self._task_progress_bar, self._main_progress_bar)
        self._yaml_report_path = yaml_report_path

    @property
    def n_completed(self) -> int:
        return sum(len(instances) for instances in self._instances_by_exit_status.values())

    def _get_eta_text(self) -> str:
        """Calculate estimated time remaining based on current progress."""
        try:
            estimated_remaining = (
                (time.time() - self._start_time) / self.n_completed * (self._total_instances - self.n_completed)
            )
            return f"eta: {timedelta(seconds=int(estimated_remaining))}"
        except ZeroDivisionError:
            return ""

    def update_exit_status_table(self):
        # We cannot update the existing table, so we need to create a new one and
        # assign it back to the render group.
        t = Table()
        t.add_column("Exit Status")
        t.add_column("Count", justify="right", style="bold cyan")
        t.add_column("Most recent instances")
        t.show_header = False
        with self._lock:
            t.show_header = True
            # Sort by number of instances in descending order
            sorted_items = sorted(self._instances_by_exit_status.items(), key=lambda x: len(x[1]), reverse=True)
            for status, instances in sorted_items:
                instances_str = _shorten_str(", ".join(reversed(instances)), 70)
                t.add_row(status, str(len(instances)), instances_str)
        assert self.render_group is not None
        self.render_group.renderables[0] = t

    def _update_total_tokens(self) -> None:
        with self._lock:
            update_kwargs = {"eta": self._get_eta_text()}
            if (stats := self._model_stats) is not None:
                update_kwargs["token_info"] = (
                    f"in:{stats.input_tokens // 1000}k out:{stats.output_tokens // 1000}k cache:{stats.cache_read_tokens // 1000}k"
                )
            self._main_progress_bar.update(self._main_task_id, **update_kwargs)

    def update_instance_status(self, instance_id: str, message: str):
        assert self._task_progress_bar is not None
        assert self._main_progress_bar is not None
        with self._lock:
            self._task_progress_bar.update(
                self._spinner_tasks[instance_id],
                status=_shorten_str(message, 45),
                instance_id=_shorten_str(instance_id, 25, shorten_left=True),
            )
        self._update_total_tokens()

    def on_instance_start(self, instance_id: str):
        with self._lock:
            self._spinner_tasks[instance_id] = self._task_progress_bar.add_task(
                description=f"Task {instance_id}",
                status="Task initialized",
                total=None,
                instance_id=instance_id,
            )

    def on_instance_end(self, instance_id: str, exit_status: str | None) -> None:
        self._instances_by_exit_status[exit_status].append(instance_id)
        with self._lock:
            try:
                self._task_progress_bar.remove_task(self._spinner_tasks[instance_id])
            except KeyError:
                pass
            self._main_progress_bar.update(TaskID(0), advance=1, eta=self._get_eta_text())
        self.update_exit_status_table()
        self._update_total_tokens()
        completed = self.n_completed
        elapsed = time.time() - self._start_time
        remaining = self._total_instances - completed
        eta_str = ""
        if completed > 0:
            eta_seconds = elapsed / completed * remaining
            eta_str = f", eta: {timedelta(seconds=int(eta_seconds))}"
        logger.info(
            f"Progress: {completed}/{self._total_instances} | "
            f"elapsed: {timedelta(seconds=int(elapsed))}{eta_str} | "
            f"status: {exit_status} | {instance_id}"
        )
        if self._yaml_report_path is not None:
            self._save_overview_data_yaml(self._yaml_report_path)

    def on_uncaught_exception(self, instance_id: str, exception: Exception) -> None:
        self.on_instance_end(instance_id, f"Uncaught {type(exception).__name__}")

    def print_report(self) -> None:
        """Print complete list of instances and their exit statuses."""
        for status, instances in self._instances_by_exit_status.items():
            print(f"{status}: {len(instances)}")
            for instance in instances:
                print(f"  {instance}")

    def _get_overview_data(self) -> dict:
        """Get data like exit statuses, total costs, etc."""
        return {
            # convert defaultdict to dict because of serialization
            "instances_by_exit_status": dict(self._instances_by_exit_status),
        }

    def _save_overview_data_yaml(self, path: Path) -> None:
        """Save a yaml report of the instances and their exit statuses."""
        with self._lock:
            path.write_text(yaml.dump(self._get_overview_data(), indent=4))
