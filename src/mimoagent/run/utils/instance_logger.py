"""Per-instance logging bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mimoagent.utils.log import AgentLogContext, use_log_context


@contextmanager
def start_instance_logging(output_dir: Path, instance_id: str, rel_dir: str | None = None) -> Iterator[AgentLogContext]:
    """Open the per-instance log area and scope it for the duration of the block.

    Creates::

        <output_dir>/<rel_dir>/
            instance.log         # orchestration events (info logger)
            agent_msgs/          # one file per agent, written by the agent itself

    ``rel_dir`` defaults to ``instance_id`` (the classic flat layout). For
    multi-rollout runs the caller passes ``<instance_id>/rollout_<k>`` so each
    rollout gets its own directory while ``instance_id`` still labels the
    logger and trajectory.

    Yields the :class:`AgentLogContext` so callers can ask it for the main
    agent's msg file path and for subagent paths when spawning.
    """
    ctx = AgentLogContext.create(output_dir / (rel_dir or instance_id), instance_id=instance_id)
    try:
        with use_log_context(ctx):
            yield ctx
    finally:
        ctx.close()
