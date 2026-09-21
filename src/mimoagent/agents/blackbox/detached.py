"""Run a blackbox harness detached from the exec transport.

A blackbox turn is one pod exec that lives for the whole harness session —
hours. On Kubernetes that exec is a websocket through a regional load balancer
with a hard connection lifetime cap (~4h) and no way to re-attach, so a session
that outlives the connection loses its exit status and the instance is written
off as InfraError although the harness kept running. Every harness writes its
real output to pod files, so the stream carried nothing we need:
``Environment.execute_detached`` launches the harness in the background and
polls a marker file with short execs instead. See
``KubernetesEnvironment.execute_detached`` for the mechanics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mimoagent import Environment
from mimoagent.environments import TransportError


def run_detached(
    env: Environment,
    command: str,
    *,
    cwd: str = "",
    timeout: int,
    tag: str,
    log_hint: str,
    logger: logging.Logger,
    idle_files: Sequence[str] = (),
    idle_timeout: int = 0,
) -> tuple[int, str]:
    """Run the harness command and reduce the result to ``(rc, output_tail)``.

    Raises :class:`TransportError` when the run ended without an exit status
    for infrastructure reasons, so the agent escalates to InfraError instead of
    misreporting Completed. ``rc`` is 1 when the outcome is unknown (the client
    deadline or the stall watchdog killed the run): never a success.
    ``log_hint`` names the pod-side log that holds the session for diagnosis.
    """
    detached = getattr(env, "execute_detached", None)
    if detached is None:
        # Not an in-repo environment: fall back to one attached exec and say so,
        # because the run then depends on the transport outliving the session.
        logger.warning(
            f"[{tag}] {type(env).__name__} has no execute_detached; running attached "
            f"(the exec connection must stay up for the whole run)"
        )
        res = env.execute(command, cwd=cwd, timeout=timeout)
    else:
        res = detached(command, cwd=cwd, timeout=timeout, idle_files=list(idle_files), idle_timeout=idle_timeout)

    output = res.get("output", "") or ""
    reason = res.get("reason", "ok")
    rc = res.get("returncode")
    if reason == "transport_error":
        raise TransportError(f"[{tag}] harness run lost its exit status (reason=transport_error); {log_hint}")
    if reason != "ok":
        logger.warning(f"[{tag}] harness run ended with reason={reason} rc={rc}; {log_hint}")
    if rc is None:
        return 1, output
    return int(rc), output
