"""KubernetesEnvironment.execute_detached — launch + poll instead of one
long-lived exec websocket.

The pod side is simulated by scripting ``execute`` (the per-probe transport)
and ``copy_to`` (the wrapper upload); the clock is faked so deadlines and the
poll cadence are deterministic and instant.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from mimoagent.environments import TransportError
from mimoagent.environments.kubernetes import KubernetesEnvironment

RC = KubernetesEnvironment._DETACHED_MARK_RC
RUNNING = KubernetesEnvironment._DETACHED_MARK_RUNNING
GONE = KubernetesEnvironment._DETACHED_MARK_GONE
LAUNCHED = "__MIMO_DETACHED_LAUNCHED"


class FakeClock:
    """time.monotonic()/time.sleep() stand-in: sleeping advances the clock."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def time(self):
        return self.now


def ok(output, rc=0):
    return {"output": output, "returncode": rc, "reason": "ok"}


class FakePod:
    """Answers the execs execute_detached issues, in order of kind.

    ``probes`` is the scripted sequence of probe replies; the launch, the kill
    and the output tail are answered from fixed attributes so a test only has
    to describe the interesting part. Every exec is recorded in ``calls``.
    """

    def __init__(self, probes, *, launch=None, tail="tail-output", kill=None):
        self.probes = list(probes)
        self.launch = launch if launch is not None else ok(f"{LAUNCHED}\n")
        self.tail = tail
        self.kill = kill if kill is not None else ok("")
        self.calls = []
        self.staged = {}

    def execute(self, command, cwd="", timeout=None):
        self.calls.append((command, timeout))
        if LAUNCHED in command:
            return self.launch
        if "kill -TERM" in command:
            return self.kill
        if command.startswith("tail -c"):
            return ok(self.tail)
        if f"echo {RC}" in command:
            if not self.probes:
                raise AssertionError("probe issued after the scripted sequence ran out")
            reply = self.probes.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        raise AssertionError(f"unexpected exec: {command!r}")

    def copy_to(self, src, dst, **kwargs):
        self.staged[dst] = Path(src).read_text()

    @property
    def kills(self):
        return [c for c, _ in self.calls if "kill -TERM" in c]

    @property
    def probe_count(self):
        return sum(1 for c, _ in self.calls if f"echo {RC}" in c)


@pytest.fixture
def clock():
    c = FakeClock()
    # execute_detached lives in DetachedExecMixin; patch the clock it reads.
    with patch("mimoagent.environments.detached.time", c):
        yield c


@pytest.fixture
def k8s_env():
    with patch("mimoagent.environments.kubernetes.config.load_kube_config"):
        with patch("mimoagent.environments.kubernetes.client.ApiClient"):
            with patch("mimoagent.environments.kubernetes.client.CoreV1Api"):
                env = KubernetesEnvironment(image="test:latest")
                env.pod_name = "mimoagent-test1234"
                env.pod_started = True
                yield env


def wire(env, pod):
    env.execute = pod.execute
    env.copy_to = pod.copy_to
    return env


# -- happy path ------------------------------------------------------------------


def test_rc_from_marker_file_and_output_from_tail(k8s_env, clock):
    pod = FakePod([ok(f"{RUNNING}\n"), ok(f"{RUNNING}\n"), ok(f"{RC}0\n")])
    res = wire(k8s_env, pod).execute_detached("run-harness", timeout=3600)

    assert res == {"output": "tail-output", "returncode": 0, "reason": "ok"}
    assert pod.probe_count == 3
    assert pod.kills == []


def test_nonzero_rc_is_ok_reason(k8s_env, clock):
    pod = FakePod([ok(f"{RC}3\n")])
    res = wire(k8s_env, pod).execute_detached("false", timeout=60)
    assert res["returncode"] == 3
    assert res["reason"] == "ok"


def test_marker_is_found_among_login_shell_noise(k8s_env, clock):
    pod = FakePod([ok(f"motd line\n{RUNNING}\ntrailing\n"), ok(f"profile noise\n{RC}0\n")])
    res = wire(k8s_env, pod).execute_detached("x", timeout=60)
    assert res["returncode"] == 0


def test_no_probe_exec_outlives_one_poll(k8s_env, clock):
    """The point of the exercise: with a 5h run, no single exec asks for more
    than the probe timeout — the transport never sees a long connection."""
    probes = [ok(f"{RUNNING}\n")] * 50 + [ok(f"{RC}0\n")]
    pod = FakePod(probes)
    wire(k8s_env, pod).execute_detached("five-hours", timeout=18000, poll_interval=300)
    assert max(t for _, t in pod.calls) <= 120


# -- wrapper / launch ------------------------------------------------------------


def test_wrapper_script_is_staged_and_launched_in_background(k8s_env, clock):
    pod = FakePod([ok(f"{RC}0\n")])
    wire(k8s_env, pod).execute_detached("echo hi", cwd="/work", timeout=42)

    ((path, script),) = pod.staged.items()
    assert path.startswith(f"{KubernetesEnvironment._DETACHED_DIR}/") and path.endswith("/run.sh")
    run_dir = path[: -len("/run.sh")]
    # runs under the pod-side timeout backstop, in a login shell, from cwd
    assert "$__to 42 /bin/bash -lc 'cd /work && echo hi'" in script
    # publishes rc atomically and the pgid the poller kills
    assert f"> {run_dir}/rc.tmp && mv -f {run_dir}/rc.tmp {run_dir}/rc" in script
    assert f"> {run_dir}/pgid" in script

    launch, _ = pod.calls[0]
    # backgrounded on its own (';' before it, not '&&'), stdio on pod files
    assert f"$__ss /bin/bash {run_dir}/run.sh </dev/null >{run_dir}/out 2>&1 & echo {LAUNCHED}" in launch
    assert "&&" not in launch


def test_launch_transport_failure_is_returned_as_is(k8s_env, clock):
    pod = FakePod([], launch={"output": "boom", "returncode": None, "reason": "transport_error"})
    res = wire(k8s_env, pod).execute_detached("x", timeout=60)
    assert res["reason"] == "transport_error"
    assert res["returncode"] is None
    assert pod.probe_count == 0


def test_launcher_shell_failure_reports_like_execute(k8s_env, clock):
    """cd into a missing cwd fails before anything is backgrounded: that is a
    command failure (rc), not a transport failure."""
    pod = FakePod([], launch=ok("bash: cd: /nope: No such file or directory\n", rc=1))
    res = wire(k8s_env, pod).execute_detached("x", cwd="/nope", timeout=60)
    assert res["reason"] == "ok"
    assert res["returncode"] == 1
    assert "No such file" in res["output"]


# -- transport blips vs. real failure --------------------------------------------


def test_transient_probe_failures_are_tolerated(k8s_env, clock):
    pod = FakePod(
        [
            {"output": "socket closed", "returncode": None, "reason": "transport_error"},
            TransportError("apiserver hiccup"),
            {"output": "", "returncode": None, "reason": "client_timeout"},
            ok(f"{RC}0\n"),
        ]
    )
    res = wire(k8s_env, pod).execute_detached("x", timeout=3600)
    assert res == {"output": "tail-output", "returncode": 0, "reason": "ok"}


def test_probe_failures_past_grace_become_transport_error(k8s_env, clock):
    grace = KubernetesEnvironment._DETACHED_TRANSPORT_GRACE_S
    n = int(grace / 20) + 2
    pod = FakePod([TransportError("down")] * n + [ok(f"{RC}0\n")])
    res = wire(k8s_env, pod).execute_detached("x", timeout=36000, poll_interval=20)
    assert res["reason"] == "transport_error"
    assert res["returncode"] is None
    assert "unreachable" in res["output"]


def test_probe_failures_raise_when_configured(k8s_env, clock):
    k8s_env.config.raise_on_transport_error = True
    n = int(KubernetesEnvironment._DETACHED_TRANSPORT_GRACE_S / 20) + 2
    pod = FakePod([TransportError("down")] * n)
    with pytest.raises(TransportError):
        wire(k8s_env, pod).execute_detached("x", timeout=36000, poll_interval=20)


def test_recovery_resets_the_failure_window(k8s_env, clock):
    """A blip, a good probe, another blip: the grace window restarts, so two
    short outages never add up to one long one."""
    half = int(KubernetesEnvironment._DETACHED_TRANSPORT_GRACE_S / 20 / 2) + 1
    pod = FakePod([TransportError("a")] * half + [ok(f"{RUNNING}\n")] + [TransportError("b")] * half + [ok(f"{RC}0\n")])
    res = wire(k8s_env, pod).execute_detached("x", timeout=36000, poll_interval=20)
    assert res["reason"] == "ok"


def test_vanished_process_needs_two_probes(k8s_env, clock):
    """The first probe can race the wrapper writing its pgid file, so one GONE
    is not a verdict; GONE then RC is a normal completion."""
    pod = FakePod([ok(f"{GONE}\n"), ok(f"{RC}0\n")])
    res = wire(k8s_env, pod).execute_detached("x", timeout=60)
    assert res["reason"] == "ok"
    assert res["returncode"] == 0


def test_vanished_process_is_transport_error(k8s_env, clock):
    pod = FakePod([ok(f"{GONE}\n"), ok(f"{GONE}\n")])
    res = wire(k8s_env, pod).execute_detached("x", timeout=3600)
    assert res["reason"] == "transport_error"
    assert "vanished" in res["output"]
    assert len(pod.kills) == 1


# -- timeouts ----------------------------------------------------------------------


def test_pod_timeout_rc124_reaps_orphans(k8s_env, clock):
    pod = FakePod([ok(f"{RC}124\n")])
    res = wire(k8s_env, pod).execute_detached("sleep 999", timeout=10)
    assert res["returncode"] == 124
    assert res["reason"] == "pod_timeout"
    assert "timed out (pod)" in res["output"]
    assert len(pod.kills) == 1


def test_client_deadline_kills_group(k8s_env, clock):
    """Pod-side timeout never reported: the client kills the group once its
    own deadline (behind the pod's) passes and reports an unknown outcome."""
    pod = FakePod([ok(f"{RUNNING}\n")] * 100)
    res = wire(k8s_env, pod).execute_detached("hang", timeout=60, poll_interval=20)
    assert res["reason"] == "client_timeout"
    assert res["returncode"] is None
    assert "timed out (client)" in res["output"]
    assert len(pod.kills) == 1
    # kill happens before the tail is read, so the tail reflects the final state
    kinds = ["kill" if "kill -TERM" in c else "tail" if c.startswith("tail -c") else None for c, _ in pod.calls]
    assert [k for k in kinds if k] == ["kill", "tail"]


# -- stall watchdog ------------------------------------------------------------------


def test_watchdog_off_by_default_ignores_idle_files(k8s_env, clock):
    pod = FakePod([ok(f"{RUNNING}\n"), ok(f"{RC}0\n")])
    wire(k8s_env, pod).execute_detached("x", timeout=60, idle_files=["/tmp/log"])
    probe, _ = next(c for c in pod.calls if f"echo {RC}" in c[0])
    assert "stat -c" not in probe


def test_watchdog_kills_after_idle_timeout(k8s_env, clock):
    pod_now = 5_000_000
    pod = FakePod(
        [
            ok(f"{RUNNING} {pod_now} {pod_now - 100}\n"),  # active 100s ago: fine
            ok(f"{RUNNING} {pod_now + 20} {pod_now - 100}\n"),  # 120s idle: fine
            ok(f"{RUNNING} {pod_now + 400} {pod_now - 100}\n"),  # 500s idle: stall
        ]
    )
    res = wire(k8s_env, pod).execute_detached(
        "x", timeout=36000, idle_files=["/tmp/a.log", "/tmp/b log"], idle_timeout=300
    )
    assert res["reason"] == "stall"
    assert res["returncode"] is None
    assert "stalled" in res["output"]
    assert len(pod.kills) == 1
    probe, _ = next(c for c in pod.calls if f"echo {RC}" in c[0])
    # both files, quoted, newest mtime wins
    assert "stat -c %Y /tmp/a.log '/tmp/b log' 2>/dev/null | sort -n | tail -1" in probe


def test_watchdog_counts_from_launch_when_no_file_exists_yet(k8s_env, clock):
    """No idle file yet means no mtime: idle time is measured since launch, so a
    harness that never even opens its log still gets killed."""
    pod = FakePod([ok(f"{RUNNING} 123\n")] * 100)  # stat printed nothing
    res = wire(k8s_env, pod).execute_detached("x", timeout=36000, idle_files=["/tmp/log"], idle_timeout=100)
    assert res["reason"] == "stall"


# -- helpers ---------------------------------------------------------------------------


def test_kill_helper_terms_then_kills_group(k8s_env, clock):
    pod = FakePod([])
    wire(k8s_env, pod)._detached_kill("/tmp/mimo-detached/abc")
    ((cmd, _),) = pod.calls
    assert 'kill -TERM -- -"$__pg"' in cmd
    assert 'kill -KILL -- -"$__pg"' in cmd
    assert "/tmp/mimo-detached/abc/pgid" in cmd


def test_tail_read_failure_does_not_mask_rc(k8s_env, clock):
    pod = FakePod([ok(f"{RC}0\n")])
    env = wire(k8s_env, pod)
    real = pod.execute

    def execute(command, cwd="", timeout=None):
        if command.startswith("tail -c"):
            raise TransportError("blip while reading output")
        return real(command, cwd=cwd, timeout=timeout)

    env.execute = execute
    res = env.execute_detached("x", timeout=60)
    assert res == {"output": "", "returncode": 0, "reason": "ok"}


def test_requires_started_pod(k8s_env, clock):
    k8s_env.pod_started = False
    with pytest.raises(AssertionError):
        k8s_env.execute_detached("x", timeout=1)


def test_other_backends_delegate_to_execute():
    from mimoagent.environments.local import LocalEnvironment

    env = LocalEnvironment()
    seen = {}

    def execute(command, cwd="", timeout=None):
        seen.update(command=command, cwd=cwd, timeout=timeout)
        return {"output": "", "returncode": 0}

    env.execute = execute
    env.execute_detached("echo", cwd="/w", timeout=7, idle_files=["x"], idle_timeout=3)
    assert seen == {"command": "echo", "cwd": "/w", "timeout": 7}


def test_probe_parser_kinds(k8s_env):
    env = k8s_env
    replies = iter(
        [
            ok(f"{RC}7\n"),
            ok(f"{RC}garbage\n"),
            ok(f"{RUNNING} 10 4\n"),
            ok(f"{RUNNING}\n"),
            ok(f"{GONE}\n"),
            ok("nothing useful\n"),
            {"output": "", "returncode": None, "reason": "client_timeout"},
        ]
    )
    env.execute = lambda command, cwd="", timeout=None: next(replies)
    assert env._detached_probe("p") == ("rc", 7)
    assert env._detached_probe("p") == ("rc", 1)
    assert env._detached_probe("p") == ("running", (10, 4))
    assert env._detached_probe("p") == ("running", None)
    assert env._detached_probe("p") == ("gone", None)
    assert env._detached_probe("p") == ("fail", "no_marker")
    assert env._detached_probe("p") == ("fail", "client_timeout")


# -- the pod-side shell, for real ---------------------------------------------------
#
# Everything above fakes the pod. These run the exact wrapper script, launch,
# probe and kill strings against the local bash, with ``execute`` standing in
# for the exec transport, so setsid / the /proc pgid read / GNU timeout
# --foreground / the atomic rc publish / kill -0 on the group / the stat-based
# watchdog are exercised end to end.

import os
import shutil
import subprocess
import time as real_time

needs_shell = pytest.mark.skipif(
    not (shutil.which("setsid") and shutil.which("timeout") and Path("/proc/self/stat").exists()),
    reason="needs linux, setsid and GNU timeout",
)


class LocalPod:
    """``execute``/``copy_to`` that hit the local box the way the pod does."""

    def __init__(self, root: Path):
        self.root = root
        self.calls = []

    def execute(self, command, cwd="", timeout=None):
        self.calls.append(command)
        cwd = cwd or str(self.root)
        try:
            r = subprocess.run(
                ["timeout", str(timeout), "/bin/bash", "-c", f"cd {cwd} && {command}"],
                text=True,
                capture_output=True,
                timeout=timeout + 5,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {"output": "", "returncode": None, "reason": "client_timeout"}
        if r.returncode == 124:
            return {"output": r.stdout + r.stderr, "returncode": 124, "reason": "pod_timeout"}
        return {"output": r.stdout + r.stderr, "returncode": r.returncode, "reason": "ok"}

    def copy_to(self, src, dst, **kwargs):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


@pytest.fixture
def local_env(k8s_env, tmp_path, monkeypatch):
    pod = LocalPod(tmp_path)
    k8s_env.execute = pod.execute
    k8s_env.copy_to = pod.copy_to
    k8s_env.config.cwd = str(tmp_path)
    # keep the run files under tmp_path instead of /tmp/mimo-detached
    monkeypatch.setattr(KubernetesEnvironment, "_DETACHED_DIR", str(tmp_path / "detached"))
    monkeypatch.setattr(KubernetesEnvironment, "_DETACHED_CLIENT_GRACE_S", 1.0)
    return k8s_env, pod


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pgid_of(run_root: Path) -> int:
    (pgid_file,) = run_root.glob("*/pgid")
    return int(pgid_file.read_text().strip())


def _wait_group_dead(pgid: int, seconds: float = 5.0) -> bool:
    end = real_time.monotonic() + seconds
    while real_time.monotonic() < end:
        if not _group_alive(pgid):
            return True
        real_time.sleep(0.1)
    return not _group_alive(pgid)


@needs_shell
def test_shell_roundtrip_rc_and_output(local_env, tmp_path):
    env, pod = local_env
    res = env.execute_detached("echo out; echo err >&2; exit 3", timeout=30, poll_interval=0.2)
    assert res["reason"] == "ok"
    assert res["returncode"] == 3
    assert res["output"] == "out\nerr\n"
    run_root = tmp_path / "detached"
    (rc_file,) = run_root.glob("*/rc")
    assert rc_file.read_text().strip() == "3"
    assert not list(run_root.glob("*/rc.tmp"))
    # the launch exec came back at once instead of waiting for the command
    launch = next(c for c in pod.calls if LAUNCHED in c)
    assert launch


@needs_shell
def test_shell_command_survives_the_launch_exec_ending(local_env, tmp_path):
    """The detached tree must not die with the launcher: the launch exec returns
    in milliseconds while the command keeps running for a couple of seconds."""
    env, pod = local_env
    t0 = real_time.monotonic()
    res = env.execute_detached("sleep 1.5; echo survived", timeout=30, poll_interval=0.2)
    assert res == {"output": "survived\n", "returncode": 0, "reason": "ok"}
    assert real_time.monotonic() - t0 >= 1.5


@needs_shell
def test_shell_pod_timeout_kills_the_whole_tree(local_env, tmp_path):
    """GNU timeout --foreground only signals bash; the grandchild sleep must be
    reaped by the follow-up group kill."""
    env, pod = local_env
    res = env.execute_detached("sleep 30 & sleep 30; echo never", timeout=1, poll_interval=0.2)
    assert res["returncode"] == 124
    assert res["reason"] == "pod_timeout"
    assert "never" not in res["output"]
    assert _wait_group_dead(_pgid_of(tmp_path / "detached"))


@needs_shell
def test_shell_client_deadline_kills_a_term_ignoring_tree(local_env, tmp_path):
    """The command ignores TERM so the pod-side timeout cannot end it; the
    client deadline (grace patched to 1s) has to kill the group."""
    env, pod = local_env
    res = env.execute_detached("trap '' TERM; sleep 60", timeout=1, poll_interval=0.2)
    assert res["reason"] == "client_timeout"
    assert res["returncode"] is None
    assert _wait_group_dead(_pgid_of(tmp_path / "detached"))


@needs_shell
def test_shell_stall_watchdog_uses_real_mtimes(local_env, tmp_path):
    env, pod = local_env
    log = tmp_path / "harness.log"
    t0 = real_time.monotonic()
    res = env.execute_detached(
        f"echo start > {log}; sleep 60",
        timeout=60,
        poll_interval=0.2,
        idle_files=[str(log)],
        idle_timeout=1,
    )
    assert res["reason"] == "stall"
    assert res["returncode"] is None
    assert real_time.monotonic() - t0 < 15
    assert _wait_group_dead(_pgid_of(tmp_path / "detached"))


@needs_shell
def test_shell_vanished_group_is_transport_error(local_env, tmp_path):
    """The whole group gets SIGKILLed from inside (container restart / OOM
    stand-in): no rc ever appears and the poller must notice."""
    env, pod = local_env
    res = env.execute_detached(
        "read -r _ s < /proc/self/stat; s=${s##*) }; set -- $s; kill -KILL -- -$3",
        timeout=60,
        poll_interval=0.2,
    )
    assert res["reason"] == "transport_error"
    assert "vanished" in res["output"]


@needs_shell
def test_shell_command_runs_in_the_recorded_fresh_group(local_env, tmp_path):
    """The command's own pgrp must be the pgid the wrapper published (that is
    what gets killed), and under setsid it is also the session id: a group
    nobody else is in, so the kill cannot reach beyond the run."""
    env, pod = local_env
    env.execute_detached(
        "read -r _ s < /proc/self/stat; s=${s##*) }; set -- $s; echo $3 $4 > ids",
        timeout=30,
        poll_interval=0.2,
    )
    pgrp, session = (int(x) for x in (tmp_path / "ids").read_text().split())
    pgid = _pgid_of(tmp_path / "detached")
    assert pgrp == pgid
    assert session == pgid
    assert not _group_alive(pgid)  # everything reaped after a normal exit
