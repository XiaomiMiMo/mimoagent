"""Detached execution shared by remote backends.

Some exec transports cannot hold a single connection for as long as a
blackbox harness session runs (hours). ``DetachedExecMixin.execute_detached``
never keeps a connection open for long: it launches the command in its own
session with stdio on remote files, then polls an rc marker with short
``execute`` calls. Everything the process tree writes lands in remote files,
so the poller only ever transfers a few bytes.

The host class supplies ``execute`` / ``copy_to`` (the transport), ``logger``,
``config.cwd`` / ``config.timeout``, ``_return_transport_error`` and
``_MAX_OUTPUT_BYTES``; it may override ``_detached_assert_started`` and
``_detached_target`` for its own readiness check and log tag.
"""

import os
import shlex
import tempfile
import time
import uuid
from typing import Any


class DetachedExecMixin:
    _DETACHED_DIR = "/tmp/mimo-detached"
    # Probes keep failing this long (apiserver unreachable, pod gone) before the
    # run is written off as a transport error; single blips are ignored.
    _DETACHED_TRANSPORT_GRACE_S: int = 600
    _DETACHED_PROBE_TIMEOUT_S: int = 60
    # Consecutive probes that find neither an rc file nor a live process group
    # before the run is declared vanished. Two, because the very first probe can
    # race the wrapper writing its pgid file.
    _DETACHED_GONE_STREAK: int = 2
    # How far the client deadline sits behind the remote-side ``timeout``, so the
    # remote gets to report rc=124 itself before the client kill steps in.
    _DETACHED_CLIENT_GRACE_S: float = 30.0
    _DETACHED_MARK_RC = "__MIMO_DETACHED_RC="
    _DETACHED_MARK_RUNNING = "__MIMO_DETACHED_RUNNING"
    _DETACHED_MARK_GONE = "__MIMO_DETACHED_GONE"
    # Log tag of the detached path; backends name it after their unit of
    # execution (``pod_exec_detached``, ``sandbox_exec_detached``).
    _DETACHED_LOG_TAG = "exec_detached"

    def _detached_assert_started(self) -> None:
        """Fail fast when the backend has no running target."""

    def _detached_target(self) -> str:
        """``key=value`` naming the remote target in log lines."""
        return ""

    def _stage_text(self, text: str, remote_path: str) -> None:
        """Write ``text`` to ``remote_path`` through copy_to (tar stream, so no
        argv length or quoting limits)."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False) as f:
            f.write(text)
            local_path = f.name
        try:
            self.copy_to(local_path, remote_path)
        finally:
            os.unlink(local_path)

    @classmethod
    def _detached_wrapper_script(cls, run_dir: str, full_command: str, timeout: int) -> str:
        """Remote-side wrapper: record the process group, run the command under
        ``timeout`` (remote-side backstop for a dead client), publish the rc
        atomically so the poller never reads a torn file."""
        d = shlex.quote(run_dir)
        return (
            "#!/bin/bash\n"
            "# Written by mimoagent execute_detached.\n"
            # pgid from /proc: correct whether or not setsid was available at
            # launch. Every process the command spawns inherits it, which is
            # what the poller kills on timeout/stall.
            "read -r _ __stat < /proc/self/stat\n"
            "__stat=${__stat##*) }\n"
            "set -- $__stat\n"
            f"printf '%s\\n' \"$3\" > {d}/pgid\n"
            # GNU timeout moves the command into a fresh process group by
            # default, which would put it out of reach of the group kill;
            # --foreground keeps the whole tree in ours. BusyBox timeout has no
            # such option and never changes groups.
            "if timeout --foreground 1 true >/dev/null 2>&1; then __to='timeout --foreground'; else __to='timeout'; fi\n"
            f"$__to {int(timeout)} /bin/bash -lc {shlex.quote(full_command)}\n"
            "__rc=$?\n"
            f"printf '%s\\n' \"$__rc\" > {d}/rc.tmp && mv -f {d}/rc.tmp {d}/rc\n"
        )

    def _detached_kill(self, run_dir: str) -> None:
        """Best effort: TERM then KILL the detached process group."""
        d = shlex.quote(run_dir)
        cmd = (
            f"__pg=$(cat {d}/pgid 2>/dev/null); "
            f'if [ -n "$__pg" ]; then kill -TERM -- -"$__pg" 2>/dev/null; sleep 2; '
            f'kill -KILL -- -"$__pg" 2>/dev/null; fi; true'
        )
        try:
            self.execute(cmd, timeout=30)
        except Exception as e:
            self.logger.warning(f"[{self._DETACHED_LOG_TAG}] kill of {run_dir} failed: {e}")

    def _detached_probe(self, probe: str) -> tuple[str, Any]:
        """One poll of a detached run.

        Returns ``("rc", int)`` once the wrapper published an exit code,
        ``("running", (remote_now, newest_mtime) | None)`` while the process
        group is alive (both on the remote clock; None when the watchdog is off
        or none of its files exists yet), ``("gone", None)`` when neither an rc
        file nor a live group was found, and ``("fail", reason)`` when the probe
        itself did not complete — a transport blip, not a verdict on the run.
        """
        try:
            chk = self.execute(probe, timeout=self._DETACHED_PROBE_TIMEOUT_S)
        except Exception as e:  # TransportError included
            self.logger.warning(f"[{self._DETACHED_LOG_TAG}] probe raised: {e}")
            return "fail", "exception"
        if chk.get("reason") != "ok":
            return "fail", chk.get("reason")
        output = chk.get("output") or ""
        marks = (self._DETACHED_MARK_RC, self._DETACHED_MARK_RUNNING, self._DETACHED_MARK_GONE)
        marker = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip().startswith(marks)), "")
        if marker.startswith(self._DETACHED_MARK_RC):
            raw_rc = marker[len(self._DETACHED_MARK_RC) :].strip()
            try:
                return "rc", int(raw_rc)
            except ValueError:
                self.logger.warning(f"[{self._DETACHED_LOG_TAG}] unparseable rc {raw_rc!r}; treating as 1")
                return "rc", 1
        if marker.startswith(self._DETACHED_MARK_GONE):
            return "gone", None
        if marker.startswith(self._DETACHED_MARK_RUNNING):
            fields = marker.split()
            if len(fields) >= 3:
                try:
                    return "running", (int(fields[1]), int(fields[2]))
                except ValueError:
                    pass
            return "running", None
        self.logger.warning(f"[{self._DETACHED_LOG_TAG}] probe output without marker: {output[-200:]!r}")
        return "fail", "no_marker"

    def execute_detached(
        self,
        command: str,
        cwd: str = "",
        timeout: int = None,
        *,
        poll_interval: float = 20.0,
        idle_files: list[str] | tuple[str, ...] = (),
        idle_timeout: int = 0,
        output_tail_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Run ``command`` detached from the exec transport and poll it to completion.

        Same contract as ``execute`` — ``{"output", "returncode", "reason"}``
        with reason ∈ {"ok", "pod_timeout", "client_timeout", "stall",
        "transport_error"} — but no single connection lives longer than one
        probe, so the run may outlast the transport's connection lifetime cap.
        ``output`` is the tail of the command's combined stdout/stderr.

        ``idle_files`` + ``idle_timeout`` enable a stall watchdog: when none of
        the files changed (mtime, remote clock) for ``idle_timeout`` seconds the
        process group is killed and reason is ``"stall"`` (returncode None).
        Off by default; only enable for commands that write steadily.
        """
        exec_t0 = time.monotonic()
        if timeout is None:
            timeout = self.config.timeout
        cwd = cwd or self.config.cwd
        self._detached_assert_started()
        if output_tail_bytes is None:
            output_tail_bytes = self._MAX_OUTPUT_BYTES

        cmd_short = command[:200] + ("..." if len(command) > 200 else "")
        run_dir = f"{self._DETACHED_DIR}/{uuid.uuid4().hex[:12]}"
        d = shlex.quote(run_dir)
        tag = self._DETACHED_LOG_TAG
        target = self._detached_target()

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            duration = time.monotonic() - exec_t0
            log_fn = self.logger.debug if result["reason"] == "ok" else self.logger.info
            log_fn(
                f"[{tag}] {target} run_dir={run_dir} duration={duration:.2f}s "
                f"rc={result['returncode']} reason={result['reason']} cmd={cmd_short!r}"
            )
            return result

        def transport_failure(message: str) -> dict[str, Any]:
            self.logger.error(f"[{tag}] {message}")
            return finish(self._return_transport_error(message))

        def killed(reason: str, note: str) -> dict[str, Any]:
            self._detached_kill(run_dir)
            output = self._detached_tail(run_dir, output_tail_bytes)
            return finish({"output": f"{output}\n{note}", "returncode": None, "reason": reason})

        full_command = f"cd {shlex.quote(cwd)} && {command}"
        self._stage_text(self._detached_wrapper_script(run_dir, full_command, timeout), f"{run_dir}/run.sh")

        # ``;`` before the backgrounded command matters: with ``&&`` the whole
        # list would be backgrounded in a subshell that still holds the exec
        # session's stdout, and the launch exec would never return. setsid
        # (when present) gives the wrapper its own session so nothing about the
        # exec's process group or its GNU-timeout group kill can reach it.
        launch = (
            f"__ss=$(command -v setsid); "
            f"$__ss /bin/bash {d}/run.sh </dev/null >{d}/out 2>&1 & "
            f"echo __MIMO_DETACHED_LAUNCHED"
        )
        res = self.execute(launch, timeout=60)
        if res.get("reason") != "ok":
            # The launch exec itself did not complete: report that as-is, the
            # command may or may not have started.
            res["output"] = f"{res.get('output') or ''}\n(detached launch did not complete)"
            return finish(res)
        if "__MIMO_DETACHED_LAUNCHED" not in (res.get("output") or ""):
            # The launcher shell failed before backgrounding anything (bad cwd,
            # unwritable /tmp, ...): surface it the way execute would.
            rc = res.get("returncode")
            return finish(
                {
                    "output": res.get("output") or "",
                    "returncode": rc if rc not in (None, 0) else 1,
                    "reason": "ok",
                }
            )
        launched_at = time.monotonic()

        watchdog = bool(idle_files) and idle_timeout > 0
        if watchdog:
            files = " ".join(shlex.quote(f) for f in idle_files)
            # date and stat run in the same remote shell, so the clocks agree.
            running = (
                f"echo {self._DETACHED_MARK_RUNNING} $(date +%s) $(stat -c %Y {files} 2>/dev/null | sort -n | tail -1)"
            )
        else:
            running = f"echo {self._DETACHED_MARK_RUNNING}"
        probe = (
            f"if [ -f {d}/rc ]; then echo {self._DETACHED_MARK_RC}$(cat {d}/rc); "
            f'elif kill -0 -- -"$(cat {d}/pgid 2>/dev/null)" 2>/dev/null; then {running}; '
            # the process finished between the two checks
            f"elif [ -f {d}/rc ]; then echo {self._DETACHED_MARK_RC}$(cat {d}/rc); "
            f"else echo {self._DETACHED_MARK_GONE}; fi"
        )

        deadline = exec_t0 + timeout + poll_interval + self._DETACHED_CLIENT_GRACE_S
        fail_since: float | None = None
        gone_streak = 0
        # Short first wait so an instant crash is caught by a cheap probe.
        time.sleep(min(3.0, poll_interval))
        while True:
            kind, payload = self._detached_probe(probe)
            if kind == "rc":
                rc = payload
                break
            if kind == "fail":
                now = time.monotonic()
                fail_since = now if fail_since is None else fail_since
                if now - fail_since >= self._DETACHED_TRANSPORT_GRACE_S:
                    return transport_failure(
                        f"Detached run unreachable for {now - fail_since:.0f}s "
                        f"(last probe: {payload}); run_dir={run_dir}"
                    )
            else:
                fail_since = None
            if kind == "gone":
                gone_streak += 1
                if gone_streak >= self._DETACHED_GONE_STREAK:
                    self._detached_kill(run_dir)
                    return transport_failure(
                        f"Detached process vanished without an exit status "
                        f"({time.monotonic() - launched_at:.0f}s after launch); run_dir={run_dir}"
                    )
            elif kind == "running":
                gone_streak = 0
                if watchdog:
                    if payload is not None:
                        remote_now, newest_mtime = payload
                        idle = remote_now - newest_mtime
                    else:
                        # none of the files exists yet: idle since launch
                        idle = int(time.monotonic() - launched_at)
                    if idle > idle_timeout:
                        self.logger.error(
                            f"[{tag}] no activity on {list(idle_files)} for {idle}s "
                            f"(> {idle_timeout}s); killing process group"
                        )
                        return killed("stall", f"Command stalled: no output-file activity for {idle}s")
            if time.monotonic() >= deadline:
                self.logger.error(f"[{tag}] client deadline passed after {timeout}s; killing process group")
                return killed("client_timeout", f"Command timed out (client) after {timeout}s")
            time.sleep(poll_interval)

        output = self._detached_tail(run_dir, output_tail_bytes)
        if rc == 124:
            # ``timeout --foreground`` (and BusyBox timeout) signal only the
            # direct child; reap whatever the command left behind.
            self._detached_kill(run_dir)
            return finish(
                {
                    "output": output + f"\nCommand timed out (pod) after {timeout}s",
                    "returncode": 124,
                    "reason": "pod_timeout",
                }
            )
        return finish({"output": output, "returncode": rc, "reason": "ok"})

    def _detached_tail(self, run_dir: str, max_bytes: int) -> str:
        """Read back the tail of the detached command's combined output.
        Best effort: the rc file is authoritative, a failed read is not."""
        try:
            res = self.execute(f"tail -c {int(max_bytes)} {shlex.quote(run_dir)}/out 2>/dev/null || true", timeout=120)
        except Exception as e:
            self.logger.warning(f"[{self._DETACHED_LOG_TAG}] could not read output of {run_dir}: {e}")
            return ""
        if res.get("reason") != "ok":
            self.logger.warning(
                f"[{self._DETACHED_LOG_TAG}] output read of {run_dir} failed: reason={res.get('reason')}"
            )
            return ""
        return res.get("output") or ""
