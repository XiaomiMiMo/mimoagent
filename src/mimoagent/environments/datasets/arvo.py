"""ARVO crash-reproduction dataset: deterministic sanitizer-verdict reward.

Each instance is one already-fixed defect from the public ARVO corpus of
reproducible OSS-Fuzz findings.  The image ships a sanitizer-instrumented
target binary under ``/root/binary``, source code under ``/home/agent/src``,
and a ``submit.sh`` wrapper.  The agent writes a PoC file and submits it;
reward is 1.0 iff the crash lands in the exact function and bug type
described in the task statement (deterministic string match, no LLM judge).

Setup injects an updated grading server (``server_arvo.py``) and the
expected-function config into the pod, then starts the server on port 8666.
The server runs as root and writes ``/root/last_result.json``; the agent
runs as a restricted user and cannot overwrite that file.
"""

import json
import re
from pathlib import Path

from mimoagent.environments.datasets.base import DatasetEnvironment


_SERVER_PY_PATH = str(Path(__file__).resolve().parent / "resources" / "server_arvo.py")


def _parse_description(description: str) -> tuple[str, str, str, str]:
    """Extract (function, file, sanitizer, error_type) from a task description.

    Every shipped ARVO row matches these patterns; a parse failure means the
    row is malformed and would silently score zero (the grading server returns
    match=false when expected_func is empty), so we fail loudly here.
    """
    func_m = re.search(r"in function `([^`]+)`", description)
    file_m = re.search(r"in file `([^`]+)`", description)
    san_m = re.match(r"(\S+):\s+(\S+)", description)
    if not func_m:
        raise ValueError(f"cannot parse function from description: {description!r}")
    return (
        func_m.group(1),
        file_m.group(1) if file_m else "",
        san_m.group(1) if san_m else "",
        san_m.group(2) if san_m else "",
    )


class ARVOEnvironment(DatasetEnvironment):
    """Environment for ARVO crash-reproduction RL."""

    REPO_PATH = "/home/agent"
    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    _ANTI_HACK_CLEANUP_DEFAULT = False

    def __init__(self, base_env, instance: dict):
        super().__init__(base_env, instance)
        self.git_leak_prevention = "none"

    @property
    def repo_path(self) -> str:
        return self.instance.get("cwd") or self.REPO_PATH

    # ---- setup ----

    def _setup_dataset_specific(self) -> None:
        desc = self.instance.get("description", self.instance.get("problem_statement", ""))
        func, file_, sanitizer, error_type = _parse_description(desc)

        max_submits = getattr(self.env.config, "max_submits", 0)
        config_json = json.dumps({
            "function": func,
            "file": file_,
            "sanitizer": sanitizer,
            "error_type": error_type,
            "max_submits": max_submits,
        })
        self.env.execute(f"cat > /root/expected_func.json << 'EOFCFG'\n{config_json}\nEOFCFG")

        try:
            server_content = Path(_SERVER_PY_PATH).read_text()
            self.copy_text_to(server_content, "/root/server.py")
        except Exception as exc:
            raise RuntimeError(f"failed to upload server_arvo.py: {exc}") from exc

        self.env.execute("(python3 /root/server.py </dev/null &>/dev/null &)")

        # Wait for the server to be ready (bounded poll, not a blind sleep).
        for _ in range(10):
            probe = self.env.execute("curl -sf http://127.0.0.1:8666/ 2>/dev/null; echo $?")
            out = (probe.get("output") or "").strip()
            if out.endswith("0") or "404" in out:
                break
            self.env.execute("sleep 0.5")

        no_binary = getattr(self.env.config, "no_binary", False)
        if no_binary:
            self.env.execute(
                "find /home/agent/binary -type f ! -name 'run.sh' -delete "
                "&& find /home/agent/binary -mindepth 1 -type d -empty -delete"
            )

    # ---- reward ----

    def _capture_model_diff(self) -> tuple[str, str]:
        return "", ""

    def _do_calculate_reward(self, timeout=None, model_patch=""):
        result = self.env.execute("cat /root/last_result.json 2>/dev/null")
        output = (result.get("output") or "").strip()

        poc_result = self.env.execute("base64 /root/last_poc 2>/dev/null")
        poc_b64 = (poc_result.get("output") or "").strip()

        if not output:
            return 0.0, "No submission found", {"last_poc_b64": poc_b64}

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return 0.0, f"Invalid result.json: {output}", {"last_poc_b64": poc_b64}

        crash = data.get("crash", False)
        match = data.get("match", False)
        expected_func = data.get("expected_func", "")
        actual_func = data.get("actual_func", "")

        extra = {"last_result": data, "last_poc_b64": poc_b64}

        if not crash:
            return 0.0, f"No crash (exit_code={data.get('exit_code', 0)})", extra
        if match:
            return 1.0, f"Crash in correct function: {actual_func}", extra
        return 0.0, f"Crash in wrong function: got `{actual_func}`, expected `{expected_func}`", extra
