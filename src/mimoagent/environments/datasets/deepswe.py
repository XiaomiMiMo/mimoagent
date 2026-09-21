"""DeepSWE (https://deepswe.datacurve.ai/) benchmark environment.

Supports both dataset generations; a record's schema decides the verifier
protocol (v1.0: ``test_sh`` + ``test_patch``, verdict in ``reward.txt``;
v1.1: adds ``grader_py`` + ``config_json``, ``schema_version = "1.1"`` —
shared ``grader.py`` grades f2p/p2p whitelists into ``reward.json``).

Grading itself is uniform across generations:

1. **Capture** (agent pod): pier-style — ``git reset --soft <base>`` (the
   v1.1 task instruction tells the agent to commit its work to a feature
   branch, so HEAD may be ahead of base; the soft-reset folds committed AND
   uncommitted work into the index), ``git add -A`` (new/deleted/renamed
   files included), then ``git -c core.fileMode=false diff --cached --binary
   <base>`` written to ``/logs/artifacts/model.patch`` on the pod and pulled
   back byte-exact (``copy_out`` tar stream on k8s, base64 over exec
   elsewhere — piping the raw diff through exec output would utf-8-mangle
   binary hunks).

2. **Fresh eval env**: a second pod is started from the SAME image and
   base-env config (``type(self.env)(**self.env.get_template_vars())``).
   The agent pod is never graded in: its worktree / caches / test files may
   have been corrupted (accidentally or adversarially), while the fresh pod
   is guaranteed pristine image state — which is also exactly the
   separate-verifier-container contract the upstream v1.1 Harbor uses.

3. **Apply + run** (eval pod): verifier files go to ``/tests``, the captured
   patch to ``/logs/artifacts/model.patch``.

   * v1.1: ``grader.py prepare`` itself per-file-resets the patch's paths to
     base and applies it (image build steps may have deliberately modified
     tracked files in-tree, hence per-file, never repo-wide), then applies
     ``test.patch``; ``test.sh`` runs the suites and ``grader.py grade``
     writes ``/logs/verifier/reward.json``. A ``reward.txt`` containing
     ``-1`` is the crash sentinel (trap in test.sh) => infra error, not a
     model fail.
   * v1.0: the eval repo is reset hard to base first (mirroring agent-pod
     setup — v1.0 capture diffs against a hard-reset worktree, so base is
     the patch's exact preimage), ``model.patch`` is git-applied by us, and
     ``test.sh`` (which embeds the whole pipeline: pier re-capture,
     test.patch reset+apply, suite run) writes ``/logs/verifier/reward.txt``
     (1 = pass, 0 = fail).

Infra faults — capture failed, eval env failed to start, or our v1.0
re-apply of the captured patch onto its exact preimage failed — surface as
``REWARD_TESTBED_CORRUPTED`` so training never sees a false reward=0.
"""

import base64
import json
import os
import re
import tempfile

from mimoagent import Environment
from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED, DatasetEnvironment


def parse_v11_reward(reward_json_raw: str, reward_txt_raw: str) -> tuple[float, dict]:
    """Map the v1.1 verifier outputs to (reward, extra_info).

    ``reward_json_raw`` is the content of ``/logs/verifier/reward.json`` ("" if
    missing), ``reward_txt_raw`` the content of ``reward.txt`` ("" if missing).
    Pure so it's unit-testable.

    * valid reward.json          -> reward from its binary ``reward`` field,
                                    full json in extra["reward_json"]
    * reward.txt == "-1"         -> verifier crashed before grading (trap
                                    sentinel): infra error, not a model fail
    * anything else              -> reward 0
    """
    if reward_json_raw.strip():
        try:
            data = json.loads(reward_json_raw)
        except json.JSONDecodeError as e:
            return 0.0, {"reward_json_error": f"invalid reward.json: {e}"}
        reward = 1.0 if data.get("reward") == 1 else 0.0
        extra = {"reward_json": data, "resolved": reward == 1.0}
        if data.get("apply_failed"):
            extra["apply_failed"] = True
        return reward, extra
    if reward_txt_raw.strip() == "-1":
        return 0.0, {
            "error_category": REWARD_TESTBED_CORRUPTED,
            "error": "verifier_crash_sentinel",
            "resolved": False,
        }
    return 0.0, {"reward_json_error": "reward.json missing", "resolved": False}


def failing_nodes_from_ctrf(ctrf_raw: str, limit: int = 200) -> dict:
    """Failing whitelisted node ids from the grader's normalized ctrf.json.

    grader.py grade rewrites ctrf.json with test names ``[p2p] <node_id>`` /
    ``[f2p] <node_id>`` covering exactly the whitelists it graded, so no
    node-id derivation is repeated here. Returns {} on any parse problem —
    this is observability only and must never affect the reward.
    """
    try:
        tests = (json.loads(ctrf_raw).get("results") or {}).get("tests") or []
    except Exception:
        return {}
    failing: dict[str, list[str]] = {"p2p": [], "f2p": []}
    for tc in tests:
        if not isinstance(tc, dict) or tc.get("status") == "passed":
            continue
        name = str(tc.get("name") or "")
        for bucket in failing:
            prefix = f"[{bucket}] "
            if name.startswith(prefix):
                failing[bucket].append(name[len(prefix) :])
    out = {}
    for bucket, ids in failing.items():
        if ids:
            out[f"failing_{bucket}_node_ids"] = ids[:limit]
            out[f"failing_{bucket}_count"] = len(ids)
    return out


_TEST_COLLISION_RE = re.compile(r"(?:error:[^\n]*already exists[^\n]*|import file mismatch[^\n]*)")


def detect_test_collisions(verifier_output: str, limit: int = 3) -> list[str]:
    """Lines suggesting the model's leftover test files collided with the
    official ones (git-apply "already exists" / pytest "import file
    mismatch"). Observability only — flagged in extra, never acted upon.
    """
    return [m.group(0).strip()[:300] for m in _TEST_COLLISION_RE.finditer(verifier_output)][:limit]


class DeepSWEEnvironment(DatasetEnvironment):
    """DeepSWE dataset implementation (v1.0 and v1.1 — see module docstring)."""

    REPO_PATH = "/app"
    LOGS_DIR = "/logs/verifier"
    ARTIFACTS_DIR = "/logs/artifacts"
    TESTS_DIR = "/tests"
    MODEL_PATCH_PATH = "/logs/artifacts/model.patch"

    # Our pod exec wraps every command in `bash -lc`; /etc/profile resets PATH,
    # dropping the image's ENV PATH entries — upstream Pier runs without a
    # login shell and keeps them. The image ENV is unrecoverable at runtime
    # (PID 1 is itself a login bash, so even /proc/1/environ holds the
    # already-reset PATH), so re-add the one convention deep-swe images
    # actually rely on: a /opt/venv virtualenv (holds the verifier's pytest on
    # python tasks; go images' /root/go/bin is re-exported by test.sh itself).
    # Applied to the agent pod at setup AND to the fresh eval pod before grading.
    _VENV_PATH_FIX = (
        "if [ -d /opt/venv/bin ]; then "
        "echo 'export VIRTUAL_ENV=/opt/venv' >> /root/.bashrc; "
        "echo 'export PATH=\"/opt/venv/bin:$PATH\"' >> /root/.bashrc; "
        "fi"
    )

    # The pod may carry proxy env vars (http_proxy) for the
    # agent's censored egress. Repos' own suites assert on
    # proxy-from-environment (httpx: 54 p2p tests read $http_proxy and fail
    # spuriously), so the verifier run strips every proxy variable. Inlined
    # into the command string so it runs after the login-shell profile.
    _UNSET_PROXY = "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY ALL_PROXY all_proxy; "

    def __init__(self, base_env: Environment, instance: dict):
        super().__init__(base_env, instance)
        self._test_sh = instance.get("test_sh", "")
        self._test_patch = instance.get("test_patch", "")
        self._grader_py = instance.get("grader_py", "")
        self._config_json = instance.get("config_json", "")
        self._base_commit = instance.get("base_commit", "")
        if not self._test_sh:
            raise ValueError(f"{instance.get('instance_id')}: DeepSWE instance missing 'test_sh'")
        # base_commit is the anchor of the whole flow: capture diffs against
        # it and the eval pod restores/applies against it.
        if not self._base_commit:
            raise ValueError(f"{instance.get('instance_id')}: DeepSWE instance missing 'base_commit'")
        # v1.1 records carry the shared grader + its per-task config alongside
        # test.sh/test.patch; their presence selects the v1.1 flow. A record
        # declaring schema 1.1 without them would silently grade wrong -> raise.
        self._is_v11 = bool(self._grader_py and self._config_json)
        if str(instance.get("schema_version", "")).startswith("1.1") and not self._is_v11:
            raise ValueError(f"{instance.get('instance_id')}: schema_version 1.1 but missing 'grader_py'/'config_json'")
        if self._is_v11:
            # History is already stripped inside the v1.1 image (main -> base
            # commit, future refs gc'd at build time); strip would delete the
            # natural `main` the task promises and hide would break the
            # instructed branch+commit workflow. yaml `git_leak_prevention`
            # still overrides (applied after construction in create_from_registry).
            self.git_leak_prevention = "none"
        # Byte-exact model.patch pulled off the agent pod by _capture_model_diff;
        # None means capture has not run or failed (=> infra error at grading).
        self._model_patch_bytes: bytes | None = None

    def _setup_dataset_specific(self) -> None:
        try:
            self.git_add_safe_directory()
            self.execute(self._VENV_PATH_FIX)

            if not self._is_v11:
                # v1.1 images must NOT be reset: main already sits at
                # base_commit and image build steps may have deliberately
                # modified tracked files / created untracked artifacts that a
                # reset --hard + clean would destroy.
                self.git_reset_hard(self._base_commit, clean=True)

            self._prevent_git_hack()
        except Exception as e:
            self.logger.error(f"{self.instance_id}: Error setting up DeepSWE environment: {repr(e)}")
            raise

    # --- model patch capture (agent pod) --------------------------------------

    def _capture_model_diff(self) -> tuple[str, str]:
        """Pier-style capture of everything the agent changed, vs base_commit.

        Soft-reset HEAD onto base (the agent may have committed — v1.1
        instructs it to; this keeps index + worktree), stage everything, and
        diff the index against base into a pod-side file. The file is then
        pulled back byte-exact into ``self._model_patch_bytes`` (the grading
        source of truth); the returned text (utf-8 with replacement) only
        feeds extra["model_patch"] / recalc.
        """
        self._model_patch_bytes = None
        capture_script = (
            "set -e\n"
            f"git reset --soft {self._base_commit}\n"
            f"mkdir -p {self.ARTIFACTS_DIR}\n"
            "git add -A >/dev/null 2>&1 || true\n"
            f"git -c core.fileMode=false diff --cached --binary {self._base_commit} "
            f"> {self.MODEL_PATCH_PATH}\n"
            "git reset -q\n"
        )
        try:
            res = self.execute(capture_script, cwd=self.repo_path, timeout=300)
            rc = res.get("returncode")
            reason = res.get("reason")
            if rc not in (0, None) or reason not in ("ok", None, ""):
                msg = f"rc={rc} reason={reason}"
                self.logger.warning(f"{self.instance_id}: _capture_model_diff failed, {msg}")
                return "", msg
            data = self._pull_file_bytes(self.env, self.MODEL_PATCH_PATH)
            self._model_patch_bytes = data
            return data.decode("utf-8", errors="replace"), ""
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: _capture_model_diff error: {e}")
            return "", str(e)

    @staticmethod
    def _pull_file_bytes(env: Environment, remote_path: str) -> bytes:
        """Read ``remote_path`` from ``env`` byte-exact.

        ``copy_out`` (tar-stream, k8s) when the env has it; otherwise base64
        through the exec channel — ASCII survives the utf-8 text decoding that
        would mangle raw binary-diff bytes. Raises on failure.
        """
        if hasattr(env, "copy_out"):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                local_path = f.name
            try:
                env.copy_out(remote_path, local_path)
                with open(local_path, "rb") as f:
                    return f.read()
            finally:
                os.unlink(local_path)
        res = env.execute(f"base64 {remote_path}", timeout=300)
        if res.get("returncode", 1) != 0:
            raise RuntimeError(f"base64 {remote_path} failed: {res.get('output', '')[:500]}")
        return base64.b64decode("".join(str(res.get("output", "")).split()))

    # --- fresh eval env --------------------------------------------------------

    def _make_eval_env(self) -> Environment:
        """A sibling environment: same class + same config as the agent env.

        ``get_template_vars`` returns the base env's config as a dict, so the
        clone lands on the same image (likely cached on the same nodes),
        kubeconfig, namespace, node selector, and resource limits.
        """
        kwargs = dict(self.env.get_template_vars())
        labels = kwargs.get("labels")
        if isinstance(labels, dict):
            kwargs["labels"] = {**labels, "mimoagent-role": "deepswe-eval"}
        eval_env = type(self.env)(**kwargs)
        if hasattr(eval_env, "instance_id"):
            eval_env.instance_id = self.instance_id
        return eval_env

    def _write_test_files(self, env: Environment) -> None:
        """Materialise the verifier files under TESTS_DIR inside ``env``.

        copy_to (tar-stream) is used since test.patch may contain non-utf8
        bytes (binary diffs) — heredocs would mangle them.
        """
        self.copy_text_to(self._test_sh, f"{self.TESTS_DIR}/test.sh", executable=True, env=env)
        self.copy_text_to(self._test_patch, f"{self.TESTS_DIR}/test.patch", env=env)
        if self._is_v11:
            self.copy_text_to(self._grader_py, f"{self.TESTS_DIR}/grader.py", env=env)
            self.copy_text_to(self._config_json, f"{self.TESTS_DIR}/config.json", env=env)

    def _push_model_patch(self, eval_env: Environment) -> None:
        """Write the captured patch bytes to MODEL_PATCH_PATH inside the eval env.

        Always pushed, even when empty: the v1.1 grader grades the base state
        (reward 0 by construction) rather than tripping the crash sentinel.
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(self._model_patch_bytes)
            local_path = f.name
        try:
            eval_env.copy_to(local_path, self.MODEL_PATCH_PATH)
        finally:
            os.unlink(local_path)

    # --- reward ------------------------------------------------------------------

    @staticmethod
    def _infra_error(error: str, output: str) -> tuple[float, str, dict]:
        return (
            0.0,
            output,
            {
                "error_category": REWARD_TESTBED_CORRUPTED,
                "error": error,
                "resolved": False,
            },
        )

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        if self._model_patch_bytes is None:
            # Capture failed (details in extra["model_patch_error"], set by the
            # base class): there is nothing trustworthy to grade.
            return self._infra_error("model_patch_capture_failed", "model.patch capture failed; nothing to grade")

        exec_timeout = (
            timeout
            if timeout is not None
            # Records ship verifier_timeout_sec=1800 which the koota-scale
            # suites exceed (observed rc=124 at exactly 30min); floor at the
            # code default 3200 so a slow-but-honest verify isn't voided.
            else max(int(self.instance.get("verifier_timeout_sec") or 0), 3200)
        )

        eval_env = None
        try:
            try:
                eval_env = self._make_eval_env()
                eval_env.start()
            except Exception as e:
                self.logger.warning(f"{self.instance_id}: eval env failed to start: {repr(e)}")
                return self._infra_error("eval_env_start_failed", f"eval env failed to start: {e}")
            self.logger.info(
                f"{self.instance_id}: grading in fresh eval env (pod={getattr(eval_env, 'pod_name', None)})"
            )
            try:
                reward, output, extra = self._grade_in_eval_env(eval_env, exec_timeout)
            except Exception as e:
                # Everything inside the eval env is harness territory — the
                # model only produced the patch. A copy/exec fault here must
                # not read as a model fail.
                self.logger.warning(f"{self.instance_id}: eval env grading errored: {repr(e)}")
                return self._infra_error("eval_env_error", f"eval env grading errored: {e}")
            pod_name = getattr(eval_env, "pod_name", None)
            if pod_name:
                extra["eval_pod_name"] = pod_name
            return reward, output, extra
        finally:
            if eval_env is not None:
                try:
                    eval_env.cleanup()
                except Exception as e:
                    self.logger.warning(f"{self.instance_id}: eval env cleanup failed: {repr(e)}")

    def _grade_in_eval_env(self, eval_env: Environment, exec_timeout: int | float) -> tuple[float, str, dict]:
        eval_env.execute(f"git config --global --add safe.directory {self.repo_path}")
        eval_env.execute(self._VENV_PATH_FIX)
        eval_env.execute(f"mkdir -p {self.LOGS_DIR} {self.ARTIFACTS_DIR} {self.TESTS_DIR}")

        if not self._is_v11:
            # Mirror agent-pod setup so base_commit is the exact preimage of
            # the captured patch. (v1.1 repos must stay at image state — the
            # grader's per-file resets handle preimage alignment there.)
            reset_res = eval_env.execute(
                f"git reset --hard {self._base_commit} && git clean -fd",
                cwd=self.repo_path,
                timeout=600,
            )
            if reset_res.get("returncode", 1) != 0:
                self.logger.warning(
                    f"{self.instance_id}: eval env git reset to base failed "
                    f"(rc={reset_res.get('returncode')}): {reset_res.get('output', '')[:500]}"
                )

        self._write_test_files(eval_env)
        self._push_model_patch(eval_env)

        if not self._is_v11 and self._model_patch_bytes.strip():
            # v1.1's grader applies the patch itself (grader.py prepare); for
            # v1.0 we apply it. The eval worktree is the patch's exact
            # preimage, so a failure here is a capture/harness fault, never
            # the model's.
            apply_res = eval_env.execute(
                f"git apply --whitespace=nowarn {self.MODEL_PATCH_PATH}",
                cwd=self.repo_path,
                timeout=300,
            )
            if apply_res.get("returncode", 1) != 0:
                self.logger.warning(
                    f"{self.instance_id}: model.patch failed to apply in eval env "
                    f"(rc={apply_res.get('returncode')}): {apply_res.get('output', '')[:2000]}"
                )
                return self._infra_error("model_patch_apply_failed", apply_res.get("output", ""))

        run_res = eval_env.execute(
            self._UNSET_PROXY + f"bash {self.TESTS_DIR}/test.sh",
            cwd=self.repo_path,
            timeout=exec_timeout,
        )
        output = run_res.get("output", "")
        verifier_rc = run_res.get("returncode", 1)
        collisions = detect_test_collisions(output)

        if self._is_v11:
            reward_json_raw = eval_env.execute(f"cat {self.LOGS_DIR}/reward.json 2>/dev/null").get("output", "")
            reward_txt_raw = eval_env.execute(f"cat {self.LOGS_DIR}/reward.txt 2>/dev/null").get("output", "")
            reward, extra_info = parse_v11_reward(reward_json_raw, reward_txt_raw)
            extra_info["verifier_returncode"] = verifier_rc
            if collisions:
                extra_info["suspect_test_collision"] = collisions
            try:
                ctrf_raw = eval_env.execute(f"cat {self.LOGS_DIR}/ctrf.json 2>/dev/null").get("output", "")
                extra_info.update(failing_nodes_from_ctrf(ctrf_raw))
            except Exception as e:
                self.logger.warning(f"{self.instance_id}: ctrf.json pull failed: {repr(e)}")
            return reward, output, extra_info

        reward_res = eval_env.execute(f"cat {self.LOGS_DIR}/reward.txt 2>/dev/null || echo MISSING")
        reward_raw = reward_res.get("output", "").strip()
        # Verifier writes 1/0; anything else (incl. MISSING) is a fail.
        reward = 1.0 if reward_raw == "1" else 0.0

        extra_info = {
            "verifier_returncode": verifier_rc,
            "verifier_reward_file": reward_raw,
            "resolved": reward == 1.0,
        }
        if collisions:
            extra_info["suspect_test_collision"] = collisions

        return reward, output, extra_info
