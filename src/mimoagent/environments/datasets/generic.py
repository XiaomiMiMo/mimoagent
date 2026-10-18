"""Generic git dataset: clone an arbitrary repo, no built-in verifier."""

import shlex
import time

from mimoagent import Environment
from mimoagent.environments.datasets.base import DatasetEnvironment

_LEAK_PATHS = ("/tmp/patch.diff", "/tmp/test_patch.diff", "/tmp/test_files.json", "/tests", "/logs")


class GenericGitEnvironment(DatasetEnvironment):
    """Generic git-repo dataset ("generic").

    Identified by ``instance['dataset_type'] == 'generic'``. Setup clones
    ``instance['git_url']`` into the base env's cwd and resets to
    ``base_commit`` when present. Instances without ``git_url`` ship the
    repository baked into the docker image — the clone is skipped and only the
    reset runs (no ``git clean``: images may ship prebuilt untracked artifacts
    the build needs).

    There is no programmatic verifier (``HAS_VERIFIER = False``). Reward comes
    from the rubric judge when the instance carries ``rubric.rubrics`` and the
    yaml sets ``judge_agent`` (see ``rubric_judge.py``); otherwise run with
    ``--skip-reward``.
    """

    HAS_VERIFIER = False

    def __init__(self, base_env: Environment, instance: dict):
        super().__init__(base_env, instance)
        self._git_url = instance.get("git_url", "") or ""
        self._base_commit = instance.get("base_commit", "") or ""

    @property
    def repo_path(self) -> str:
        return getattr(self.env.config, "cwd", "") or self.REPO_PATH

    def _setup_dataset_specific(self) -> None:
        repo = self.repo_path
        if self._git_url:
            self._clone(repo)
        else:
            mkdir_res = self.execute(f"mkdir -p {shlex.quote(repo)}", cwd="/")
            if mkdir_res.get("returncode", 1) != 0:
                raise RuntimeError(
                    f"{self.instance_id}: failed to create workdir {repo}: {mkdir_res.get('output', '')[:2000]}"
                )

        self.git_add_safe_directory()

        if self._base_commit:
            self.git_reset_hard(self._base_commit)

        self._remove_leak_paths()

        if self._base_commit:
            self._prevent_git_hack()

    def _clone(self, repo: str) -> None:
        clone_cmd = f"git clone {shlex.quote(self._git_url)} {shlex.quote(repo)}"
        max_attempts = 3
        clone_res: dict = {}
        for attempt in range(1, max_attempts + 1):
            # cwd="/": the repo dir doesn't exist yet, and the base env would
            # otherwise cd into its (missing) default cwd before running the clone.
            clone_res = self.execute(clone_cmd, cwd="/", timeout=1800)
            if clone_res.get("returncode", 1) == 0:
                break
            self.logger.warning(
                f"{self.instance_id}: git clone attempt {attempt}/{max_attempts} failed "
                f"(rc={clone_res.get('returncode')}): {clone_res.get('output', '')[:500]}"
            )
            if attempt < max_attempts:
                # Drop the partial clone (git refuses a non-empty target dir).
                if repo and repo != "/":
                    self.execute(f"rm -rf {shlex.quote(repo)}", cwd="/", timeout=300)
                time.sleep(5 * attempt)
        else:
            raise RuntimeError(
                f"{self.instance_id}: git clone failed after {max_attempts} attempts "
                f"(rc={clone_res.get('returncode')}): {clone_res.get('output', '')[:2000]}"
            )

    def _remove_leak_paths(self) -> None:
        paths = [f"{self.repo_path}/.build_env", *_LEAK_PATHS]
        try:
            res = self.execute("rm -rf " + " ".join(shlex.quote(p) for p in paths))
            if res.get("returncode", 1) != 0:
                self.logger.warning(f"{self.instance_id}: failed to remove leak paths: {res.get('output', '')[:300]}")
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: leak-path removal failed (non-fatal): {repr(e)}")

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        raise NotImplementedError(
            "generic dataset has no verifier; supply rubric.rubrics + judge_agent for a rubric reward, "
            "or run with --skip-reward"
        )
