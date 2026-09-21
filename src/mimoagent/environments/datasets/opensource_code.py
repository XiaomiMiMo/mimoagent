"""Open-source code dataset: ``test_patch`` + ``test_command``, exit-code verdict.

One environment for every code-RL task, built on two data fields and three
reward steps. It replaces a family of per-source environments whose only real
differences were *where the verifier lived* and *how the verdict was derived*.

Row contract (all eight fields are required)::

    dataset_type          "opensource-code"
    docker_image          fully-qualified image, registry included
    cwd                   absolute path of the repository inside the container
    instance_id           identity, used in logs and dumps
    problem_statement     the task text handed to the agent
    test_patch            unified diff: the hidden tests AND the verifier script
    test_command          one line, absolute path, e.g. bash /testbed/mimo_test_command.sh
    verifier_timeout_sec  wall clock for the verifier

``cwd`` is the single source of truth for the repository location: this class
overrides :attr:`repo_path` to read it, so the data field and the path used by
``git apply`` cannot drift apart. ``test_command`` is expected to reference the
same absolute path, which keeps it independent of whatever ``cwd`` the caller
happens to pass to ``execute``.

Why the verifier ships inside ``test_patch`` rather than in the image: the
agent and the grader share one pod and one filesystem, so anything present
during the rollout is readable with ``cat``. The patch is applied only at
reward time, so the hidden tests and the script that runs them never exist
while the agent works. Shipping the script in the patch also means it is
covered by the same anti-tamper reset as the tests (step 1 below) instead of
needing a mechanism of its own.

Reward contract (order matters)::

    1. reset every path the test patch touches back to the captured base ref
    2. git apply test_patch
    3. run test_command; reward = 1.0 iff it exits 0

Step 1 is not a hardening measure, it is what makes step 2 reliable: an agent
that created a file at a path the patch ADDS would otherwise make ``git apply``
fail with "already exists in working directory", which is a reward of 0 that
the model did not earn. Failures in steps 1 and 2 are reported with
``error_category`` so the trainer can separate a broken testbed from a wrong
answer.

Expectations on the image (none of these are checked beyond the cheap history
assertion in setup, because they are build-time properties):

* the working tree is already in the task state, so no reset is needed;
* git history is truncated at the task's base commit, with branches, remotes,
  newer tags, reflog and unreachable objects removed — otherwise the agent can
  read the reference fix out of ``git log``;
* no verifier, test file or task specification is left anywhere on disk;
* language toolchains are on ``PATH`` with their environment exported.
"""

from __future__ import annotations

import time

from mimoagent import Environment
from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED, DatasetEnvironment

_REQUIRED_FIELDS = (
    "cwd",
    "docker_image",
    "instance_id",
    "problem_statement",
    "test_command",
    "test_patch",
    "verifier_timeout_sec",
)


class OpenSourceCodeEnvironment(DatasetEnvironment):
    """Code-RL dataset graded by applying a test patch and running one command."""

    REPO_PATH = "/testbed"
    """Unused: :attr:`repo_path` reads ``instance["cwd"]``. Kept because the
    base class declares it and a stale value here would be misleading."""

    HAS_VERIFIER = True

    _ANTI_HACK_CLEANUP_DEFAULT = False
    """The build-env purges exist for images that ship build residue. Images for
    this dataset are expected to be clean already; a run against images that are
    not can switch this on with the yaml ``anti_hack_cleanup`` key."""

    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    """History truncation is a build-time property of the image, so there is
    nothing to strip at rollout time. Setup still asserts the property holds --
    see :meth:`_assert_history_truncated`. A run against untruncated images can
    fall back to ``git_leak_prevention: strip`` in the yaml, which does the
    stripping in-pod at the cost of one garbage collection per rollout."""

    _TEST_PATCH_REMOTE = "/tmp/_opensource_code_test.patch"

    def __init__(self, base_env: Environment, instance: dict):
        super().__init__(base_env, instance)
        missing = [f for f in _REQUIRED_FIELDS if not instance.get(f)]
        if missing:
            raise ValueError(
                f"{instance.get('instance_id') or '<no instance_id>'}: opensource-code instance "
                f"missing required field(s) {missing}"
            )
        self._test_patch: str = instance["test_patch"]
        self._test_command: str = instance["test_command"]
        self._verifier_timeout = int(instance["verifier_timeout_sec"])
        # The commit that represents the task's starting state. Captured in
        # setup rather than read from the row: an immutable sha taken before the
        # agent runs cannot disagree with the image, and stays valid even if the
        # agent moves HEAD with its own commits.
        self._base_ref: str = ""

    @property
    def repo_path(self) -> str:
        return self.instance["cwd"]

    # --- setup ----------------------------------------------------------------

    def _setup_dataset_specific(self) -> None:
        self.git_add_safe_directory()
        self._ensure_work_tree()
        self._base_ref = self._capture_base_ref()
        self._assert_history_truncated()

    def _ensure_work_tree(self) -> None:
        """Make sure ``repo_path`` is a git work tree.

        ``git apply`` and the step-1 reset both need one. Images built from a
        real repository already have it; images that ship a bare source tree
        (no history at all) get a baseline commit here, which then serves as
        their base ref.
        """
        if self.execute("git rev-parse --git-dir", cwd=self.repo_path).get("returncode", 1) == 0:
            return
        res = self.execute(
            "git init -q && git add -A && git commit -q -m baseline --allow-empty",
            cwd=self.repo_path,
        )
        if res.get("returncode", 1) != 0:
            raise RuntimeError(
                f"{self.instance_id}: could not initialise a git work tree at {self.repo_path}: "
                f"{str(res.get('output'))[:500]}"
            )

    def _capture_base_ref(self) -> str:
        res = self.execute("git rev-parse HEAD", cwd=self.repo_path)
        ref = str(res.get("output") or "").strip().split()[-1] if res.get("output") else ""
        if res.get("returncode", 1) != 0 or len(ref) != 40:
            raise RuntimeError(
                f"{self.instance_id}: could not resolve HEAD at {self.repo_path} "
                f"(rc={res.get('returncode')}): {str(res.get('output'))[:500]}"
            )
        return ref

    def _assert_history_truncated(self) -> None:
        """Fail setup if any commit outside the base ref's ancestry is reachable.

        The reference fix usually lives in a commit made after the base: on a
        branch tip, a remote-tracking ref, or a release tag. If the image was
        built without truncating the history, the agent can read the answer out
        of ``git log`` and the resulting scores are meaningless -- a failure
        worth aborting on rather than discovering later in the metrics.

        ``rev-list --all --not <base>`` answers this directly. The alternative
        of comparing commit timestamps needs GNU ``date`` and mis-sorts
        timezone-suffixed dates, which is how a release tag carrying the fix
        survived an earlier version of this check elsewhere in the codebase.
        """
        res = self.execute(
            f"git rev-list --all --not {self._base_ref} | head -n 5",
            cwd=self.repo_path,
        )
        if res.get("returncode", 1) != 0:
            raise RuntimeError(
                f"{self.instance_id}: git history check failed to run "
                f"(rc={res.get('returncode')}): {str(res.get('output'))[:500]}"
            )
        extra_commits = [line for line in str(res.get("output") or "").split() if len(line) == 40]
        if extra_commits:
            raise RuntimeError(
                f"{self.instance_id}: image git history is not truncated at {self._base_ref[:12]}; "
                f"commits outside its ancestry are reachable (e.g. {extra_commits[:3]}). "
                f"Rebuild the image with the history truncated, or set "
                f"git_leak_prevention: strip to truncate it per rollout."
            )

    # --- reward ---------------------------------------------------------------

    def _reset_test_files(self) -> dict:
        return self.reset_test_files_to_base(self._test_patch, self._base_ref)

    def _apply_test_patch(self) -> dict:
        """Copy the patch in as a file and apply it.

        A file plus ``git apply`` rather than a heredoc: the transfer is a tar
        stream, so it is byte-exact and has no command-length ceiling. No
        ``--reject`` retry -- a patch that only partly applies leaves an
        incomplete test suite, and a verdict computed from it is worse than a
        reported testbed failure.
        """
        try:
            self.copy_text_to(self._test_patch, self._TEST_PATCH_REMOTE)
        except Exception as e:  # noqa: BLE001
            return {"returncode": 1, "output": f"failed to copy test patch into the container: {e!r}"}
        try:
            return self.execute(f"git apply --verbose {self._TEST_PATCH_REMOTE}", cwd=self.repo_path)
        finally:
            self.execute(f"rm -f {self._TEST_PATCH_REMOTE}")

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        try:
            reset = self._reset_test_files()
            if reset.get("returncode", 1) != 0:
                return (
                    0.0,
                    str(reset.get("output") or ""),
                    {"error_category": REWARD_TESTBED_CORRUPTED, "error": "reset_tests_failed"},
                )

            applied = self._apply_test_patch()
            if applied.get("returncode", 1) != 0:
                return (
                    0.0,
                    str(applied.get("output") or ""),
                    {"error_category": REWARD_TESTBED_CORRUPTED, "error": "apply_test_patch_failed"},
                )

            exec_timeout = timeout if timeout is not None else self._verifier_timeout
            start = time.time()
            res = self.execute(self._test_command, cwd=self.repo_path, timeout=exec_timeout)
            duration = time.time() - start
            rc = res.get("returncode", 1)
            reward = 1.0 if rc == 0 else 0.0
            return (
                reward,
                str(res.get("output") or ""),
                {
                    "verifier_returncode": rc,
                    "resolved": reward == 1.0,
                    "test_duration": duration,
                    "test_command": self._test_command,
                },
            )
        finally:
            # Leave the tests as they were found so a second calculate_reward
            # starts from the same state; without this the next git apply would
            # hit "already exists in working directory".
            self._reset_test_files()
