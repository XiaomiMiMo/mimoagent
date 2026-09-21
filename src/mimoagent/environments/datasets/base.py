"""Base class + shared helpers for dataset-specific environments.

A :class:`DatasetEnvironment` wraps a base execution environment (k8s pod,
docker container, ...) with dataset-specific setup and reward calculation.
Subclasses implement two hooks:

* ``_setup_dataset_specific()`` — prepare the container for the agent run
  (git hygiene, anti-leak evictions, toolchain paths).
* ``_do_calculate_reward()`` — materialise the verifier, run it, grade.

The base class owns the cross-cutting concerns: model-patch capture, the
git leak-prevention lifecycle (strip/hide modes, see ``GIT_LEAK_PREVENTION_MODES``),
the optional rubric-judge reward stage (see ``REWARD_MODES`` and
``mimoagent.environments.rubric_judge``), TransportError handling, and a
toolbox of helpers (text transfer, patch-file extraction, git reset) that every
dataset needs in some combination.
"""

import os
import re
import tempfile
import uuid
from abc import ABC, abstractmethod

from mimoagent import Environment
from mimoagent.environments import TransportError
from mimoagent.utils.log import get_logger as _get_logger

# Returned as extra["error_category"] on a reward-time INFRA fault (not a model
# fault) — e.g. the gold test_patch fails to apply because an agent-planted file
# collides with a gold-added test path. RL trainers that consume mimoagent
# rewards treat this category as an infrastructure error and mask the whole
# sequence instead of training on a false reward=0. Kept as a literal so the
# trainer side can match on it; keep the string stable.
REWARD_TESTBED_CORRUPTED = "reward/testbed_corrupted"

_GIT_SHA_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def parse_git_head_output(output: str) -> str | None:
    """Return the commit hash while ignoring shell startup diagnostics."""
    hashes = _GIT_SHA_RE.findall(output or "")
    return hashes[-1].lower() if hashes else None


GIT_LEAK_PREVENTION_MODES = ("hide", "strip", "none")
"""Valid values for the ``git_leak_prevention`` yaml key (environment block):

* ``hide``  — move ``.git`` out of the repo to a random on-pod stash path for
  the whole agent run; restore it for reward calculation, re-hide after.
* ``strip`` — strip post-base commits/tags/refs in place (agent keeps a usable
  ``.git``); if the strip fails verification or errors out, fall back to
  ``hide``. **Default.**
* ``none``  — leave ``.git`` untouched.
"""

REWARD_MODES = ("auto", "programmatic", "rubric", "both")
"""Valid values for the ``reward_mode`` yaml key (environment block). Decides
which reward stages :meth:`DatasetEnvironment.calculate_reward` runs:

* ``programmatic`` — only the dataset's own verifier (``_do_calculate_reward``).
  Never runs the judge, even when rubrics and a judge are configured.
* ``rubric``       — only the rubric judge. Requires the instance to carry
  ``rubric.rubrics`` and the yaml to set ``judge_agent``; otherwise env
  creation fails loudly.
* ``both``         — judge first (it must see the doer's delivered state, before
  the verifier applies test patches / resets files), then the verifier. The
  scalar is combined per ``judge_agent.combine`` (default: the verifier's
  reward; see ``rubric_judge.RUBRIC_COMBINE_MODES``). Requires the same as
  ``rubric`` plus a dataset with a verifier (``HAS_VERIFIER``).
* ``auto``         — **Default.** ``both`` when rubrics AND a judge are present
  on a dataset with a verifier; ``rubric`` when they are present on a dataset
  without one (generic); ``programmatic`` otherwise. A test-graded dataset
  therefore behaves exactly as before until rubrics and a judge are both
  supplied.
"""


class DatasetEnvironment(ABC):
    """Base class for dataset-specific environment extensions."""

    REPO_PATH = "/testbed"
    """Path to the repository inside the container. Subclasses override."""

    HAS_VERIFIER = True
    """Whether ``_do_calculate_reward`` implements a programmatic verifier.
    Datasets without one (generic) set this False so ``reward_mode: auto``
    resolves to rubric-only when a judge is available, instead of calling
    into a NotImplementedError."""

    _REWARD_MODE_DEFAULT = "auto"
    """Default reward mode (see ``REWARD_MODES``). The yaml ``reward_mode``
    key (threaded via ``create_from_registry``) overrides this per-run."""

    _ANTI_HACK_CLEANUP_DEFAULT = False
    """Default for the build-env anti-hack cleanup (residue / build artifacts /
    global caches). OFF globally; a dataset built from committed build-env
    images can set this True. The yaml ``anti_hack_cleanup`` key (threaded via
    ``create_from_registry``) overrides this per-run when present."""

    _GIT_LEAK_PREVENTION_DEFAULT = "strip"
    """Default git leak-prevention mode (see ``GIT_LEAK_PREVENTION_MODES``).
    The yaml ``git_leak_prevention`` key (threaded via ``create_from_registry``)
    overrides this per-run when present."""

    @classmethod
    def default_docker_image(cls, instance: dict) -> str | None:
        """Image to run ``instance`` in when it carries no ``docker_image``.

        Datasets whose rows map deterministically onto published images
        override this; the default is "no idea", which makes
        :func:`mimoagent.environments.utils.make_dataset_env` fail loudly.
        """
        return None

    def __init__(self, base_env: Environment, instance: dict):
        """
        Args:
            base_env: The underlying execution environment.
            instance: The dataset instance (e.g., SWE-bench task).
        """
        self.env = base_env
        self.instance = instance

        # Git leak-prevention state. ``_git_stash_path`` is assigned on first
        # hide (random per-instance path) and marks that the hide lifecycle is
        # active; ``_git_hidden`` tracks whether .git is currently stashed away.
        self.git_leak_prevention = self._GIT_LEAK_PREVENTION_DEFAULT
        self._git_stash_path: str | None = None
        self._git_hidden = False
        # Normalised language ("go"/"python"/...) used by the build-artifact
        # purge to pick which dependency dirs to keep. Instances may carry
        # "language"; datasets without it get "" → _CLEAN_KEEP_UNKNOWN.
        self._language = (instance.get("language") or "").lower()
        # Whether to run the build-env anti-hack cleanup after setup. Class
        # default unless overridden by the yaml flag in create_from_registry.
        self.anti_hack_cleanup = self._ANTI_HACK_CLEANUP_DEFAULT
        # Idempotency guard: a subclass may call the cleanup at a specific point
        # inside its own setup (to control ordering). Once run, the
        # ``setup_environment`` post-hook skips it instead of running it twice.
        self._anti_hack_cleanup_done = False

        # Rubric-judge reward stage (see REWARD_MODES). ``rubrics`` is parsed
        # from the instance's optional ``rubric.rubrics`` (None when absent);
        # ``judge_agent_config`` (yaml ``environment.judge_agent`` block) and
        # ``reward_mode`` are injected by create_from_registry after
        # construction — the ctor keeps the uniform (base_env, instance)
        # signature for embedding frameworks.
        from mimoagent.environments.rubric_judge import parse_rubrics

        self.rubrics: list[dict] | None = parse_rubrics(instance)
        self.judge_agent_config: dict | None = None
        self.reward_mode: str = self._REWARD_MODE_DEFAULT
        # Rollout context for the judge, attached by the runner right before
        # reward (see attach_rollout).
        self._doer_agent: object | None = None
        self._task: str = ""
        self._doer_result: str = ""

    @property
    def logger(self):
        """The scoped instance logger (see ``mimoagent.utils.log.use_instance_logger``).

        Resolved on each access so the same env instance routes to whatever
        logger is currently bound — falls back to the module logger if unset.
        """
        return _get_logger()

    @property
    def repo_path(self) -> str:
        return self.REPO_PATH

    @property
    def instance_id(self) -> str:
        return self.instance.get("instance_id", "")

    def setup_environment(self) -> None:
        """Setup test environment after container creation.

        This should create test scripts, install dependencies, etc.
        """
        # Start the base environment first
        self.env.start()
        # Then run dataset-specific setup
        self._setup_dataset_specific()
        # Finally, the shared build-env anti-hack cleanup (gated by yaml flag /
        # class default). OFF for most datasets. Idempotent: if a subclass already
        # ran it mid-setup (to control ordering), this skips.
        if self.anti_hack_cleanup:
            self._run_build_env_anti_hack_cleanup()
        # Answer-leak blocklist: when the environment can reach the public
        # internet, blackhole the domains that host reference solutions (PR /
        # patch endpoints of the task's upstream repo) by pointing them at
        # 0.0.0.0 in /etc/hosts. Applied at the very end of setup so dependency
        # installation still works and only the rollout is restricted. Fails
        # closed: a write failure must abort rather than run with the leak open.
        blocklist = getattr(self.env.config, "answer_leak_blocklist", None)
        if blocklist:
            lines = "\\n".join(f"0.0.0.0 {h}" for h in blocklist)
            res = self.env.execute(
                f"printf '%b\\n' '# mimoagent answer-leak blocklist\\n{lines}' >> /etc/hosts && tail -n {len(blocklist) + 1} /etc/hosts"
            )
            if res.get("returncode") != 0:
                raise RuntimeError(
                    f"answer_leak_blocklist: writing /etc/hosts failed (rc={res.get('returncode')}): "
                    f"{str(res.get('output'))[:500]}"
                )
            self.logger.info(f"answer_leak_blocklist: blackholed {len(blocklist)} domain(s) {blocklist}")

    @abstractmethod
    def _setup_dataset_specific(self) -> None:
        """Dataset-specific setup logic. Override this in subclasses."""

    # --- Rubric judge stage --------------------------------------------------------

    def attach_rollout(self, *, agent: object | None = None, task: str = "", result: str = "") -> None:
        """Hand the finished rollout's context to the env for the rubric judge.

        Called by the runner between ``agent.run()`` and ``calculate_reward()``
        (``calculate_reward`` deliberately doesn't take these). All parts are
        optional — in recalc mode there is no agent and the judge works from
        task + patch + repo state alone. A no-op for datasets that never run
        the judge, so runners can call it unconditionally.
        """
        self._doer_agent = agent
        self._task = task or ""
        self._doer_result = result or ""

    def resolve_reward_plan(self) -> tuple[bool, bool]:
        """Return ``(run_judge, run_verifier)`` for the configured ``reward_mode``.

        Raises ValueError when an explicit mode asks for something the run
        can't provide (rubric/both without rubrics or a judge, both on a
        verifier-less dataset). ``create_from_registry`` calls this right
        after injecting the yaml overrides so misconfiguration fails before a
        pod is ever started; ``calculate_reward`` calls it again to act on it.
        """
        mode = self.reward_mode
        if mode not in REWARD_MODES:
            raise ValueError(f"Invalid reward_mode {mode!r}; expected one of {REWARD_MODES}")
        judge_available = bool(self.rubrics) and bool(self.judge_agent_config)
        if mode == "auto":
            if not judge_available:
                return False, True
            return True, self.HAS_VERIFIER
        if mode == "programmatic":
            return False, True
        # rubric / both: the judge is mandatory, so missing inputs are errors.
        if not self.rubrics:
            raise ValueError(f"{self.instance_id}: reward_mode={mode!r} but the instance carries no 'rubric.rubrics'")
        if not self.judge_agent_config:
            raise ValueError(
                f"reward_mode={mode!r} requires a judge agent: set the `judge_agent:` key "
                "(with its own `model:`) inside the yaml `environment:` block."
            )
        if mode == "both" and not self.HAS_VERIFIER:
            raise ValueError(
                f"{self.instance_id}: reward_mode='both' but {type(self).__name__} has no programmatic "
                "verifier; use reward_mode: rubric (or auto)"
            )
        return True, mode == "both"

    def _run_rubric_judge(self, model_patch: str) -> tuple[float, str, dict]:
        """Run the in-environment judge over ``self.rubrics`` (see rubric_judge.py)."""
        from mimoagent.environments.rubric_judge import RubricJudge, RubricJudgeConfig

        cfg = RubricJudgeConfig.from_dict(self.judge_agent_config)
        judge = RubricJudge(self, self.rubrics or [], cfg)
        return judge.run(
            model_patch=model_patch,
            task=self._task,
            doer_agent=self._doer_agent,
            doer_result=self._doer_result,
        )

    def _combine_rewards(
        self,
        verifier_out: tuple[float, str, dict] | None,
        judge_out: tuple[float, str, dict] | None,
    ) -> tuple[float, str, dict]:
        """Fold the verifier's and the judge's results into one reward triple.

        * verifier only → returned untouched (the historical contract; no new
          keys, so test-graded datasets are byte-for-byte unaffected).
        * judge only → the judge's triple, tagged ``reward_mode="rubric"``.
        * both → extras merged (verifier keys win on collision), both scores
          reported (``programmatic_reward`` / ``rubric_reward``), scalar per
          ``judge_agent.combine``. A judge that failed to deliver
          (``error_category`` set) fails closed: reward 0 with the category
          kept, so an RL trainer masks the sequence rather than training on a reward
          missing one of its components.
        """
        if judge_out is None:
            if verifier_out is None:
                raise RuntimeError(f"{self.instance_id}: reward plan ran neither verifier nor judge")
            return verifier_out
        j_reward, j_output, j_extra = judge_out
        if verifier_out is None:
            extra = {**j_extra, "reward_mode": "rubric", "rubric_reward": j_reward}
            return j_reward, j_output, extra

        from mimoagent.environments.rubric_judge import RubricJudgeConfig

        cfg = RubricJudgeConfig.from_dict(self.judge_agent_config)
        p_reward, p_output, p_extra = verifier_out
        extra = dict(p_extra)
        for k, v in j_extra.items():
            extra.setdefault(k, v)
        extra.update(
            {
                "reward_mode": "both",
                "reward_combine": cfg.combine,
                "programmatic_reward": p_reward,
                "rubric_reward": j_reward,
            }
        )
        if j_extra.get("error_category"):
            reward = 0.0
        else:
            reward = cfg.combine_rewards(p_reward, j_reward)
        # Verifier output first, judge summary last: batch.py keeps the TAIL of
        # test_output when truncating, and the short per-rubric summary is the
        # part worth surviving.
        test_output = f"{p_output}\n\n=== rubric judge ===\n{j_output}"
        return reward, test_output, extra

    @staticmethod
    def _merge_judge_extra(result: dict, judge_out: tuple[float, str, dict] | None) -> None:
        """On a verifier exception, keep whatever the (already finished) judge produced."""
        if judge_out is None:
            return
        j_reward, _, j_extra = judge_out
        for k, v in j_extra.items():
            result.setdefault(k, v)
        result["rubric_reward"] = j_reward

    def calculate_reward(self, timeout: int | float | None = None) -> tuple[float, str, dict]:
        """Template method: runs reward with TransportError handling + git restore.

        Subclasses implement _do_calculate_reward() with the actual reward logic.
        This base handles:
        - restoring a hidden .git before grading and re-hiding it after (only
          when the hide lifecycle is active — see _prevent_git_hack)
        - model patch capture (always returned in extra["model_patch"])
        - the optional rubric-judge stage (see REWARD_MODES): judge first, so it
          sees the state the doer delivered — verifiers routinely mutate the
          testbed (apply the gold test patch, reset test files, git clean)
        - TransportError → (0.0, str(e), {"model_patch": ..., "transport_error": True})
        - Generic exception catch → (0.0, str(e), {"model_patch": ...})
        """
        model_patch = ""
        patch_error = ""
        judge_out: tuple[float, str, dict] | None = None
        try:
            if self._git_hidden:
                self._restore_git()
            model_patch, patch_error = self._capture_model_diff()
            run_judge, run_verifier = self.resolve_reward_plan()
            if run_judge:
                judge_out = self._run_rubric_judge(model_patch)
            verifier_out = None
            if run_verifier:
                verifier_out = self._do_calculate_reward(timeout=timeout, model_patch=model_patch)
            reward, test_output, extra = self._combine_rewards(verifier_out, judge_out)
            extra["model_patch"] = model_patch
            if patch_error:
                extra["model_patch_error"] = patch_error
            return reward, test_output, extra
        except TransportError as e:
            self.logger.warning(f"{self.instance_id}: TransportError during reward: {repr(e)}")
            result = {"model_patch": model_patch, "transport_error": True}
            if patch_error:
                result["model_patch_error"] = patch_error
            self._merge_judge_extra(result, judge_out)
            return 0.0, str(e), result
        except Exception as e:
            self.logger.error(f"{self.instance_id}: Error calculating reward: {repr(e)}")
            result = {"model_patch": model_patch}
            if patch_error:
                result["model_patch_error"] = patch_error
            self._merge_judge_extra(result, judge_out)
            return 0.0, str(e), result
        finally:
            # Re-hide so a re-run (recalc) starts from the same hidden state.
            if self._git_stash_path is not None and not self._git_hidden:
                try:
                    self._hide_git()
                except Exception:
                    self.logger.warning(f"{self.instance_id}: re-hiding .git failed (pod likely dead)")

    @abstractmethod
    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        """Subclass implements actual reward logic. May raise freely.

        Args:
            timeout: Max execution time for test commands.
            model_patch: The unified diff captured by the base class (staged+unstaged
                changes relative to HEAD). Subclasses may use this if they need the
                diff for grading/reporting, avoiding redundant git operations.

        Called by calculate_reward() inside a try/except that handles
        TransportError and generic exception fallback.
        Git restore/hide is handled by the base — do NOT call them here.
        """

    def _capture_model_diff(self) -> tuple[str, str]:
        """Capture staged+unstaged changes as unified diff.

        Assumes .git is available (caller handles restore/hide lifecycle).
        Returns (diff_text, error_msg). error_msg is empty on success.
        Never raises.
        """
        try:
            res = self.execute(
                "git add -A >/dev/null 2>&1 && git -c core.fileMode=false diff HEAD 2>/dev/null",
                cwd=self.repo_path,
                timeout=120,
            )
            rc = res.get("returncode")
            reason = res.get("reason")
            if rc not in (0, None) or reason not in ("ok", None, ""):
                msg = f"rc={rc} reason={reason}"
                self.logger.warning(f"{self.instance_id}: _capture_model_diff failed, {msg}")
                return "", msg
            return str(res.get("output", "") or ""), ""
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: _capture_model_diff error: {e}")
            return "", str(e)

    # --- Pass through to base environment -------------------------------------

    def execute(self, command: str, **kwargs) -> dict:
        return self.env.execute(command, **kwargs)

    def cleanup(self):
        if hasattr(self.env, "cleanup"):
            self.env.cleanup()

    def get_template_vars(self) -> dict:
        if hasattr(self.env, "get_template_vars"):
            return self.env.get_template_vars()
        return {}

    # --- Shared helpers --------------------------------------------------------

    def copy_text_to(
        self, text: str, remote_path: str, *, executable: bool = False, env: Environment | None = None
    ) -> None:
        """Write ``text`` to ``remote_path`` inside the container via copy_to.

        Tar-stream transfer: no ARG_MAX ceiling, no shell quoting, byte-exact.
        Raises RuntimeError on failure. ``env`` targets a different environment
        than the wrapped one (e.g. a separate eval pod); default ``self.env``.
        """
        target = env if env is not None else self.env
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False) as f:
            f.write(text)
            local_path = f.name
        try:
            target.copy_to(local_path, remote_path)
        except Exception as e:
            raise RuntimeError(f"Failed to copy file to container ({remote_path}): {repr(e)}")
        finally:
            os.unlink(local_path)

        if executable:
            result = target.execute(f"chmod +x {remote_path}")
            if result.get("returncode", 1) != 0:
                raise RuntimeError(f"Failed to chmod +x {remote_path}: {result.get('output', '')}")

    @staticmethod
    def get_patch_modified_files(patch: str) -> list[str]:
        """File paths a git patch modifies — ``--- a/<path>`` entries only.

        Catches edits to existing files but NOT newly added files (those have
        ``--- /dev/null``). Use :meth:`get_patch_touched_files` when new files
        matter (e.g. anti-tamper resets).
        """
        files: list[str] = []
        for line in patch.split("\n"):
            if line.startswith("--- a/"):
                fp = line[6:].strip()
                if fp and fp != "/dev/null" and fp not in files:
                    files.append(fp)
        return files

    @staticmethod
    def get_patch_touched_files(patch: str) -> list[str]:
        """All file paths a git patch touches, including new files.

        Both sides of each ``diff --git a/<old> b/<new>`` header are collected:
        for renames a-side != b-side, and the rename SOURCE must also be reset
        to base — otherwise re-applying the patch onto a worktree where it was
        already applied dies with ``<old>: No such file or directory``.
        """
        files: list[str] = []
        for line in patch.split("\n"):
            m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
            if m:
                for fp in (m.group(1), m.group(2)):
                    if fp not in files:
                        files.append(fp)
        return files

    def build_reset_test_files_cmd(self, test_patch: str, base_commit: str) -> str | None:
        """Bash that gives every test_patch-touched path a clean, base-aligned
        starting point before the gold test_patch is applied.

        Why per-file (not ``git checkout base -- <files>``): the agent may have
        created a NEW file at a path the gold test_patch ADDS. ``git checkout
        base`` cannot remove a path absent from base, so the file (often also
        staged by an upstream ``git add -A``) lingers and ``git apply`` then
        fails with ``already exists in working directory`` — a non-model-fault
        reward=0. Decide per file by existence IN base_commit (not by what git
        currently tracks):
          exists in base → restore to base
          absent in base → gold-added (possibly agent-planted): unstage + rm

        Returns the bash snippet, or ``None`` when there is nothing to reset.
        """
        touched = self.get_patch_touched_files(test_patch)
        if not (touched and base_commit):
            return None
        files_blob = "\n".join(touched)
        return (
            "while IFS= read -r tf; do\n"
            '  [ -z "$tf" ] && continue\n'
            f'  if git cat-file -e {base_commit}:"$tf" 2>/dev/null; then\n'
            f'    git checkout {base_commit} -- "$tf" 2>/dev/null || true\n'
            "  else\n"
            '    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n'
            '    rm -f "$tf"\n'
            "  fi\n"
            "done <<'EOF_RESET_TEST_FILES'\n"
            f"{files_blob}\n"
            "EOF_RESET_TEST_FILES\n"
        )

    def reset_test_files_to_base(self, test_patch: str, base_commit: str) -> dict:
        """Execute :meth:`build_reset_test_files_cmd` in ``repo_path`` (python-level
        callers). Returns the execute() result; ``returncode==0`` with a no-op
        message when there is nothing to reset."""
        cmd = self.build_reset_test_files_cmd(test_patch, base_commit)
        if cmd is None:
            return {"returncode": 0, "output": "no test files to reset"}
        return self.execute(cmd, cwd=self.repo_path)

    def git_reset_hard(self, base_commit: str, *, clean: bool = False, timeout: int = 600) -> dict:
        """``git reset --hard <base_commit>`` (optionally ``+ git clean -fd``) in repo_path.

        A ``base_commit`` missing from the clone (e.g. a PR merged after the
        image was built) is fetched from origin and the reset retried. Raises
        ``RuntimeError`` unless the command succeeds and HEAD verifiably lands
        on ``base_commit`` — running a task from the wrong commit silently
        invalidates it, so this is fatal rather than a warning.

        ``clean=False`` matters for images that ship prebuilt untracked
        artifacts (e.g. compiled .so files): ``git clean`` would wipe them.
        """
        cmd = f"git reset --hard {base_commit}"
        if clean:
            cmd += " && git clean -fd"
        result = self.execute(cmd, cwd=self.repo_path, timeout=timeout)
        if result.get("returncode", 1) != 0:
            self.execute(f"git fetch origin {base_commit}", cwd=self.repo_path, timeout=1800)
            result = self.execute(cmd, cwd=self.repo_path, timeout=timeout)
        rev_parse = self.execute("git rev-parse HEAD", cwd=self.repo_path)
        head = parse_git_head_output(rev_parse.get("output", ""))
        if (
            result.get("returncode", 1) != 0
            or rev_parse.get("returncode", 1) != 0
            or not head
            or not head.startswith(base_commit[:12].lower())
        ):
            raise RuntimeError(
                f"{self.instance_id}: git reset --hard {base_commit} failed, "
                f"HEAD={(head or rev_parse.get('output', ''))[:12]} "
                f"(rc={result.get('returncode')}): {result.get('output', '')[:2000]}"
            )
        return result

    def git_add_safe_directory(self) -> dict:
        return self.env.execute(f"git config --global --add safe.directory {self.repo_path}")

    def append_goroot_to_bashrc(self) -> None:
        """Export GOROOT/PATH for Go images where /usr/local/go isn't on PATH."""
        res = self.env.execute("echo -e '\nexport GOROOT=/usr/local/go\nexport PATH=$GOROOT/bin:$PATH' >> ~/.bashrc")
        if res.get("returncode", 1) != 0:
            raise RuntimeError(f"Failed to append /usr/local/go to PATH: {res.get('output', '')}")

    # --- Build-env anti-hack cleanup (shared; gated by ``anti_hack_cleanup``) ---
    #
    # Three best-effort purges that strip answer-leaking residue baked into a
    # committed build-env image: build-time logs / verifier output (residue),
    # compiled artifacts that decompile back to the golden fix (build artifacts),
    # and the project's own caches living outside the repo (global caches).
    # Any dataset can opt in via the yaml ``anti_hack_cleanup`` flag.
    # Repo-relative paths use ``self.repo_path`` (/testbed by default,
    # deepswe=/app).

    # Global (absolute, repo-agnostic) residue paths. The repo-relative bits
    # (node_modules caches under repo_path) are appended in _purge_build_residue.
    # Version/engine agnostic: ``find /var/log -type f -delete`` covers
    # postgresql-N / mysql / mongodb without hardcoding versions.
    _RESIDUE_SCRUB_GLOBAL = (
        "rm -f /tmp/fail.log /tmp/pass.log /tmp/patch.diff /tmp/test_patch.diff /tmp/*.log 2>/dev/null; "
        "rm -rf /tmp/claude-0 /tmp/claude-* /tmp/testem-* /tmp/puppeteer_dev_chrome_profile-* 2>/dev/null; "
        # extra /tmp build/verifier residue (test_files.json = hidden-test file list;
        # jest transform cache holds expected results/snapshots; /tmp/build leftovers).
        "rm -f /tmp/test_files.json 2>/dev/null; "
        "rm -rf /tmp/jest_* /tmp/jest-* /tmp/build /tmp/build_env 2>/dev/null; "
        # more /tmp verifier residue found by env_hacker audit: pytest tmp projects,
        # go test build cache, Coverlet (.NET) tmp*.tmp, unpacked toolchains/clones,
        # editor .bak copies, expect/diff dumps, PHPCS standards git clone.
        "rm -rf /tmp/pytest-of-root /tmp/pytest-* /tmp/.pytest_cache /tmp/__pycache__ 2>/dev/null; "
        "rm -rf /tmp/go-build* /tmp/standards /tmp/wordpress /tmp/zig-* /tmp/testbase /tmp/testbed 2>/dev/null; "
        "rm -f /tmp/tmp*.tmp /tmp/*.bak /tmp/expect* /tmp/butwas* 2>/dev/null; "
        "rm -rf /root/.cache/go-build 2>/dev/null; "
        "rm -rf /tests /logs 2>/dev/null; "
        "find /var/log -type f -delete 2>/dev/null || true; "
        "rm -rf /var/lib/postgresql/*/*/log /var/lib/postgresql/*/*/pg_log 2>/dev/null || true; "
        "find /var/lib/mysql /var/lib/mongodb -type f -name '*.log' -delete 2>/dev/null || true; "
        "rm -rf /root/.npm/_logs /root/.babel.json /root/.pytest_cache 2>/dev/null || true; "
        "find / -maxdepth 4 -xdev -name task_description.md -path '*/.build_env/*' -delete 2>/dev/null || true; "
        "true"
    )

    def _purge_build_residue(self) -> None:
        """Delete build-time answer-leak residue baked into the image.

        Best-effort: the agent run shouldn't be blocked if scrub partially fails,
        so we just log. KEEPS .build_env/test_command.sh (only its sibling
        task_description.md is dropped) — build-env rewards need the test runner.
        """
        # Repo-relative build caches inside kept dep dirs (regenerated on demand).
        repo_scrub = f"rm -rf {self.repo_path}/node_modules/.cache {self.repo_path}/node_modules/.vitest 2>/dev/null; "
        try:
            self.env.execute(repo_scrub + self._RESIDUE_SCRUB_GLOBAL, timeout=120)
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: build-residue scrub failed (non-fatal): {repr(e)}")

    # Dirs to KEEP during `git clean -fdx` (offline rebuild needs them). git clean
    # removes everything else untracked/ignored — including build OUTPUT (maven
    # target/, gradle build/, dist/, __pycache__, *.class ...) which is what we want
    # gone. _CLEAN_KEEP_COMMON applies to ALL languages (unambiguous dependency /
    # tooling dirs that never hold the project's own source answer); per-language
    # entries add the cache-vs-output-ambiguous dirs (target = rust dep cache, etc.).
    _CLEAN_KEEP_COMMON = [
        "node_modules",
        "bower_components",
        ".husky",  # js/ts deps + git hooks
        "vendor",
        "third_party",
        "_deps",
        "vcpkg_installed",  # go/php/cmake deps
    ]
    _CLEAN_KEEP = {
        "python": [".venv", "venv"],
        "java": [".gradle"],
        "kotlin": [".gradle"],
        "scala": [".gradle"],
        "rust": ["target"],
        "swift": [".build"],
        "crystal": ["lib"],  # shards deps
        "ruby": [".bundle"],  # bundler config / git-source gems (vendor in common)
        "julia": ["Manifest.toml"],  # resolved-deps manifest (else Pkg.instantiate fails)
    }
    # Fallback for unknown languages: union of all per-language ambiguous dirs.
    _CLEAN_KEEP_UNKNOWN = [
        ".venv",
        "venv",
        ".gradle",
        "target",
        ".build",
        "lib",
        ".bundle",
        "Manifest.toml",
        "_build",
        ".stack-work",
        "dist-newstyle",
    ]

    # Rust: strip the project's OWN compiled crates + test binaries from target/
    # while keeping third-party crate rlibs in target/*/deps (offline rebuild). The
    # prior pipeline only deleted the root [package].name; workspaces (e.g. RustPython
    # with rustpython_vm / rustpython_ast / ... members) leaked the members' rlibs.
    # Here we enumerate every crate name from all Cargo.toml files and drop them.
    # The ``cd <repo>`` line is prepended at call time (avoids brace-escaping the
    # bash ${var} expansions below through str.format).
    _RUST_PURGE_BODY = r"""
NAMES=$(grep -hoE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"[^"]+"' \
          $(find . -maxdepth 4 -name Cargo.toml 2>/dev/null) 2>/dev/null \
        | sed -E 's/.*"([^"]+)".*/\1/' | sort -u)
for n in $NAMES; do
  u=$(printf '%s' "$n" | tr '-' '_')
  for d in target/debug target/release; do
    [ -d "$d" ] || continue
    rm -f "$d/$n" "$d/lib$u.rlib" "$d/lib$u.so" "$d/lib$u.dylib" "$d/lib$u.a" 2>/dev/null || true
    rm -f "$d"/deps/lib${u}-*.rlib "$d"/deps/lib${u}-*.rmeta \
          "$d"/deps/lib${u}-*.so "$d"/deps/${u}-* "$d"/deps/${n}-* 2>/dev/null || true
  done
done
rm -rf target/*/incremental target/*/.fingerprint 2>/dev/null || true
true
"""

    def _purge_build_artifacts(self) -> None:
        """Remove compiled build artifacts that leak the golden fix / hidden test,
        keeping the per-language dependency caches needed for an offline rebuild
        at reward time. Best-effort (never blocks the run)."""
        lang = self._language
        lang_keep = self._CLEAN_KEEP.get(lang, self._CLEAN_KEEP_UNKNOWN)
        keep = list(dict.fromkeys(self._CLEAN_KEEP_COMMON + lang_keep))  # dedup, keep order
        excludes = " ".join(f"--exclude={d}" for d in keep)
        try:
            # ① git clean: drop untracked/ignored build OUTPUT (target/build/dist/
            #    __pycache__/*.class ...), keep dep caches via --exclude.
            self.env.execute(f"git clean -fdx {excludes}", cwd=self.repo_path, timeout=300)
            # ② Rust/Swift: target//.build kept wholesale → strip project crates + tests.
            if lang == "rust":
                self.env.execute(f"cd {self.repo_path}\n" + self._RUST_PURGE_BODY, timeout=300)
            elif lang == "swift":
                self.env.execute(f"cd {self.repo_path} && rm -rf .build/*/build 2>/dev/null || true", timeout=120)
            # ③ Project installed OUTSIDE the repo (git clean can't reach it).
            if lang == "python":
                # A `pip install .` puts a (fixed) copy of the project into system/
                # global site-packages, which also shadows the repo at import time.
                # Overwrite it with an editable link to the base-commit source.
                self.env.execute(
                    f"pip install -e {self.repo_path} --no-deps --force-reinstall -q 2>/dev/null || true",
                    timeout=300,
                )
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: build-artifact purge failed (non-fatal): {repr(e)}")

    # Answer-leaking artifacts OUTSIDE the repo (git clean can't reach). env_hacker
    # audit found these are the single biggest leak vector. All are caches that the
    # build tool regenerates on demand at reward time (cost: recompile/re-precompile).
    #   - ~/.m2 SNAPSHOT jars: the project's OWN built jar/sources/tests (SNAPSHOT =
    #     locally built; third-party releases are not SNAPSHOT, so they're kept). poms
    #     are kept so resolution still works; mvn rebuilds the jar from the repo.
    #   - ~/.julia/compiled *.ji: golden source strings embedded in precompiled cache.
    #   - ~/.gradle build-cache (compiled bytecode) + daemon logs (javac warnings leak
    #     source lines, test reports leak hidden tests) + script/jar caches.
    #   - ~/.cache/bazel: stale objects + testlogs.
    _GLOBAL_CACHE_SCRUB = (
        "find /root/.m2 -type f \\( -name '*-SNAPSHOT.jar' -o -name '*-SNAPSHOT-sources.jar' "
        "-o -name '*-SNAPSHOT-tests.jar' -o -name '*-SNAPSHOT-test-sources.jar' \\) -delete 2>/dev/null; "
        "rm -rf /root/.julia/compiled 2>/dev/null; "
        "rm -rf /root/.gradle/caches/build-cache-* /root/.gradle/daemon 2>/dev/null; "
        "rm -rf /root/.gradle/caches/*/scripts /root/.gradle/caches/jars-* 2>/dev/null; "
        "rm -rf /root/.cache/bazel 2>/dev/null; "
        "true"
    )

    def _purge_global_caches(self) -> None:
        """Delete answer-leaking caches outside the repo (~/.m2 SNAPSHOT jars,
        Julia/Gradle/Bazel compile caches). Best-effort; build tools regenerate
        them at reward time (extra recompile cost)."""
        try:
            self.env.execute(self._GLOBAL_CACHE_SCRUB, timeout=180)
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: global-cache purge failed (non-fatal): {repr(e)}")

    def _run_build_env_anti_hack_cleanup(self) -> None:
        """Run the three build-env anti-hack purges (residue / build artifacts /
        global caches). Each is individually best-effort. Idempotent: runs at most
        once per env (guarded by ``_anti_hack_cleanup_done``) so a subclass calling
        it mid-setup and the ``setup_environment`` post-hook don't double-run it.

        If ``.git`` is currently hidden (hide-mode git leak prevention ran earlier
        in setup), it is restored for the purge — ``git clean -fdx`` needs it —
        and re-hidden after."""
        if self._anti_hack_cleanup_done:
            return
        self._anti_hack_cleanup_done = True
        restore_needed = self._git_hidden
        if restore_needed:
            self._restore_git()
        try:
            self._purge_build_residue()
            self._purge_build_artifacts()
            self._purge_global_caches()
        finally:
            if restore_needed:
                self._hide_git()

    # --- Git leak prevention ------------------------------------------------------

    def _prevent_git_hack(self, cutoff_commit: str | None = None) -> None:
        """Keep the agent from mining the golden fix out of git history.

        Dispatches on ``self.git_leak_prevention`` (yaml ``git_leak_prevention``,
        see ``GIT_LEAK_PREVENTION_MODES``):

        * ``hide``  — stash the whole ``.git`` outside the repo for the agent
          run; :meth:`calculate_reward` restores it for grading.
        * ``strip`` (default) — strip future commits in place so the agent
          keeps a usable ``.git``; on strip failure/timeout fall back to
          ``hide`` (degraded but leak-free) instead of aborting the rollout.
        * ``none``  — no-op.

        ``cutoff_commit`` overrides the strip point (default: the instance's
        ``base_commit``). Everything not in the cutoff's ancestry is stripped.
        some datasets pass the PR HEAD so the PR's own base..head commits (the
        subject under review) survive while the merge / follow-up fixes are
        hidden.
        """
        mode = self.git_leak_prevention
        if mode == "none":
            return
        if mode == "hide":
            self._hide_git()
            return
        if mode != "strip":
            raise ValueError(f"Invalid git_leak_prevention mode {mode!r}; expected one of {GIT_LEAK_PREVENTION_MODES}")
        base = cutoff_commit or self.instance["base_commit"]
        if not self._strip_future_commits(base):
            self.logger.warning(
                f"{self.instance_id}: in-place git strip failed/unverified; falling back to hiding .git"
            )
            self._hide_git()

    def _strip_future_commits(self, base: str) -> bool:
        """Strip everything not in ``base``'s ancestry from ``.git``, in place.

        Returns True iff the post-strip verification confirms no post-base
        commits remain reachable. Never raises — any error (including execute
        timeouts / transport faults) returns False so the caller can fall back
        to hiding ``.git``.

        Reference: https://github.com/SWE-bench/SWE-bench/pull/471/files#diff-ffb13d93f3c39cdd17161d974faa704d32d3500f8495a60bfdc31c3c30adf52e

        How the strip works:
          - ``git checkout --detach {base}`` re-points HEAD onto base FIRST, so
            a current-branch tip that sits *ahead* of base (the main reason the
            old verify failed) no longer keeps future commits reachable.
          - all branches are then deleted (none is current after detach),
            remotes removed, tags newer than base deleted (old tags kept so
            ``git describe`` / setuptools_scm-style version detection still
            works), reflog expired, and ``gc --prune=now`` physically drops the
            now-unreachable future-commit objects (so ``git --git-dir`` /
            ``fsck`` / ``cat-file`` can't recover them).
        """
        cleanup_commands = [
            # ★ detach HEAD onto base FIRST so the current branch tip (possibly
            #   ahead of base) doesn't keep future commits reachable.
            f"git checkout --detach {base} 2>/dev/null || true",
            # delete every local branch (none is current after detach)
            'git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true',
            "git remote | xargs -r -n1 git remote remove 2>/dev/null || true",
            # drop non-branch/tag refs that can keep future commits reachable
            # (git-replace can even swap base for another commit; notes/stash/pull too).
            'git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true',
            # ★ delete every tag whose commit is NOT in base's history. Uses
            #   `merge-base --is-ancestor` (reliable) instead of comparing %ci
            #   date strings (the old bug — timezone-suffixed dates mis-sort, so
            #   the release tag carrying the golden fix survived). Ancestor tags
            #   (real old releases) are KEPT so `git describe` / setuptools_scm
            #   version detection still works.
            f'git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" {base} 2>/dev/null; then git tag -d "${{t#refs/tags/}}" >/dev/null 2>&1; fi; done || true',
            "git reflog expire --expire=now --all 2>/dev/null || true",
            # physically delete the now-unreachable future commits. Plain
            # `gc --prune=now` is NOT enough: git's repack/cruft-pack step honours
            # `gc.pruneExpire` (default 2 weeks), so RECENTLY-created unreachable
            # objects (the just-built future commits) get parked in a cruft pack
            # and stay recoverable via `git cat-file`/`fsck --lost-found`. Forcing
            # `gc.pruneExpire=now` + disabling cruft packs drops them immediately
            # (this single gc already repacks + prunes — no extra repack -ad needed,
            # which only added cost on large .git without changing the result).
            "git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now 2>/dev/null || true",
        ]

        verify_commands = [
            f"TARGET_TIMESTAMP=$(git show -s --format=%ci {base})",
            "AFTER_TIMESTAMP=$(date -d \"$TARGET_TIMESTAMP + 1 second\" '+%Y-%m-%d %H:%M:%S')",
            'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)',
            'echo "COMMIT_COUNT: $COMMIT_COUNT"',
            '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        ]

        cleanup_script = "set -eo pipefail\n" + "\n".join(cleanup_commands)
        verify_script = "set -euo pipefail\n" + "\n".join(verify_commands)

        try:
            self.execute(cleanup_script, timeout=600)
            result = self.execute(verify_script, timeout=120)
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: git strip errored: {repr(e)}")
            return False
        return result.get("returncode", 1) == 0

    def _hide_git(self) -> None:
        """Move ``.git`` out of the repo to a random on-pod stash path.

        The random path (fixed per instance after first hide) avoids the old
        guessable ``{repo}_git_backup`` location; a root agent could still find
        it via ``find / -name HEAD``, which is the accepted trade-off of hide
        mode vs. physically deleting history. Any agent-planted ``.git`` in the
        repo is dropped first so a later restore lands cleanly.
        """
        if self._git_stash_path is None:
            self._git_stash_path = f"/usr/lib/.{uuid.uuid4().hex}"
        res = self.env.execute(f"mv {self.repo_path}/.git {self._git_stash_path}")
        if res.get("returncode", 1) != 0:
            raise RuntimeError(f"Failed to hide .git directory: {res.get('output', '')}")
        self._git_hidden = True

    def _restore_git(self) -> None:
        """Move the stashed ``.git`` back into the repo for reward calculation.

        Drops any ``.git`` the agent may have planted in the repo (e.g. its own
        ``git init``) so the stash lands at the canonical path.
        """
        res = self.env.execute(f"rm -rf {self.repo_path}/.git && mv {self._git_stash_path} {self.repo_path}/.git")
        if res.get("returncode", 1) != 0:
            raise RuntimeError(f"Failed to restore .git directory: {res.get('output', '')}")
        self._git_hidden = False
