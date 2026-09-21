"""Unit tests for the git leak-prevention modes (``git_leak_prevention``):
hide / strip (with hide fallback) / none, plus the restore↔re-hide lifecycle
around ``calculate_reward``."""

import pytest

from mimoagent.environments.datasets import create_from_registry
from mimoagent.environments.datasets.base import DatasetEnvironment


class _FakeBaseEnv:
    """Records executed commands; scriptable per-command returncodes."""

    def __init__(self):
        self.commands: list[str] = []
        # (substring, returncode) rules checked in order; default rc=0.
        self.rules: list[tuple[str, int]] = []

    def execute(self, command: str, **kwargs) -> dict:
        self.commands.append(command)
        for substr, rc in self.rules:
            if substr in command:
                return {"returncode": rc, "output": ""}
        return {"returncode": 0, "output": ""}


class _Dummy(DatasetEnvironment):
    def _setup_dataset_specific(self) -> None:
        self._prevent_git_hack()

    def _do_calculate_reward(self, timeout=None, model_patch=""):
        return (1.0, "ok", {})


def _env(mode: str | None = None) -> _Dummy:
    instance = {"instance_id": "i1", "base_commit": "BASE123"}
    env = _Dummy(_FakeBaseEnv(), instance)
    if mode is not None:
        env.git_leak_prevention = mode
    return env


def _joined(env: _Dummy) -> str:
    return "\n===\n".join(env.env.commands)


def test_default_mode_is_strip():
    assert _env().git_leak_prevention == "strip"


def test_none_mode_touches_nothing():
    env = _env("none")
    env._prevent_git_hack()
    assert env.env.commands == []
    assert not env._git_hidden


def test_hide_mode_moves_git_to_random_stash():
    env = _env("hide")
    env._prevent_git_hack()
    assert env._git_hidden
    assert env._git_stash_path is not None
    assert f"mv {env.repo_path}/.git {env._git_stash_path}" in _joined(env)
    # no strip attempted
    assert "gc --prune=now" not in _joined(env)


def test_strip_mode_success_does_not_hide():
    env = _env("strip")
    env._prevent_git_hack()
    joined = _joined(env)
    assert "gc --prune=now" in joined
    assert "COMMIT_COUNT" in joined  # verify script ran
    assert not env._git_hidden
    assert env._git_stash_path is None


def test_strip_falls_back_to_hide_when_verify_fails():
    env = _env("strip")
    env.env.rules = [("COMMIT_COUNT", 1)]  # verify script fails
    env._prevent_git_hack()
    assert env._git_hidden
    assert f"mv {env.repo_path}/.git" in _joined(env)


def test_strip_falls_back_to_hide_when_strip_errors():
    env = _env("strip")
    orig = env.env.execute

    def flaky(command, **kwargs):
        if "gc --prune=now" in command:
            raise TimeoutError("execute timed out")
        return orig(command, **kwargs)

    env.env.execute = flaky
    env._prevent_git_hack()
    assert env._git_hidden


def test_hide_raises_when_mv_fails():
    env = _env("hide")
    env.env.rules = [("mv ", 1)]
    with pytest.raises(RuntimeError, match="hide .git"):
        env._prevent_git_hack()


def test_invalid_mode_rejected_by_registry():
    with pytest.raises(ValueError, match="git_leak_prevention"):
        create_from_registry(
            _FakeBaseEnv(),
            {"instance_id": "i1", "base_commit": "B"},
            git_leak_prevention="bogus",
        )


def test_registry_threads_mode_onto_env():
    env = create_from_registry(
        _FakeBaseEnv(),
        {"instance_id": "i1", "base_commit": "B"},
        git_leak_prevention="hide",
    )
    assert env.git_leak_prevention == "hide"


def test_calculate_reward_restores_then_rehides():
    env = _env("hide")
    env._prevent_git_hack()
    stash = env._git_stash_path
    env.env.commands.clear()

    reward, _, extra = env.calculate_reward()
    assert reward == 1.0
    joined = _joined(env)
    restore_idx = joined.index(f"mv {stash} {env.repo_path}/.git")
    rehide_idx = joined.index(f"mv {env.repo_path}/.git {stash}")
    assert restore_idx < rehide_idx  # restore before grading, re-hide after
    assert env._git_hidden  # ends hidden, ready for recalc
    # stash path is stable across the cycle
    assert env._git_stash_path == stash


def test_calculate_reward_skips_git_lifecycle_when_not_hidden():
    env = _env("strip")
    env._prevent_git_hack()  # strip succeeds → no hide lifecycle
    env.env.commands.clear()
    reward, _, _ = env.calculate_reward()
    assert reward == 1.0
    assert "mv " not in _joined(env)
