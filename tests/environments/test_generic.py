from types import SimpleNamespace

import pytest

from mimoagent.environments.datasets.generic import GenericGitEnvironment

BASE = "0123456789abcdef0123456789abcdef01234567"


class _FakeEnvironment:
    def __init__(self, cwd="/testbed"):
        self.config = SimpleNamespace(cwd=cwd)
        self.commands = []

    def start(self):
        pass

    def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return {"output": "", "returncode": 0}


class _ResetEnvironment(_FakeEnvironment):
    """Fake env scripting git reset / rev-parse behavior for git_reset_hard."""

    def __init__(self, fail_resets=0, head=BASE):
        super().__init__()
        self.fail_resets = fail_resets
        self.head = head
        self.reset_calls = 0

    def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("git reset --hard"):
            self.reset_calls += 1
            if self.reset_calls <= self.fail_resets:
                return {"output": "fatal: unknown revision", "returncode": 128}
            return {"output": f"HEAD is now at {BASE[:12]}", "returncode": 0}
        if command == "git rev-parse HEAD":
            return {"output": self.head, "returncode": 0}
        return {"output": "", "returncode": 0}


def _env(base_env, base_commit=BASE):
    return GenericGitEnvironment(
        base_env,
        {"instance_id": "reset-under-test", "git_url": "", "base_commit": base_commit},
    )


def test_setup_creates_workdir_when_clone_is_skipped():
    base_env = _FakeEnvironment()
    env = GenericGitEnvironment(
        base_env,
        {"instance_id": "empty-workdir", "git_url": "", "base_commit": ""},
    )

    env.setup_environment()

    assert base_env.commands[0] == ("mkdir -p /testbed", {"cwd": "/"})
    assert base_env.commands[1][0] == "git config --global --add safe.directory /testbed"


def test_reset_success_verifies_head_and_does_not_fetch():
    base_env = _ResetEnvironment()
    _env(base_env).setup_environment()

    cmds = [c for c, _ in base_env.commands]
    assert cmds.count(f"git reset --hard {BASE}") == 1
    assert "git rev-parse HEAD" in cmds
    assert not any(c.startswith("git fetch") for c in cmds)


def test_reset_fetches_and_retries_when_base_commit_is_missing():
    """PRs merged after the image was built: fetch the commit, retry the reset."""
    base_env = _ResetEnvironment(fail_resets=1)
    _env(base_env).setup_environment()

    cmds = [c for c, _ in base_env.commands]
    assert cmds.count(f"git reset --hard {BASE}") == 2
    assert f"git fetch origin {BASE}" in cmds


def test_reset_raises_when_head_does_not_land_on_base():
    base_env = _ResetEnvironment(head="f" * 40)
    with pytest.raises(RuntimeError, match=f"git reset --hard {BASE}"):
        _env(base_env).setup_environment()


def test_reset_raises_when_retry_also_fails():
    base_env = _ResetEnvironment(fail_resets=2, head="f" * 40)
    with pytest.raises(RuntimeError, match="rc=128"):
        _env(base_env).setup_environment()


def test_setup_scrubs_leak_paths_and_strips_git_history():
    """Image-baked repos: gold/verifier leftovers go, git leak prevention runs after the reset."""
    base_env = _ResetEnvironment()
    _env(base_env).setup_environment()

    cmds = [c for c, _ in base_env.commands]
    rm = next(c for c in cmds if c.startswith("rm -rf ") and ".build_env" in c)
    assert rm.startswith("rm -rf /testbed/.build_env")
    for leak in ("/tmp/patch.diff", "/tmp/test_patch.diff", "/tmp/test_files.json"):
        assert leak in rm
    assert "/tests" in rm.split()
    assert "/logs" in rm.split()
    strip_idx = next(i for i, c in enumerate(cmds) if "git checkout --detach" in c)
    assert strip_idx > cmds.index(f"git reset --hard {BASE}")
    assert strip_idx > cmds.index(rm)


def test_setup_without_base_commit_skips_git_leak_prevention():
    base_env = _FakeEnvironment()
    GenericGitEnvironment(base_env, {"instance_id": "no-base", "git_url": "", "base_commit": ""}).setup_environment()

    cmds = [c for c, _ in base_env.commands]
    assert not any("git checkout --detach" in c for c in cmds)
    assert any(c.startswith("rm -rf ") and ".build_env" in c for c in cmds)


def test_generic_has_no_programmatic_verifier():
    assert GenericGitEnvironment.HAS_VERIFIER is False
    with pytest.raises(NotImplementedError, match="rubric"):
        _env(_FakeEnvironment())._do_calculate_reward()
