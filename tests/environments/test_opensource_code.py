import subprocess

import pytest

from mimoagent.environments.datasets.opensource_code import OpenSourceCodeEnvironment
from mimoagent.environments.local import LocalEnvironment

_TEST_PATCH = """diff --git a/mimo_test_command.sh b/mimo_test_command.sh
new file mode 100644
--- /dev/null
+++ b/mimo_test_command.sh
@@ -0,0 +1,2 @@
+#!/usr/bin/env bash
+grep -q fixed src.py
"""


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "t")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "t@example.com")


def _git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _readable(repo, sha) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo, capture_output=True).returncode == 0


@pytest.fixture
def upstream(tmp_path):
    up = tmp_path / "upstream"
    up.mkdir()
    (up / "src.py").write_text("broken\n")
    _git(up, "init", "-q", "-b", "master")
    _git(up, "add", "-A")
    _git(up, "commit", "-qm", "base")
    _git(up, "tag", "v1")
    (up / "src.py").write_text("fixed\n")
    _git(up, "commit", "-qam", "golden")
    _git(up, "tag", "v2")
    return up, _git(up, "rev-parse", "HEAD~1"), _git(up, "rev-parse", "HEAD")


def _clone_reset(upstream, tmp_path):
    up, base, _ = upstream
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", "--no-local", str(up), str(repo))
    _git(repo, "reset", "-q", "--hard", base)
    return repo


def _shallow_graft(upstream, tmp_path):
    repo = _clone_reset(upstream, tmp_path)
    _git(repo, "checkout", "-q", "--detach")
    for ref in _git(repo, "for-each-ref", "--format=%(refname)").split():
        _git(repo, "update-ref", "-d", ref)
    (repo / ".git" / "shallow").write_text(_git(repo, "rev-parse", "HEAD") + "\n")
    return repo


def _synthetic_base(upstream, tmp_path):
    up, _, _ = upstream
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", "--no-local", str(up), str(repo))
    _git(repo, "checkout", "-q", "--orphan", "task")
    (repo / "src.py").write_text("broken\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "task base")
    _git(repo, "branch", "-q", "-D", "master")
    _git(repo, "branch", "-q", "-m", "master")
    return repo


def _dangling_remote_head(upstream, tmp_path):
    repo = _clone_reset(upstream, tmp_path)
    _git(repo, "config", "--remove-section", "remote.origin")
    _git(repo, "update-ref", "-d", "--no-deref", "refs/remotes/origin/master")
    assert (repo / ".git" / "refs" / "remotes" / "origin" / "HEAD").read_text() == "ref: refs/remotes/origin/master\n"
    return repo


_SHAPES = [_clone_reset, _shallow_graft, _synthetic_base, _dangling_remote_head]


def _env(repo, test_patch: str = _TEST_PATCH) -> OpenSourceCodeEnvironment:
    instance = {
        "dataset_type": "opensource-code",
        "docker_image": "unused",
        "cwd": str(repo),
        "instance_id": "t1",
        "problem_statement": "fix src.py",
        "test_patch": test_patch,
        "test_command": f"bash {repo}/mimo_test_command.sh",
        "verifier_timeout_sec": 60,
    }
    return OpenSourceCodeEnvironment(LocalEnvironment(cwd=str(repo), timeout=120), instance)


@pytest.mark.parametrize("shape", _SHAPES)
def test_setup_removes_every_commit_outside_the_base(shape, upstream, tmp_path):
    _, _, golden = upstream
    repo = shape(upstream, tmp_path)
    head, branch = _git(repo, "rev-parse", "HEAD"), _git(repo, "branch", "--show-current")
    assert _readable(repo, golden)

    _env(repo).setup_environment()

    assert not _readable(repo, golden)
    assert _git(repo, "remote") == ""
    assert subprocess.run(["git", "config", "--get-regexp", r"^remote\."], cwd=repo).returncode == 1
    assert not (repo / ".git" / "logs").exists()
    assert not (repo / ".git" / "ORIG_HEAD").exists()
    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "branch", "--show-current") == branch
    assert (repo / "src.py").read_text() == "broken\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_setup_keeps_ancestor_tags_and_drops_newer_ones(upstream, tmp_path):
    repo = _clone_reset(upstream, tmp_path)
    _env(repo).setup_environment()
    assert _git(repo, "tag").split() == ["v1"]


@pytest.mark.parametrize("shape", _SHAPES)
def test_reward_still_grades_after_the_purge(shape, upstream, tmp_path):
    repo = shape(upstream, tmp_path)
    env = _env(repo)
    env.setup_environment()

    reward, _, extra = env.calculate_reward()
    assert reward == 0.0 and "error_category" not in extra

    (repo / "src.py").write_text("fixed\n")
    reward, _, extra = env.calculate_reward()
    assert reward == 1.0 and "error_category" not in extra


def test_setup_fails_when_objects_outside_the_base_survive(upstream, tmp_path):
    up, base, _ = upstream
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", "--shared", str(up), str(repo))
    _git(repo, "reset", "-q", "--hard", base)

    with pytest.raises(RuntimeError, match="git history purge failed"):
        _env(repo).setup_environment()


def test_setup_deletes_a_remote_head_that_points_at_a_deleted_branch(upstream, tmp_path):
    repo = _dangling_remote_head(upstream, tmp_path)
    _env(repo).setup_environment()
    assert not (repo / ".git" / "refs" / "remotes").joinpath("origin", "HEAD").exists()
    assert subprocess.run(["git", "for-each-ref"], cwd=repo, capture_output=True, text=True).stderr == ""


@pytest.mark.parametrize("via_hooks_path", [False, True])
def test_reward_reset_does_not_run_hooks(via_hooks_path, upstream, tmp_path):
    repo = _clone_reset(upstream, tmp_path)
    env = _env(repo, test_patch="diff --git a/src.py b/src.py\n")
    env.setup_environment()
    hooks = tmp_path / "planted-hooks" if via_hooks_path else repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    marker = tmp_path / "hook-ran"
    (hooks / "post-checkout").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (hooks / "post-checkout").chmod(0o755)
    if via_hooks_path:
        _git(repo, "config", "core.hooksPath", str(hooks))
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "src.py").write_text("agent edit\n")
    _git(repo, "checkout", base, "--", "src.py")
    assert marker.exists(), "the planted hook must fire on a plain checkout, or this test proves nothing"
    marker.unlink()

    (repo / "src.py").write_text("agent edit\n")
    assert env._reset_test_files()["returncode"] == 0

    assert (repo / "src.py").read_text() == "broken\n"
    assert not marker.exists()
