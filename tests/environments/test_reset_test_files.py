"""Unit tests for DatasetEnvironment.build_reset_test_files_cmd — the per-file
test-reset that fixes the gold-test-patch ``already exists in working directory``
collision (agent plants a file at a path the gold test_patch newly adds)."""

from mimoagent.environments.datasets.base import DatasetEnvironment


class _Dummy(DatasetEnvironment):
    """Concrete subclass so we can build an instance (via object.__new__, skipping
    the env-dependent __init__) and exercise the pure command-builder logic."""

    def _setup_dataset_specific(self) -> None:  # pragma: no cover - unused
        ...

    def _do_calculate_reward(self, timeout=None, model_patch=""):  # pragma: no cover - unused
        return (0.0, "", {})


def _env() -> _Dummy:
    return object.__new__(_Dummy)


# one MODIFIED existing test file + one NEW (gold-added) test file
PATCH = (
    "diff --git a/pkg/handlers/metrics_test.go b/pkg/handlers/metrics_test.go\n"
    "index aaa..bbb 100644\n"
    "--- a/pkg/handlers/metrics_test.go\n"
    "+++ b/pkg/handlers/metrics_test.go\n"
    "@@ -1 +1 @@\n-old\n+new\n"
    "diff --git a/tests/new_suite_test.go b/tests/new_suite_test.go\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/tests/new_suite_test.go\n"
    "@@ -0,0 +1 @@\n+package x\n"
)


def test_builds_cmd_for_both_modified_and_new_files():
    cmd = _env().build_reset_test_files_cmd(PATCH, "BASE123")
    assert cmd is not None
    # both b-side paths land in the heredoc payload (incl. the gold-NEW file)
    assert "pkg/handlers/metrics_test.go" in cmd
    assert "tests/new_suite_test.go" in cmd
    # per-file branch: exists-in-base -> checkout; absent -> unstage + rm
    assert 'git cat-file -e BASE123:"$tf"' in cmd
    assert 'git checkout BASE123 -- "$tf"' in cmd
    assert 'git rm -f --cached "$tf"' in cmd
    assert 'rm -f "$tf"' in cmd


def test_returns_none_without_base_commit():
    assert _env().build_reset_test_files_cmd(PATCH, "") is None


def test_returns_none_with_empty_patch():
    assert _env().build_reset_test_files_cmd("", "BASE123") is None


def test_new_files_are_included_unlike_modified_only_helper():
    """get_patch_modified_files (old behavior) misses the gold-NEW file; the
    builder must use get_patch_touched_files so the collision-causing new path
    is reset/removed."""
    modified_only = DatasetEnvironment.get_patch_modified_files(PATCH)
    touched = DatasetEnvironment.get_patch_touched_files(PATCH)
    assert "tests/new_suite_test.go" not in modified_only  # the bug
    assert "tests/new_suite_test.go" in touched  # the fix


# a pure rename: a-side and b-side differ, and BOTH must be reset — restoring
# only the b-side leaves the rename source deleted, so re-applying the patch
# fails with "<old>: No such file or directory" (seen on deepswe-v1.1
# dynamodb-toolbox-lazy-recursive-schemas recalc).
RENAME_PATCH = (
    "diff --git a/docs/17-actions/1-parse.md b/docs/18-actions/1-parse.md\n"
    "similarity index 100%\n"
    "rename from docs/17-actions/1-parse.md\n"
    "rename to docs/18-actions/1-parse.md\n"
)


def test_rename_includes_both_sides():
    touched = DatasetEnvironment.get_patch_touched_files(RENAME_PATCH)
    assert "docs/17-actions/1-parse.md" in touched  # rename source (a-side)
    assert "docs/18-actions/1-parse.md" in touched  # rename target (b-side)
    cmd = _env().build_reset_test_files_cmd(RENAME_PATCH, "BASE123")
    assert "docs/17-actions/1-parse.md" in cmd
    assert "docs/18-actions/1-parse.md" in cmd
