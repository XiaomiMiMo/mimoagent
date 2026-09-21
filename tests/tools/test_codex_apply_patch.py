from __future__ import annotations

import stat

import pytest

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.codex.apply_patch import ApplyPatchTool


def _apply(tmp_path, patch: str, *, env=None):
    environment = env or LocalEnvironment(cwd=str(tmp_path))
    return ApplyPatchTool().execute({"input": patch}, {"env": environment})


def test_patch_requires_strict_begin_and_end_boundaries(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("before\n")

    missing_begin = _apply(
        tmp_path,
        "*** Update File: target.txt\n@@\n-before\n+after\n*** End Patch",
    )
    missing_end = _apply(
        tmp_path,
        "*** Begin Patch\n*** Update File: target.txt\n@@\n-before\n+after",
    )
    trailing_content = _apply(
        tmp_path,
        "*** Begin Patch\n*** Delete File: target.txt\n*** End Patch\nextra",
    )

    assert not missing_begin.success
    assert "first line" in missing_begin.output
    assert not missing_end.success
    assert "last line" in missing_end.output
    assert not trailing_content.success
    assert "last line" in trailing_content.output
    assert target.read_text() == "before\n"


def test_all_operations_are_validated_before_any_write(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("before\n")
    patch = """*** Begin Patch
*** Add File: created.txt
+created
*** Update File: target.txt
@@
-missing
+after
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert not result.success
    assert "Failed to find expected lines" in result.output
    assert not (tmp_path / "created.txt").exists()
    assert target.read_text() == "before\n"


def test_repeated_context_uses_first_match_and_locator_disambiguates(tmp_path):
    target = tmp_path / "module.py"
    original = "def first():\n    return 1\n\ndef second():\n    return 1\n"
    target.write_text(original)
    first_match = """*** Begin Patch
*** Update File: module.py
@@
-    return 1
+    return 2
*** End Patch"""

    succeeded = _apply(tmp_path, first_match)

    assert succeeded.success
    assert target.read_text() == "def first():\n    return 2\n\ndef second():\n    return 1\n"

    target.write_text(original)

    located = """*** Begin Patch
*** Update File: module.py
@@ def second():
-    return 1
+    return 2
*** End Patch"""
    succeeded = _apply(tmp_path, located)

    assert succeeded.success
    assert target.read_text() == "def first():\n    return 1\n\ndef second():\n    return 2\n"


def test_context_free_addition_without_eof_marker_appends_at_eof(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("first\nsecond\n")
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
+third
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text() == "first\nsecond\nthird\n"


def test_operations_observe_prior_operations_in_the_same_patch(tmp_path):
    patch = """*** Begin Patch
*** Add File: new/file.txt
+original
*** Update File: new/file.txt
*** Move to: moved/file.txt
@@
-original
+changed
*** Add File: remove.txt
+remove
*** Delete File: remove.txt
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert (tmp_path / "moved/file.txt").read_text() == "changed\n"
    assert not (tmp_path / "new/file.txt").exists()
    assert not (tmp_path / "remove.txt").exists()


def test_add_and_move_can_overwrite_regular_files(tmp_path):
    added = tmp_path / "added.txt"
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    added.write_text("old add\n")
    source.write_text("source\n")
    destination.write_text("old destination\n")
    added.chmod(0o754)
    destination.chmod(0o640)
    patch = """*** Begin Patch
*** Add File: added.txt
+new add
*** Update File: source.txt
*** Move to: destination.txt
@@
-source
+moved
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert added.read_text() == "new add\n"
    assert stat.S_IMODE(added.stat().st_mode) == 0o754
    assert not source.exists()
    assert destination.read_text() == "moved\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_update_preserves_executable_mode(tmp_path):
    target = tmp_path / "run.sh"
    target.write_text("echo before\n")
    target.chmod(0o751)
    patch = """*** Begin Patch
*** Update File: run.sh
@@
-echo before
+echo after
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text() == "echo after\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o751


def test_stacked_locators_narrow_nested_context(tmp_path):
    target = tmp_path / "module.py"
    target.write_text(
        "class A:\n    def method(self):\n        return 1\n\nclass B:\n    def method(self):\n        return 1\n"
    )
    patch = """*** Begin Patch
*** Update File: module.py
@@ class B:
@@     def method(self):
-        return 1
+        return 2
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text().endswith("class B:\n    def method(self):\n        return 2\n")


def test_end_of_file_and_bare_blank_context_follow_v4a_semantics(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("a\nb\n\nc\n")
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
 a
-b
+B
@@

-c
+C
@@
+trailer
*** End of File
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text() == "a\nB\n\nC\ntrailer\n"


def test_end_of_file_accepts_trailing_newline_context_sentinel(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("line\n")
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
-line
+LINE

*** End of File
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text() == "LINE\n"


def test_move_overwrite_is_rolled_back_when_later_validation_fails(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("source\n")
    target.write_text("target\n")
    patch = """*** Begin Patch
*** Update File: source.txt
*** Move to: target.txt
@@
-source
+updated
*** Update File: target.txt
@@
-missing
+unexpected
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert not result.success
    assert "Failed to find expected lines" in result.output
    assert source.read_text() == "source\n"
    assert target.read_text() == "target\n"


def test_commit_failure_rolls_back_already_replaced_files(tmp_path):
    class FailSecondReplaceEnvironment(LocalEnvironment):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacements = 0

        def execute(self, command: str, cwd: str = "", timeout: int = None):
            if command.startswith("mv -f --"):
                self.replacements += 1
                if self.replacements == 2:
                    return {"output": "injected failure", "returncode": 1}
            return super().execute(command, cwd=cwd, timeout=timeout)

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n")
    second.write_text("two\n")
    first.chmod(0o751)
    second.chmod(0o640)
    env = FailSecondReplaceEnvironment(cwd=str(tmp_path))
    patch = """*** Begin Patch
*** Update File: first.txt
@@
-one
+ONE
*** Update File: second.txt
@@
-two
+TWO
*** End Patch"""

    result = _apply(tmp_path, patch, env=env)

    assert not result.success
    assert "injected failure" in result.output
    assert first.read_text() == "one\n"
    assert second.read_text() == "two\n"
    assert stat.S_IMODE(first.stat().st_mode) == 0o751
    assert stat.S_IMODE(second.stat().st_mode) == 0o640
    assert not list(tmp_path.glob("*.mimo-apply-patch-*"))


# --- A-1: an *** End of File hunk anchors at EOF. If it does not match there,
# it must fail rather than silently relocate to the first match elsewhere.


def test_end_of_file_hunk_that_misses_eof_is_rejected_not_relocated(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    # `alpha` matches only at the top of the file, but the hunk is anchored to
    # EOF, so it must not be applied there.
    patch = """*** Begin Patch
*** Update File: f.txt
@@
-alpha
+ALPHA
*** End of File
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert not result.success
    assert "Failed to find expected lines" in result.output
    assert target.read_text() == "alpha\nbeta\ngamma\n"


def test_end_of_file_hunk_still_applies_when_it_matches_at_eof(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    patch = """*** Begin Patch
*** Update File: f.txt
@@
-gamma
+GAMMA
*** End of File
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    assert target.read_text() == "alpha\nbeta\nGAMMA\n"


# --- A-4: a non-UTF-8 file must be refused, not silently rewritten with U+FFFD.


class _CopyOutEnvironment:
    """kubernetes-shaped: copy_out moves bytes verbatim; execute is lossy."""

    def __init__(self, tmp_path):
        self.config = type("_Config", (), {"cwd": str(tmp_path)})()

    def copy_out(self, remote, local):
        import shutil

        shutil.copyfile(remote, local)

    def copy_to(self, local, remote):
        import shutil

        shutil.copyfile(local, remote)

    def execute(self, command, cwd="", timeout=None):
        import subprocess

        proc = subprocess.run(["bash", "-lc", command], capture_output=True)
        return {
            "output": (proc.stdout + proc.stderr).decode("utf-8", "replace"),
            "returncode": proc.returncode,
            "reason": "ok",
        }


class _NoCopyOutEnvironment:
    """local/docker-shaped: no copy_out, so the base64 read path is exercised."""

    def __init__(self, tmp_path):
        self.config = type("_Config", (), {"cwd": str(tmp_path)})()

    def copy_to(self, local, remote):
        import shutil

        shutil.copyfile(local, remote)

    def execute(self, command, cwd="", timeout=None):
        import subprocess

        proc = subprocess.run(["bash", "-lc", command], capture_output=True)
        return {
            "output": (proc.stdout + proc.stderr).decode("utf-8", "replace"),
            "returncode": proc.returncode,
            "reason": "ok",
        }


@pytest.mark.parametrize("env_cls", [_CopyOutEnvironment, _NoCopyOutEnvironment])
def test_non_utf8_file_is_refused_and_left_untouched(tmp_path, env_cls):
    target = tmp_path / "legacy.py"
    original = "value = 1\ncaf\xe9\n".encode("latin-1")  # 0xE9 is not valid UTF-8
    target.write_bytes(original)
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-value = 1
+value = 2
*** End Patch"""

    result = ApplyPatchTool().execute({"input": patch}, {"env": env_cls(tmp_path)})

    assert not result.success
    assert "non-UTF-8" in result.output
    # The file the patch never reached must be byte-for-byte unchanged.
    assert target.read_bytes() == original


@pytest.mark.parametrize("env_cls", [_CopyOutEnvironment, _NoCopyOutEnvironment])
def test_utf8_file_still_patches_through_both_read_paths(tmp_path, env_cls):
    target = tmp_path / "ok.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-value = 1
+value = 2
*** End Patch"""

    result = ApplyPatchTool().execute({"input": patch}, {"env": env_cls(tmp_path)})

    assert result.success
    assert target.read_text() == "value = 2\n"


# --- The pod's shell is a login shell (`bash -lc`), so any /etc/profile banner
# lands on stdout ahead of a probe's own output. Parsing the whole stdout made
# apply_patch fail 100% of the time on such an image.

_BANNER = "Welcome to the testbed image!\nLast login: never\n"


class _BannerCopyOutEnvironment(_CopyOutEnvironment):
    """kubernetes-shaped, plus a profile banner on every command."""

    def execute(self, command, cwd="", timeout=None):
        result = super().execute(command, cwd=cwd, timeout=timeout)
        return {**result, "output": _BANNER + result["output"]}


class _BannerNoCopyOutEnvironment(_NoCopyOutEnvironment):
    """local/docker-shaped with a banner, so the base64 read path sees it too."""

    def execute(self, command, cwd="", timeout=None):
        result = super().execute(command, cwd=cwd, timeout=timeout)
        return {**result, "output": _BANNER + result["output"]}


@pytest.mark.parametrize("env_cls", [_BannerCopyOutEnvironment, _BannerNoCopyOutEnvironment])
def test_a_login_shell_banner_does_not_break_the_probes(tmp_path, env_cls):
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-value = 1
+value = 2
*** End Patch"""

    result = ApplyPatchTool().execute({"input": patch}, {"env": env_cls(tmp_path)})

    assert result.success, result.output
    assert target.read_text() == "value = 2\n"


def test_a_banner_does_not_hide_a_missing_update_target(tmp_path):
    """The state probe must still distinguish states, not just tolerate noise."""

    patch = """*** Begin Patch
*** Update File: absent.py
@@
-value = 1
+value = 2
*** End Patch"""

    result = ApplyPatchTool().execute({"input": patch}, {"env": _BannerCopyOutEnvironment(tmp_path)})

    assert not result.success
    assert "not a regular file" in result.output


def test_a_banner_does_not_disturb_the_preserved_file_mode(tmp_path):
    target = tmp_path / "run.sh"
    target.write_text("echo one\n", encoding="utf-8")
    target.chmod(0o755)
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-echo one
+echo two
*** End Patch"""

    result = ApplyPatchTool().execute({"input": patch}, {"env": _BannerCopyOutEnvironment(tmp_path)})

    assert result.success, result.output
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


# --- The failure text has to name the file: 165 of 337 observed misses were on
# multi-file patches, where "hunk not found" alone does not say which file.


def test_a_hunk_miss_names_the_file_and_the_expected_lines(tmp_path):
    first = tmp_path / "first.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second = tmp_path / "second.py"
    second.write_text("other = 2\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {first}
@@
-value = 1
+value = 3
*** Update File: {second}
@@
-not_here = 9
+other = 4
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert not result.success
    assert str(second) in result.output
    assert "not_here = 9" in result.output
    # The file that did match must not be reported as the failure.
    assert str(first) not in result.output
    assert second.read_text() == "other = 2\n"


def test_a_locator_miss_names_the_file(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {target}
@@ def absent_function():
-value = 1
+value = 2
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert not result.success
    assert str(target) in result.output
    assert "absent_function" in result.output


def test_a_successful_patch_reports_the_upstream_header_and_file_lines(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {target}
@@
-value = 1
+value = 2
*** Add File: {tmp_path}/created.py
+created = True
*** End Patch"""

    result = _apply(tmp_path, patch)

    assert result.success
    lines = result.output.splitlines()
    assert lines[0] == "Success. Updated the following files:"
    assert f"M {target}" in lines
    assert f"A {tmp_path}/created.py" in lines
