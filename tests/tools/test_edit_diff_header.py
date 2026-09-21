"""The Edit observation must not carry per-call random paths.

The engine diffs the target against a scratch copy under
``/tmp/mimo_edit_<hex>/``; naming that copy in the diff header put a fresh
random path into 97% of cc trajectories. The header must name the target on
both sides.
"""

from mimoagent.environments.local import LocalEnvironment
from mimoagent.tools.cc.edit import EditTool as CCEditTool
from mimoagent.tools.edit import EditTool as DefaultEditTool


def test_cc_edit_diff_names_the_target_not_the_scratch_copy(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    result = CCEditTool().execute(
        {"file_path": str(target), "old_string": "b = 2", "new_string": "b = 3"},
        {"env": LocalEnvironment(cwd=str(tmp_path))},
    )

    assert result.success, result.output
    assert target.read_text(encoding="utf-8") == "a = 1\nb = 3\n"
    assert "mimo_edit_" not in result.output
    assert "/out" not in result.output
    assert f"--- a{target}" in result.output
    assert f"+++ b{target}" in result.output
    assert "-b = 2" in result.output and "+b = 3" in result.output


def test_cc_edit_diff_does_not_report_a_mode_change_for_an_executable_target(tmp_path):
    """Both diff sides now name the target, so a mode line would claim the
    edit changed the file's mode. It does not (cat > target keeps the inode);
    the 0644 scratch copy just has to match. Only the executable bit matters:
    git canonicalises every non-executable mode to 100644, so a 0600 target
    never showed `old mode`/`new mode` — a 0755 script did."""
    target = tmp_path / "run.sh"
    target.write_text("#!/bin/sh\necho one\n", encoding="utf-8")
    target.chmod(0o755)

    result = CCEditTool().execute(
        {"file_path": str(target), "old_string": "echo one", "new_string": "echo two"},
        {"env": LocalEnvironment(cwd=str(tmp_path))},
    )

    assert result.success, result.output
    assert "old mode" not in result.output and "new mode" not in result.output
    assert "+echo two" in result.output
    assert target.stat().st_mode & 0o777 == 0o755


def test_default_edit_diff_names_the_target_not_the_scratch_copy(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    result = DefaultEditTool().execute(
        {"path": str(target), "old_str": "b = 2", "new_str": "b = 3"},
        {"env": LocalEnvironment(cwd=str(tmp_path))},
    )

    assert result.success, result.output
    assert "mimo_edit_" not in result.output
    assert f"+++ b{target}" in result.output
