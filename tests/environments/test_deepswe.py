"""Unit tests for DeepSWEEnvironment — v1.0/v1.1 flow selection, pier-style
model.patch capture, fresh-eval-pod grading, and the v1.1 reward.json /
crash-sentinel parsing."""

import json

import pytest

from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED
from mimoagent.environments.datasets.deepswe import DeepSWEEnvironment, parse_v11_reward


class _FakeEnv:
    """Minimal base-env stand-in recording executed commands.

    ``get_template_vars``/``start``/``cleanup``/``copy_out`` mimic the k8s env
    surface the eval-pod flow needs. ``clones`` collects every sibling env
    created via ``type(env)(**template_vars)`` so tests can inspect what ran
    in the eval pod vs the agent pod.
    """

    def __init__(self, outputs: dict[str, dict] | None = None, **kwargs):
        self.commands: list[str] = []
        self.copied: list[str] = []
        self.copied_out: list[str] = []
        self._outputs = outputs or {}
        self._template_vars = {"outputs": self._outputs}
        self._template_vars.update(kwargs)
        self.started = False
        self.cleaned_up = False
        self.clones: list[_FakeEnv] = []
        self.pod_name = "fake-pod"
        # file contents copy_out writes locally, keyed by remote path
        self.files: dict[str, bytes] = {}

    def execute(self, command: str, **kwargs) -> dict:
        self.commands.append(command)
        for needle, res in self._outputs.items():
            if needle in command:
                return res
        if command.strip() == "git rev-parse HEAD":
            # git_reset_hard verifies HEAD landed on base_commit ("abc123")
            return {"returncode": 0, "output": "abc123" + "0" * 34}
        return {"returncode": 0, "output": ""}

    def copy_to(self, local_path: str, remote_path: str) -> None:
        self.copied.append(remote_path)

    def copy_out(self, remote_path: str, local_path: str) -> None:
        self.copied_out.append(remote_path)
        with open(local_path, "wb") as f:
            f.write(self.files.get(remote_path, b""))

    def get_template_vars(self) -> dict:
        return dict(self._template_vars)

    def start(self) -> None:
        self.started = True

    def cleanup(self) -> None:
        self.cleaned_up = True


class _CloneTrackingEnv(_FakeEnv):
    """FakeEnv whose type() clones register themselves on the root env."""

    _root: "_CloneTrackingEnv | None" = None

    def __init__(self, outputs=None, **kwargs):
        super().__init__(outputs, **kwargs)
        if _CloneTrackingEnv._root is not None:
            _CloneTrackingEnv._root.clones.append(self)


@pytest.fixture
def clone_root():
    """A _CloneTrackingEnv that records eval-env clones in ``.clones``."""
    _CloneTrackingEnv._root = None
    root = _CloneTrackingEnv()
    _CloneTrackingEnv._root = root
    yield root
    _CloneTrackingEnv._root = None


V10_INSTANCE = {
    "instance_id": "t-v10",
    "dataset_type": "deepswe",
    "base_commit": "abc123",
    "docker_image": "registry.example.com/swe-bench-202605:x",
    "test_sh": "#!/bin/bash\necho hi",
    "test_patch": "diff --git a/f b/f\n",
}

V11_INSTANCE = {
    **V10_INSTANCE,
    "instance_id": "t-v11",
    "schema_version": "1.1",
    "docker_image": "registry.example.com/swe-bench-202605:x-v1.1",
    "grader_py": "# grader",
    "config_json": '{"base_commit": "abc123"}',
}


# --- construction / flow selection -----------------------------------------


def test_v10_instance_keeps_default_leak_prevention():
    env = DeepSWEEnvironment(_FakeEnv(), dict(V10_INSTANCE))
    assert env._is_v11 is False
    assert env.git_leak_prevention == "strip"


def test_v11_instance_detected_and_leak_prevention_off():
    env = DeepSWEEnvironment(_FakeEnv(), dict(V11_INSTANCE))
    assert env._is_v11 is True
    # image history is pre-stripped at build time; strip/hide would break the
    # instructed branch+commit workflow
    assert env.git_leak_prevention == "none"


def test_declared_v11_without_grader_raises():
    bad = {**V10_INSTANCE, "schema_version": "1.1"}
    with pytest.raises(ValueError, match="grader_py"):
        DeepSWEEnvironment(_FakeEnv(), bad)


def test_without_base_commit_raises():
    for base in (V10_INSTANCE, V11_INSTANCE):
        bad = {**base, "base_commit": ""}
        with pytest.raises(ValueError, match="base_commit"):
            DeepSWEEnvironment(_FakeEnv(), bad)


def test_missing_test_sh_raises():
    bad = {**V10_INSTANCE, "test_sh": ""}
    with pytest.raises(ValueError, match="test_sh"):
        DeepSWEEnvironment(_FakeEnv(), bad)


# --- setup -----------------------------------------------------------------


def test_v10_setup_resets_to_base():
    fake = _FakeEnv()
    env = DeepSWEEnvironment(fake, dict(V10_INSTANCE))
    env.git_leak_prevention = "none"  # keep the fake simple
    env._setup_dataset_specific()
    assert any("git reset --hard abc123" in c for c in fake.commands)


def test_v11_setup_does_not_reset():
    fake = _FakeEnv()
    env = DeepSWEEnvironment(fake, dict(V11_INSTANCE))
    env._setup_dataset_specific()
    assert not any("git reset --hard" in c for c in fake.commands)


# --- capture (pier-style, both generations) ----------------------------------


@pytest.mark.parametrize("instance", [V10_INSTANCE, V11_INSTANCE], ids=["v10", "v11"])
def test_capture_writes_pod_side_patch_and_pulls_bytes(instance):
    fake = _FakeEnv()
    fake.files["/logs/artifacts/model.patch"] = b"DIFF"
    env = DeepSWEEnvironment(fake, dict(instance))
    diff, err = env._capture_model_diff()
    assert (diff, err) == ("DIFF", "")
    assert env._model_patch_bytes == b"DIFF"
    cmd = next(c for c in fake.commands if "git reset --soft" in c)
    assert "git reset --soft abc123" in cmd
    assert "--binary abc123 > /logs/artifacts/model.patch" in cmd
    assert fake.copied_out == ["/logs/artifacts/model.patch"]


def test_capture_failure_leaves_no_patch_bytes():
    fake = _FakeEnv(outputs={"git reset --soft": {"returncode": 1, "output": "boom"}})
    env = DeepSWEEnvironment(fake, dict(V11_INSTANCE))
    diff, err = env._capture_model_diff()
    assert diff == ""
    assert err
    assert env._model_patch_bytes is None


def test_pull_file_bytes_base64_fallback_without_copy_out():
    import base64

    class _NoCopyOutEnv(_FakeEnv):
        copy_out = property()  # hasattr(env, "copy_out") -> raises -> False

    payload = b"\x00\x01binary"
    fake = _NoCopyOutEnv(outputs={"base64": {"returncode": 0, "output": base64.b64encode(payload).decode() + "\n"}})
    assert not hasattr(fake, "copy_out")
    got = DeepSWEEnvironment._pull_file_bytes(fake, "/logs/artifacts/model.patch")
    assert got == payload


# --- reward: fresh eval env ----------------------------------------------------


def _grading_outputs(reward_json: str = "", reward_txt: str = "") -> dict:
    return {
        "cat /logs/verifier/reward.json": {"returncode": 0, "output": reward_json},
        "cat /logs/verifier/reward.txt": {"returncode": 0, "output": reward_txt},
        "bash /tests/test.sh": {"returncode": 0, "output": "suite output"},
    }


def _run_reward(clone_root, instance, patch_bytes=b"diff --git a/f b/f\n", **outputs):
    clone_root._outputs.update(outputs)
    clone_root._template_vars["outputs"] = clone_root._outputs
    env = DeepSWEEnvironment(clone_root, dict(instance))
    env._model_patch_bytes = patch_bytes
    return env, env._do_calculate_reward(model_patch=patch_bytes.decode())


def test_reward_runs_in_fresh_eval_env(clone_root):
    _, (reward, output, extra) = _run_reward(
        clone_root, V11_INSTANCE, **_grading_outputs(reward_json=json.dumps({"reward": 1}))
    )
    assert reward == 1.0
    assert extra["resolved"] is True
    assert output == "suite output"
    assert extra["eval_pod_name"] == "fake-pod"
    # exactly one eval env: created, started, graded in, cleaned up
    assert len(clone_root.clones) == 1
    ev = clone_root.clones[0]
    assert ev.started and ev.cleaned_up
    assert any("bash /tests/test.sh" in c for c in ev.commands)
    # the agent env only did the capture-side work; the verifier never ran there
    assert not any("bash /tests/test.sh" in c for c in clone_root.commands)
    # verifier files + model.patch landed in the EVAL env
    assert ev.copied == [
        "/tests/test.sh",
        "/tests/test.patch",
        "/tests/grader.py",
        "/tests/config.json",
        "/logs/artifacts/model.patch",
    ]
    assert clone_root.copied == []


def test_eval_env_gets_role_label(clone_root):
    clone_root._template_vars["labels"] = {"app": "mimoagent"}
    _run_reward(clone_root, V11_INSTANCE, **_grading_outputs(reward_json=json.dumps({"reward": 1})))
    ev = clone_root.clones[0]
    assert ev._template_vars["labels"] == {"app": "mimoagent", "mimoagent-role": "deepswe-eval"}


def test_v11_eval_env_not_reset_to_base(clone_root):
    _run_reward(clone_root, V11_INSTANCE, **_grading_outputs(reward_json=json.dumps({"reward": 1})))
    ev = clone_root.clones[0]
    # grader.py prepare owns per-file preimage alignment; a repo-wide reset
    # would destroy deliberate image-build modifications
    assert not any("git reset --hard" in c for c in ev.commands)
    assert not any("git apply" in c for c in ev.commands)


def test_v10_eval_env_reset_and_apply_before_verifier(clone_root):
    _, (reward, _, extra) = _run_reward(clone_root, V10_INSTANCE, **_grading_outputs(reward_txt="1\n"))
    assert reward == 1.0
    assert extra["resolved"] is True
    ev = clone_root.clones[0]
    reset_i = next(i for i, c in enumerate(ev.commands) if "git reset --hard abc123" in c)
    apply_i = next(i for i, c in enumerate(ev.commands) if "git apply" in c)
    run_i = next(i for i, c in enumerate(ev.commands) if "bash /tests/test.sh" in c)
    assert reset_i < apply_i < run_i
    assert "/logs/artifacts/model.patch" in ev.commands[apply_i]


def test_v10_empty_patch_skips_apply(clone_root):
    _, (reward, _, _) = _run_reward(clone_root, V10_INSTANCE, patch_bytes=b"", **_grading_outputs(reward_txt="0\n"))
    assert reward == 0.0
    ev = clone_root.clones[0]
    assert not any("git apply" in c for c in ev.commands)
    # the (empty) patch file is still materialised for the verifier
    assert "/logs/artifacts/model.patch" in ev.copied


def test_v10_apply_failure_is_infra_error(clone_root):
    _, (reward, _, extra) = _run_reward(
        clone_root,
        V10_INSTANCE,
        **{"git apply": {"returncode": 1, "output": "corrupt patch"}},
        **_grading_outputs(reward_txt="0\n"),
    )
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "model_patch_apply_failed"
    ev = clone_root.clones[0]
    assert not any("bash /tests/test.sh" in c for c in ev.commands)
    assert ev.cleaned_up


def test_missing_capture_is_infra_error():
    env = DeepSWEEnvironment(_FakeEnv(), dict(V11_INSTANCE))
    assert env._model_patch_bytes is None
    reward, _, extra = env._do_calculate_reward()
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "model_patch_capture_failed"


def test_eval_env_start_failure_is_infra_error(clone_root):
    env = DeepSWEEnvironment(clone_root, dict(V11_INSTANCE))
    env._model_patch_bytes = b"diff"

    def _raise():
        raise RuntimeError("no pods for you")

    env._make_eval_env = _raise
    reward, _, extra = env._do_calculate_reward()
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "eval_env_start_failed"


def test_eval_env_grading_error_is_infra_error_and_cleans_up(clone_root):
    env = DeepSWEEnvironment(clone_root, dict(V11_INSTANCE))
    env._model_patch_bytes = b"diff"

    def _explode(eval_env, exec_timeout):
        raise RuntimeError("grading exploded")

    env._grade_in_eval_env = _explode
    reward, _, extra = env._do_calculate_reward()
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "eval_env_error"
    assert clone_root.clones[0].cleaned_up


def test_v11_reward_fail(clone_root):
    _, (reward, _, extra) = _run_reward(
        clone_root, V11_INSTANCE, **_grading_outputs(reward_json=json.dumps({"reward": 0, "f2p": 0.5}))
    )
    assert reward == 0.0
    assert extra["resolved"] is False


def test_v11_crash_sentinel_maps_to_infra_error(clone_root):
    _, (reward, _, extra) = _run_reward(clone_root, V11_INSTANCE, **_grading_outputs(reward_txt="-1\n"))
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED


def test_v10_missing_reward_file_is_fail(clone_root):
    _, (reward, _, extra) = _run_reward(
        clone_root,
        V10_INSTANCE,
        **{
            "bash /tests/test.sh": {"returncode": 0, "output": "ok"},
            "cat /logs/verifier/reward.txt": {"returncode": 0, "output": "MISSING"},
        },
    )
    assert reward == 0.0
    assert extra["resolved"] is False


# --- parse_v11_reward (pure) --------------------------------------------------


@pytest.mark.parametrize(
    ("reward_json", "reward_txt", "expected_reward"),
    [
        ('{"reward": 1}', "", 1.0),
        ('{"reward": 0}', "", 0.0),
        ('{"reward": 0, "apply_failed": 1}', "", 0.0),
        ("", "0", 0.0),
        ("", "", 0.0),
        ("not json", "", 0.0),
    ],
)
def test_parse_v11_reward_values(reward_json, reward_txt, expected_reward):
    reward, _ = parse_v11_reward(reward_json, reward_txt)
    assert reward == expected_reward


def test_parse_v11_reward_apply_failed_flag():
    _, extra = parse_v11_reward('{"reward": 0, "apply_failed": 1}', "")
    assert extra["apply_failed"] is True


def test_parse_v11_reward_sentinel_only_without_json():
    # a valid reward.json wins over a stale sentinel
    reward, extra = parse_v11_reward('{"reward": 1}', "-1")
    assert reward == 1.0 and "error_category" not in extra
    reward, extra = parse_v11_reward("", "-1")
    assert reward == 0.0 and extra["error_category"] == REWARD_TESTBED_CORRUPTED
