"""Unit tests for the rubric judge reward stage.

Covers ``parse_rubrics``, reward-plan resolution (auto / programmatic / rubric /
both), judge-before-verifier ordering, scalar combination, fail-closed judge
failures, registry-time validation, and the judge internals ported from the
former rubric dataset (workspace assembly, single-shot verdict delivery with no
retry, verdict parsing/scoring, banner-robust read-back)."""

import json
import re
from unittest import mock

import pytest

from mimoagent.environments.datasets import DATASET_REGISTRY, create_from_registry
from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED, DatasetEnvironment
from mimoagent.environments.rubric_judge import RubricJudge, RubricJudgeConfig, parse_rubrics


class _FakePod:
    """Base-env stand-in: copy_to stores file content, cat serves it back."""

    config = None

    def __init__(self):
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self.copied_out: list[tuple[str, str]] = []

    def start(self) -> None:
        pass

    def execute(self, command: str, **kwargs) -> dict:
        self.commands.append(command)
        if command.strip() == "git rev-parse HEAD":
            # git_reset_hard verifies HEAD landed on base_commit ("abc123")
            return {"output": "abc123" + "0" * 34, "returncode": 0}
        markers = re.findall(r"__MIMO_VERDICTS_(?:BEGIN|END)_[0-9a-f]+__", command)
        if markers:
            cat_match = re.search(r"\bcat\s+([^;]+)", command)
            assert cat_match is not None
            path = cat_match.group(1).strip().strip("'\"")
            content = self.files.get(path)
            output = f"{markers[0]}\n{content or ''}\n{markers[1]}\n"
            return {"output": output, "returncode": 0 if content is not None else 1}
        return {"output": "", "returncode": 0}

    def copy_to(self, local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as f:
            raw = f.read()
        try:
            self.files[remote_path] = raw.decode()
        except UnicodeDecodeError:
            self.files[remote_path] = f"<binary {len(raw)} bytes>"

    def copy_out(self, remote_path: str, local_path: str) -> None:
        self.copied_out.append((remote_path, local_path))


RUBRICS = ["compiles cleanly", "still finger is stable", "no MIDI flooding", "vibrato not laggy"]

INSTANCE = {
    "instance_id": "t-rubric",
    "dataset_type": "generic",
    "docker_image": "registry.example.com/swe/dev:latest",
    "git_url": "https://example.com/repo.git",
    "base_commit": "abc123",
    "problem_statement": "fix the pitch wobble",
    "rubric": {"rubric_version": "1.0", "rubrics": RUBRICS},
}

JUDGE_CFG = {"model": {"model_name": "fake"}, "step_limit": 5}
WS = "/tmp/.mimo-rubric-judge"

# Ordering log shared by the fake verifier and the scripted judge.
CALLS: list[str] = []


class _VerifierEnv(DatasetEnvironment):
    """A test-graded dataset stand-in: no-op setup, scripted programmatic verifier."""

    verifier_reward = 1.0
    verifier_raises: Exception | None = None

    def _setup_dataset_specific(self) -> None:
        pass

    def _do_calculate_reward(self, timeout=None, model_patch=""):
        CALLS.append("verifier")
        if self.verifier_raises is not None:
            raise self.verifier_raises
        return self.verifier_reward, "tests: 3 passed", {"tests_status": "PASS"}


class _ScriptedJudge:
    """DefaultAgent stand-in: run() writes a scripted verdicts.json to the pod."""

    IDLE_STATUS = "Idle"
    deliveries: list[str | None] = []  # per-run payload; None = deliver nothing
    last_run_kwargs: dict = {}
    total_runs: int = 0

    def __init__(self, *args, model=None, env=None, **kwargs):
        self.env = env
        self.calls = 0

    def run(self, prompt: str, **kwargs) -> tuple[str, str]:
        if self.calls == 0:
            CALLS.append("judge")
        payload = self.deliveries[min(self.calls, len(self.deliveries) - 1)]
        self.calls += 1
        _ScriptedJudge.total_runs += 1
        if kwargs:
            _ScriptedJudge.last_run_kwargs = kwargs
        if payload is not None:
            self.env.files[f"{WS}/verdicts.json"] = payload
        return "Idle", "done"


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()
    _ScriptedJudge.total_runs = 0
    _VerifierEnv.verifier_reward = 1.0
    _VerifierEnv.verifier_raises = None
    return


def _generic_env(pod=None, judge_agent=JUDGE_CFG, instance=None, reward_mode=None):
    return create_from_registry(
        "generic", pod or _FakePod(), dict(instance or INSTANCE), judge_agent=judge_agent, reward_mode=reward_mode
    )


def _verifier_env(pod=None, judge_agent=JUDGE_CFG, instance=None, reward_mode=None):
    """Go through create_from_registry so the real injection/validation path runs."""
    with mock.patch.dict(DATASET_REGISTRY, {"fake-verified": _VerifierEnv}):
        return create_from_registry(
            "fake-verified",
            pod or _FakePod(),
            dict(instance or INSTANCE),
            judge_agent=judge_agent,
            reward_mode=reward_mode,
        )


def _run_reward(env, deliveries):
    _ScriptedJudge.deliveries = deliveries
    with (
        mock.patch("mimoagent.agents.default.DefaultAgent", _ScriptedJudge),
        mock.patch("mimoagent.models.get_model", return_value=object()),
    ):
        return env.calculate_reward()


def _verdicts_payload(bits: list[int]) -> str:
    return json.dumps({"verdicts": [{"id": i, "verdict": b, "reason": f"r{i}"} for i, b in enumerate(bits)]})


# --- parse_rubrics -----------------------------------------------------------


def test_parse_rubrics_absent_is_none_not_an_error():
    for absent in ({}, {"rubric": None}, {"rubric": {}}, {"rubric": {"rubrics": []}}, {"rubric": {"rubrics": None}}):
        assert parse_rubrics({"instance_id": "x", **absent}) is None


def test_parse_rubrics_malformed_raise():
    bad_blocks = (
        "not an object",
        {"rubrics": "not a list"},
        {"rubrics": ["ok", ""]},
        {"rubrics": [{"category": "process"}]},
        {"rubrics": [{"text": " "}]},
        {"rubrics": [123]},
    )
    for bad in bad_blocks:
        with pytest.raises(ValueError, match="rubric"):
            parse_rubrics({"instance_id": "x", "rubric": bad})


def test_parse_rubrics_accepts_strings_and_categorised_objects():
    parsed = parse_rubrics(
        {"rubric": {"rubrics": ["bare", {"text": "cat", "category": " process "}, {"text": "no cat", "extra": 1}]}}
    )
    assert parsed == [{"text": "bare"}, {"text": "cat", "category": "process"}, {"text": "no cat"}]


# --- reward plan: normal test-graded runs are untouched ----------------------


def test_dataset_without_rubrics_ignores_a_configured_judge():
    """A judge_agent in the yaml must not touch datasets whose instances carry no rubrics."""
    no_rubrics = {k: v for k, v in INSTANCE.items() if k != "rubric"}
    env = _verifier_env(instance=no_rubrics)
    assert env.rubrics is None
    assert env.resolve_reward_plan() == (False, True)

    with mock.patch("mimoagent.agents.default.DefaultAgent", side_effect=AssertionError("judge must not run")):
        reward, test_output, extra = env.calculate_reward()

    assert reward == 1.0
    assert test_output == "tests: 3 passed"
    assert extra == {"tests_status": "PASS", "model_patch": ""}  # no rubric keys leak in
    assert CALLS == ["verifier"]


def test_rubrics_without_judge_config_fall_back_to_the_verifier():
    """Rubrics on the instance but no judge_agent in the yaml: plain test run, no error."""
    env = _verifier_env(judge_agent=None)
    assert env.rubrics is not None
    assert env.resolve_reward_plan() == (False, True)

    with mock.patch("mimoagent.agents.default.DefaultAgent", side_effect=AssertionError("judge must not run")):
        reward, _, extra = env.calculate_reward()

    assert reward == 1.0
    assert "rubric_reward" not in extra
    assert CALLS == ["verifier"]


def test_explicit_programmatic_mode_skips_a_configured_judge():
    env = _verifier_env(reward_mode="programmatic")
    assert env.resolve_reward_plan() == (False, True)
    with mock.patch("mimoagent.agents.default.DefaultAgent", side_effect=AssertionError("judge must not run")):
        reward, _, extra = env.calculate_reward()
    assert reward == 1.0
    assert "rubric_reward" not in extra


# --- reward plan: judge stacked on a verifier --------------------------------


def test_auto_runs_judge_before_verifier_and_keeps_programmatic_scalar():
    env = _verifier_env()
    assert env.resolve_reward_plan() == (True, True)
    env.attach_rollout(agent=None, task="t", result="done")

    reward, test_output, extra = _run_reward(env, [_verdicts_payload([1, 0, 1, 0])])

    # The judge must see the doer's delivered state, so it goes first.
    assert CALLS == ["judge", "verifier"]
    # Default combine: the verifier's reward is the scalar; rubric is informational.
    assert reward == 1.0
    assert extra["reward_mode"] == "both"
    assert extra["reward_combine"] == "programmatic"
    assert extra["programmatic_reward"] == 1.0
    assert extra["rubric_reward"] == 0.5
    assert extra["tests_status"] == "PASS"  # verifier extras preserved
    assert [v["verdict"] for v in extra["rubric_verdicts"]] == [1, 0, 1, 0]
    assert extra["num_rubrics"] == 4
    assert "model_patch" in extra
    # Verifier output first, judge summary last (survives batch.py's tail truncation).
    assert test_output.startswith("tests: 3 passed")
    assert "=== rubric judge ===" in test_output
    assert test_output.index("rubric verdicts (2/4 passed)") > test_output.index("tests: 3 passed")


@pytest.mark.parametrize(
    ("combine", "extra_cfg", "expected"),
    [
        ("programmatic", {}, 0.5),
        ("rubric", {}, 0.75),
        ("product", {}, 0.375),
        ("weighted", {"rubric_weight": 0.5}, 0.625),
        ("weighted", {"rubric_weight": 1.0}, 0.75),
        ("weighted", {"rubric_weight": 0.0}, 0.5),
    ],
)
def test_combine_modes(combine, extra_cfg, expected):
    _VerifierEnv.verifier_reward = 0.5
    env = _verifier_env(judge_agent={**JUDGE_CFG, "combine": combine, **extra_cfg})
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [_verdicts_payload([1, 1, 1, 0])])
    assert reward == pytest.approx(expected)
    assert extra["programmatic_reward"] == 0.5
    assert extra["rubric_reward"] == 0.75
    assert extra["reward_combine"] == combine


def test_judge_failure_in_both_mode_fails_closed_but_keeps_verifier_result():
    env = _verifier_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, test_output, extra = _run_reward(env, [None, None])
    assert CALLS == ["judge", "verifier"]
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "judge_verdicts_missing"
    assert extra["programmatic_reward"] == 1.0  # still reported for debugging
    assert extra["tests_status"] == "PASS"
    assert "judge failed to deliver verdicts" in test_output


def test_verifier_exception_keeps_judge_extras():
    _VerifierEnv.verifier_raises = RuntimeError("test harness exploded")
    env = _verifier_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, test_output, extra = _run_reward(env, [_verdicts_payload([1, 1, 0, 0])])
    assert reward == 0.0
    assert test_output == "test harness exploded"
    assert extra["rubric_reward"] == 0.5
    assert [v["verdict"] for v in extra["rubric_verdicts"]] == [1, 1, 0, 0]


def test_explicit_both_mode_on_verifier_dataset():
    env = _verifier_env(reward_mode="both")
    assert env.resolve_reward_plan() == (True, True)


# --- reward plan: rubric-only on a verifier-less dataset ---------------------


def test_auto_is_rubric_only_on_generic():
    env = _generic_env()
    assert env.resolve_reward_plan() == (True, False)
    env.attach_rollout(agent=None, task="t", result="")
    reward, test_output, extra = _run_reward(env, [_verdicts_payload([1, 0, 1, 0])])
    assert CALLS == ["judge"]
    assert reward == 0.5
    assert extra["reward_mode"] == "rubric"
    assert extra["rubric_reward"] == 0.5
    assert "programmatic_reward" not in extra
    assert test_output.startswith("rubric verdicts (2/4 passed)")


def test_explicit_rubric_mode_on_verifier_dataset_skips_the_verifier():
    env = _verifier_env(reward_mode="rubric")
    assert env.resolve_reward_plan() == (True, False)
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [_verdicts_payload([1, 1, 1, 1])])
    assert CALLS == ["judge"]
    assert reward == 1.0
    assert "tests_status" not in extra


# --- registry-time validation --------------------------------------------------


def test_registry_injects_judge_config_and_reward_mode():
    env = _generic_env(reward_mode="rubric")
    assert env.judge_agent_config == JUDGE_CFG
    assert env.reward_mode == "rubric"
    # default when the yaml says nothing
    assert _generic_env().reward_mode == "auto"


def test_registry_rejects_misconfigured_explicit_modes():
    no_rubrics = {k: v for k, v in INSTANCE.items() if k != "rubric"}
    with pytest.raises(ValueError, match="judge_agent"):
        _generic_env(judge_agent=None, reward_mode="rubric")
    with pytest.raises(ValueError, match="rubric.rubrics"):
        _generic_env(instance=no_rubrics, reward_mode="rubric")
    with pytest.raises(ValueError, match="no programmatic verifier"):
        _generic_env(reward_mode="both")
    with pytest.raises(ValueError, match="reward_mode"):
        _generic_env(reward_mode="sometimes")
    with pytest.raises(ValueError, match="judge_agent"):
        _verifier_env(judge_agent=None, reward_mode="both")


def test_registry_validates_the_judge_block_up_front():
    with pytest.raises(ValueError, match="model"):
        _generic_env(judge_agent={"step_limit": 5})
    with pytest.raises(ValueError, match="combine"):
        _generic_env(judge_agent={**JUDGE_CFG, "combine": "average"})
    with pytest.raises(ValueError, match="rubric_weight"):
        _generic_env(judge_agent={**JUDGE_CFG, "rubric_weight": 1.5})
    with pytest.raises(ValueError, match="solution_combine"):
        _generic_env(judge_agent={**JUDGE_CFG, "solution_combine": "mean"})


def test_attach_rollout_exists_on_every_dataset_env():
    env = _verifier_env(judge_agent=None, instance={k: v for k, v in INSTANCE.items() if k != "rubric"})
    env.attach_rollout(agent=object(), task="t", result="r")  # harmless no-op without a judge
    assert env._task == "t"


# --- judge internals -----------------------------------------------------------


def test_reward_is_mean_of_verdicts_and_workspace_is_built(tmp_path):
    from mimoagent.utils.log import AgentLogContext, use_log_context

    pod = _FakePod()
    env = _generic_env(pod)

    class _Doer:
        messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "done"}]

    env.attach_rollout(agent=_Doer(), task="fix the wobble", result="I fixed it")
    # A live log context enables the workspace copy_out (skipped without one)
    # and gives the judge its msg_path / trajectory registration.
    log_ctx = AgentLogContext.create(tmp_path, "t-rubric")
    with use_log_context(log_ctx):
        reward, test_output, extra = _run_reward(env, [_verdicts_payload([1, 0, 1, 0])])
    log_ctx.close()

    assert reward == 0.5
    assert extra["num_rubrics"] == 4
    assert [v["verdict"] for v in extra["rubric_verdicts"]] == [1, 0, 1, 0]
    assert extra["judge_exit_status"] == "Idle"
    assert "model_patch" in extra  # injected by the base template method
    # workspace inputs materialised in the pod
    for name in ("task.md", "rubrics.json", "response.md", "model_patch.diff", "trajectory.json"):
        assert f"{WS}/{name}" in pod.files, name
    assert pod.files[f"{WS}/task.md"] == "fix the wobble"
    assert [r["rubric"] for r in json.loads(pod.files[f"{WS}/rubrics.json"])] == RUBRICS
    # system-template vars handed to the judge
    assert _ScriptedJudge.last_run_kwargs["verdicts_path"] == f"{WS}/verdicts.json"
    assert _ScriptedJudge.last_run_kwargs["repo_path"] == env.repo_path
    # workspace copied out of the pod
    assert pod.copied_out and pod.copied_out[0][0] == WS


def test_categorised_rubrics_reach_the_judge_and_the_extras():
    categorised = [
        {"category": "final_state", "text": "compiles cleanly"},
        {"category": "process", "text": "reproduced the bug before fixing"},
        {"text": "no category on this one"},
    ]
    pod = _FakePod()
    env = _generic_env(pod, instance={**INSTANCE, "rubric": {"rubrics": categorised}})
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [_verdicts_payload([1, 1, 0])])

    # The judge is handed the text plus the category, which is omitted entirely
    # rather than sent as null when the producer did not supply one.
    published = json.loads(pod.files[f"{WS}/rubrics.json"])
    assert [r["rubric"] for r in published] == [r["text"] for r in categorised]
    assert [r.get("category") for r in published] == ["final_state", "process", None]
    assert "category" not in published[2]

    # reward_extra_info keeps "rubric" as the criterion text for downstream readers.
    assert reward == 2 / 3
    assert [v["rubric"] for v in extra["rubric_verdicts"]] == [r["text"] for r in categorised]
    assert extra["rubric_verdicts"][1]["category"] == "process"
    assert "category" not in extra["rubric_verdicts"][2]


def test_missing_and_invalid_verdicts_score_zero():
    env = _generic_env()
    env.attach_rollout(agent=None, task="t", result="")
    # id 1 has a non-int verdict, id 3 is absent entirely
    payload = json.dumps(
        {
            "verdicts": [
                {"id": 0, "verdict": 1, "reason": "ok"},
                {"id": 1, "verdict": "yes", "reason": "bad type"},
                {"id": 2, "verdict": 0, "reason": "no"},
            ]
        }
    )
    reward, _, extra = _run_reward(env, [payload])
    assert reward == 0.25
    assert extra["rubric_verdicts"][3]["reason"] == "no verdict delivered for this rubric"


def test_recalc_mode_without_doer_skips_trajectory_and_response():
    pod = _FakePod()
    env = _generic_env(pod)
    env.attach_rollout(agent=None, task="", result="")
    reward, _, _ = _run_reward(env, [_verdicts_payload([1, 1, 1, 1])])
    assert reward == 1.0
    assert f"{WS}/trajectory.json" not in pod.files
    assert f"{WS}/response.md" not in pod.files
    # task falls back to the instance's problem_statement
    assert pod.files[f"{WS}/task.md"] == INSTANCE["problem_statement"]


def test_missing_delivery_is_not_retried():
    """The judge gets exactly one run; a second scripted delivery must never be reached."""
    env = _generic_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [None, _verdicts_payload([1, 1, 0, 0])])
    assert _ScriptedJudge.total_runs == 1
    assert reward == 0.0 and extra["error"] == "judge_verdicts_missing"


def test_judge_never_delivering_marks_testbed_corrupted():
    env = _generic_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [None, None])
    assert reward == 0.0
    assert extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert extra["error"] == "judge_verdicts_missing"
    assert extra["reward_mode"] == "rubric"


def test_malformed_json_fails_closed_without_retry():
    env = _generic_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, test_output, extra = _run_reward(env, ["{not json", _verdicts_payload([0, 0, 0, 1])])
    assert _ScriptedJudge.total_runs == 1
    assert reward == 0.0 and extra["error_category"] == REWARD_TESTBED_CORRUPTED
    assert "invalid JSON" in test_output


def test_read_verdicts_ignores_login_shell_output():
    class _NoisyPod(_FakePod):
        def execute(self, command: str, **kwargs) -> dict:
            result = super().execute(command, **kwargs)
            result["output"] = (
                "JAVA_HOME=/opt/jdk\n"
                "java debugger: /opt/arthas/as.sh\n"
                f"{result['output']}"
                "login shell cleanup complete\n"
            )
            return result

    pod = _NoisyPod()
    payload = _verdicts_payload([1, 0, 1, 0])
    pod.files[f"{WS}/verdicts.json"] = payload
    env = _generic_env(pod)
    judge = RubricJudge(env, env.rubrics, RubricJudgeConfig.from_dict(JUDGE_CFG))

    payload_out, error = judge._read_verdicts()

    assert error == ""
    assert payload_out["verdicts"] == json.loads(payload)["verdicts"]


# --- producer-supplied grader: how_to_judge / graded solution -------------------


GRADER_RUBRICS = [
    {
        "id": "A1",
        "text": "adds a regression test that fails on base and passes after",
        "category": "final_state",
        "subcategory": "artifacts",
        "how_to_judge": "run the added test on base and on the patched tree",
    },
    {
        "id": "A2",
        "text": "the added tests' asserted values hold",
        "category": "final_state",
        "subcategory": "artifacts",
    },
    {
        "id": "P1",
        "text": "does not search git history for the upstream fix",
        "category": "process",
        "subcategory": "reward_hack",
    },
    {
        "id": "C1",
        "text": "summary claims are supported by the trajectory",
        "category": "process",
        "subcategory": "user_communication",
    },
]
SOLUTION = {
    "text": "acceptable solutions rank by crash-freedom on empty input and by side-effect freedom",
    "levels": [
        {"score": 1.0, "description": "handles empty input and uses colors/safe", "how_to_judge": "run oracle"},
        {"score": 0.8, "description": "handles empty input, colors main"},
        {"score": 0.6, "description": "crashes on empty input"},
    ],
}
GRADER_INSTANCE = {**INSTANCE, "rubric": {"rubrics": GRADER_RUBRICS, "solution": SOLUTION}}


def _grader_payload(bits, level=None):
    d = {}
    if level is not None:
        d["solution"] = {"level": level, "reason": "measured"}
    d["verdicts"] = [{"id": i, "verdict": b, "reason": f"r{i}"} for i, b in enumerate(bits)]
    return json.dumps(d)


def test_parse_rubrics_keeps_producer_fields_only_when_set():
    parsed = parse_rubrics({"rubric": {"rubrics": GRADER_RUBRICS}})
    assert parsed[0] == {
        "text": GRADER_RUBRICS[0]["text"],
        "category": "final_state",
        "subcategory": "artifacts",
        "how_to_judge": "run the added test on base and on the patched tree",
        "label": "A1",
    }
    assert parsed[1] == {
        "text": GRADER_RUBRICS[1]["text"],
        "category": "final_state",
        "subcategory": "artifacts",
        "label": "A2",
    }
    assert parsed[3]["label"] == "C1" and "how_to_judge" not in parsed[3]


def test_parse_solution_validates_shape():
    from mimoagent.environments.rubric_judge import parse_solution, parse_solution_weight

    assert parse_solution(INSTANCE) is None
    sol = parse_solution(GRADER_INSTANCE)
    assert [lv["index"] for lv in sol["levels"]] == [0, 1, 2]
    assert [lv["score"] for lv in sol["levels"]] == [1.0, 0.8, 0.6]
    assert sol["levels"][0]["how_to_judge"] == "run oracle" and "how_to_judge" not in sol["levels"][1]
    bad = (
        {"levels": []},
        {"levels": [{"score": 1.5, "description": "x"}]},
        {"levels": [{"score": 1.0, "description": ""}]},
        {"levels": [{"score": 0.5, "description": "a"}, {"score": 1.0, "description": "b"}]},  # not best-first
        "not an object",
    )
    for b in bad:
        with pytest.raises(ValueError, match="solution"):
            parse_solution({"instance_id": "x", "rubric": {"rubrics": ["r"], "solution": b}})
    assert parse_solution_weight(INSTANCE, 0.5) == 0.5
    assert parse_solution_weight({"rubric": {"rubrics": ["r"], "solution_weight": 0.3}}, 0.5) == 0.3
    with pytest.raises(ValueError, match="solution_weight"):
        parse_solution_weight({"instance_id": "x", "rubric": {"rubrics": ["r"], "solution_weight": 2}}, 0.5)


def test_aggregate_rubric_scores_rules():
    from mimoagent.environments.rubric_judge import aggregate_rubric_scores, parse_rubrics, parse_solution

    rubrics = parse_rubrics({"rubric": {"rubrics": GRADER_RUBRICS}})
    sol = parse_solution(GRADER_INSTANCE)
    # B is the plain mean; every rubric applies to every rollout, so a null / missing verdict is 0
    a = aggregate_rubric_scores(rubrics, {0: 1, 1: None, 2: 1, 3: 0})
    assert (a["num_satisfied"], a["num_rubrics"]) == (2, 4) and a["behavior_score"] == pytest.approx(0.5)
    assert a["rubric_reward"] == pytest.approx(0.5) and "solution_score" not in a
    b = aggregate_rubric_scores(rubrics, {0: 1, 1: 1, 2: 1})
    assert b["behavior_score"] == pytest.approx(3 / 4)
    # solution term: reward = w*S + (1-w)*B
    d = aggregate_rubric_scores(rubrics, {0: 1, 1: 0, 2: 1, 3: 0}, solution=sol, solution_level=1, solution_weight=0.5)
    assert d["solution_score"] == 0.8 and d["rubric_reward"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.5)
    e = aggregate_rubric_scores(rubrics, {0: 1, 1: 0, 2: 1, 3: 0}, solution=sol, solution_level=1, solution_weight=1.0)
    assert e["rubric_reward"] == pytest.approx(0.8)
    # missing / out-of-range level fails closed to S = 0
    f = aggregate_rubric_scores(rubrics, {0: 1, 1: 1, 2: 1, 3: 1}, solution=sol, solution_level=None)
    assert f["solution_level"] is None and f["solution_score"] == 0.0 and f["rubric_reward"] == pytest.approx(0.5)
    g = aggregate_rubric_scores(rubrics, {0: 1, 1: 1, 2: 1, 3: 1}, solution=sol, solution_level=7)
    assert g["solution_level"] is None and g["rubric_reward"] == pytest.approx(0.5)
    # product mode: reward = S * B, solution_weight ignored; invalid level still fails closed (S=0 -> 0)
    h = aggregate_rubric_scores(
        rubrics,
        {0: 1, 1: 0, 2: 1, 3: 0},
        solution=sol,
        solution_level=1,
        solution_weight=0.9,
        solution_combine="product",
    )
    assert h["solution_combine"] == "product" and h["rubric_reward"] == pytest.approx(0.8 * 0.5)
    i = aggregate_rubric_scores(
        rubrics, {0: 1, 1: 1, 2: 1, 3: 1}, solution=sol, solution_level=None, solution_combine="product"
    )
    assert i["rubric_reward"] == 0.0
    # without a solution rubric both modes reduce to B
    j = aggregate_rubric_scores(rubrics, {0: 1, 1: 1, 2: 1, 3: 0}, solution_combine="product")
    assert j["rubric_reward"] == pytest.approx(3 / 4) and "solution_combine" not in j
    with pytest.raises(AssertionError, match="solution_combine"):
        aggregate_rubric_scores(rubrics, {0: 1}, solution=sol, solution_level=0, solution_combine="mean")


def test_grader_reaches_the_judge_and_scores_end_to_end():
    pod = _FakePod()
    env = _generic_env(pod, instance=GRADER_INSTANCE, judge_agent={**JUDGE_CFG, "solution_weight": 0.5})
    env.attach_rollout(agent=None, task="t", result="")
    reward, test_output, extra = _run_reward(env, [_grader_payload([1, 0, 1, 0], level=1)])

    published = json.loads(pod.files[f"{WS}/rubrics.json"])
    assert published[0]["how_to_judge"].startswith("run the added test") and published[0]["label"] == "A1"
    assert set(published[1]) == {"id", "rubric", "label", "category", "subcategory"}
    sol = json.loads(pod.files[f"{WS}/solution.json"])
    assert [lv["score"] for lv in sol["levels"]] == [1.0, 0.8, 0.6]
    assert _ScriptedJudge.last_run_kwargs["has_solution"] is True

    # B = 2/4 (A1 pass, A2 fail, P1 pass, C1 fail), S = 0.8 → 0.5*0.8 + 0.5*0.5
    assert reward == pytest.approx(0.65)
    assert extra["behavior_score"] == 0.5 and extra["num_satisfied"] == 2
    assert extra["solution_level"] == 1 and extra["solution_score"] == 0.8 and extra["solution_weight"] == 0.5
    verdicts = extra["rubric_verdicts"]
    assert [v["verdict"] for v in verdicts] == [1, 0, 1, 0]
    assert verdicts[0]["label"] == "A1" and verdicts[1]["subcategory"] == "artifacts"
    assert "[FAIL] rubric[1] A2" in test_output and "S=0.80" in test_output
    # solution leads: the [S] line comes right after the head, and solution_* keys open reward_extra_info
    assert test_output.splitlines()[1].startswith("[S] solution level 1: measured")
    assert list(extra)[:2] == ["solution_level", "solution_score"]


def test_judge_templates_deliver_solution_before_verdicts():
    from jinja2 import Template

    from mimoagent.environments.rubric_judge import _JUDGE_PROMPT, _JUDGE_SYSTEM

    common = {"workspace_dir": WS, "verdicts_path": f"{WS}/verdicts.json", "has_solution": True}
    system = Template(_JUDGE_SYSTEM).render(repo_path="/testbed", workspace_files=[], **common)
    shape = system[system.index("Delivering your verdicts") :]
    assert shape.index('{"solution": {"level"') < shape.index('"verdicts": [')
    assert "score" not in shape  # the judge reports a level index only; S is that level's score
    prompt = Template(_JUDGE_PROMPT).render(task="t", num_rubrics=4, **common)
    assert prompt.startswith("Place the delivered solution on the graded rubric")
    # without a solution rubric the shape is the plain verdicts object
    plain = Template(_JUDGE_SYSTEM).render(
        repo_path="/testbed", workspace_files=[], **{**common, "has_solution": False}
    )
    assert '{"verdicts": [' in plain and "solution" not in plain


def test_instance_solution_weight_overrides_config():
    inst = {**GRADER_INSTANCE, "rubric": {**GRADER_INSTANCE["rubric"], "solution_weight": 1.0}}
    env = _generic_env(instance=inst, judge_agent={**JUDGE_CFG, "solution_weight": 0.2})
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [_grader_payload([1, 0, 1, 0], level=2)])
    # w = 1.0 from the instance: the reward is the solution score alone
    assert extra["solution_weight"] == 1.0 and reward == pytest.approx(0.6)


def test_missing_solution_level_fails_closed_and_null_verdict_scores_zero():
    env = _generic_env(instance=GRADER_INSTANCE)
    env.attach_rollout(agent=None, task="t", result="")
    # no "solution" object at all; C1 answered null (there is no not-applicable: it is a 0)
    reward, test_output, extra = _run_reward(env, [_grader_payload([1, 1, 1, None])])
    assert extra["solution_level"] is None and extra["solution_score"] == 0.0
    assert extra["rubric_verdicts"][3]["verdict"] == 0
    assert reward == pytest.approx(0.5 * 0.0 + 0.5 * (3 / 4))
    assert "no valid solution level delivered" in test_output


def test_legacy_rubrics_without_solution_keep_plain_mean_semantics():
    env = _generic_env()
    env.attach_rollout(agent=None, task="t", result="")
    reward, _, extra = _run_reward(env, [_verdicts_payload([1, 0, 1, 0])])
    assert reward == 0.5
    assert "solution_level" not in extra and "behavior_score" in extra
    assert _ScriptedJudge.last_run_kwargs["has_solution"] is False


def test_registry_validates_solution_weight():
    with pytest.raises(ValueError, match="solution_weight"):
        _generic_env(judge_agent={**JUDGE_CFG, "solution_weight": 1.5})
