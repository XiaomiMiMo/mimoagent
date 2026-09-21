"""Rubric judge: an agentic reward stage that stacks on any dataset environment.

Rubric is **not** a dataset type. Any instance —
may carry ``rubric.rubrics``: a list of natural-language criteria, each
independently decidable as pass/fail against the environment's final state. An
item is either a bare string or an object ``{"text": ..., "category": ...}``,
where the optional category labels what kind of behavior the criterion judges
(e.g. final_state / process / user_communication — an open vocabulary, passed
through verbatim; the judge treats it as a pointer to the primary evidence).

Setup and programmatic grading stay with whatever dataset the instance belongs
to. The judge only enters the picture at reward time, from
:meth:`DatasetEnvironment.calculate_reward`, when the instance has rubrics AND
the yaml ``environment.judge_agent`` block is present (see ``reward_mode`` in
``datasets/base.py`` for how the judge combines with the dataset's verifier).

At reward time a **judge agent** (a :class:`DefaultAgent` with its own model
config, same pattern as the user-agent sidecar) enters the doer's pod — the
final state the doer left behind, ``.git`` already restored by the base class.
A private workspace inside the pod gives it everything it needs to judge:

* ``task.md``       — the original task
* ``response.md``   — the doer's final response
* ``model_patch.diff`` — the doer's captured diff vs base
* ``rubrics.json``  — the criteria to judge,
  ``[{"id": 0, "rubric": ..., "category": ...}, ...]`` (category when supplied)
* the doer's trajectory — blackbox scaffolds already write their session log
  inside the pod (referenced in place via ``ENV_TRAJECTORY_FILES``); native
  agents get their host-side msg file mirrored in (same three-tier logic as
  ``UserAgentDriver._publish_trajectory``).

The judge may do anything in the environment to decide each rubric — read
code, run tests, build, git diff. Delivery is file-based (never parsed from
chat text): it writes ``verdicts.json`` (solution placement first, then the
per-rubric verdicts) into the workspace, the env reads it back exactly once —
a missing or malformed file is an infra fault reported to the caller, never
re-prompted here — and the whole workspace is copied out of the pod next to
the instance's logs.

Scoring (see :func:`aggregate_rubric_scores`). With the plain shapes above the
rubric reward is the mean of the per-rubric 1/0 verdicts, exactly as before.
A producer that knows more about its rubrics can say so, per item or per
instance, and the score follows:

* ``how_to_judge`` — the concrete procedure (command, probe, diff property)
  that decides the rubric; handed to the judge verbatim.
* ``id`` / ``label``, ``subcategory`` — carried through to the judge and to
  ``reward_extra_info`` untouched.
* ``rubric.solution`` — one optional **graded** rubric: ``levels`` listed
  best-first, each with a ``score`` in [0, 1] and the ``description`` (and
  optionally ``how_to_judge``) of the solution property that defines the
  level. The judge names the highest level the delivered solution reaches;
  its score is ``S``. ``B`` is the mean of the binary verdicts (every rubric
  applies to every rollout — a rubric is written as an obligation, never as
  "if the doer did X…"). ``judge_agent.solution_combine`` picks how the two
  fold: ``weighted`` (default) gives ``rubric_reward = w·S + (1−w)·B`` with
  ``w`` = ``judge_agent.solution_weight`` (default 0.5, overridable per
  instance via ``rubric.solution_weight``); ``product`` gives
  ``rubric_reward = S·B`` (solution quality scales the behaviour score, so a
  poor solution caps the reward no matter how clean the process — used by the
  RL side to rank test-passing rollouts). Without a solution rubric the reward
  is ``B`` in both modes. Rubrics never gate each other; the only gate is the dataset's own
  verifier, through ``judge_agent.combine: product``.

The rollout context (doer agent handle, task, final response) is not available
to ``calculate_reward()`` by design — the runner attaches it via
:meth:`DatasetEnvironment.attach_rollout` right before reward (see
``batch.py``); the env ctor keeps the uniform ``(base_env, instance)``
signature for embedding frameworks. Judge agent config comes from the yaml
``environment.judge_agent`` block, threaded through ``make_dataset_env`` like
``anti_hack_cleanup``.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template

from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs
from mimoagent.utils.log import get_log_context

if TYPE_CHECKING:
    from mimoagent.environments.datasets.base import DatasetEnvironment

_JUDGE_SYSTEM = """\
You are a rigorous, impartial judge of an agent's work. An agent (the "doer") just finished working on a task inside this environment; you are now inside the SAME environment, seeing the exact final state the doer left behind (repository: {{repo_path}}). Your job is {% if has_solution %}first to place the delivered solution on the graded solution rubric in {{workspace_dir}}/solution.json, then {% endif %}to judge a list of rubrics — each one expected behavior of the doer's work, independently decidable — and decide 1 (satisfied) or 0 (not satisfied) for every one of them.

Evidence for your judgment lives in {{workspace_dir}}:

{% for f in workspace_files %}- {{ f.path }} — {{ f.desc }}
{% endfor %}
How to judge:

- Each rubric states an expected behavior; your job is to establish from evidence whether it occurred. A rubric may carry a `category` labeling what kind of behavior is judged — for example the delivered final state (code, files, artifacts, the work product), the doer's process (how it actually worked, evidenced in its trajectory), or its communication with the user. Treat the category as a pointer to the primary evidence, not a restriction: use whatever evidence decides the criterion.
- Deliverables differ by task: a code change, a written report or analysis, a diagnosis, a design, execution evidence, or a mix. When the deliverable IS the response (an explanation, a review, a report), judge the response as the artifact — checking its claims against the actual environment, not just its plausibility. When a rubric concerns the behavior of the delivered system, verify in the environment rather than trusting what the response or trajectory asserts: read the actual code/diff, run things.
- Judge the work AS DELIVERED. Never fix, complete, or improve the doer's work before judging — repairing a defect and then judging the repaired version is a false verdict. If a check requires mutating state (installing a dependency, building, applying a patch, writing a probe), keep the mutation reversible and restore the doer's state afterwards, so every rubric is judged against the doer's state, not yours.
- Trajectory files can be long — read selectively (grep/tail/sed ranges) rather than dumping them.
- When the primary evidence for a rubric is unavailable in this run (no trajectory on record, a missing toolchain, an artifact that was never produced), judge from the best remaining evidence — in both directions: do not fail a rubric solely because a tool is unavailable when inspection can decide it, and do not pass one on claims the available evidence cannot support. State the limitation in your `reason` so the verdict is auditable.
- A rubric may carry `how_to_judge`: the procedure its author used to decide it. It is there to help you — use it when it applies; judge by whatever evidence decides the rubric most reliably in this environment.
{% if has_solution %}- The solution rubric ({{workspace_dir}}/solution.json) is graded, not binary: its `levels` are listed best-first, each describing a property of the delivered solution (with `how_to_judge` where the author could pin one). Establish, from the highest level downward, the first level whose description the delivered solution actually meets, and report its `index`. A level that describes a defect the solution exhibits, or a property it lacks, is not met.
{% endif %}- Be strict: a rubric is 1 only if the evidence establishes it is satisfied. Partial credit is 0. Every rubric applies to every rollout: a rubric that names an obligation the doer never carried out (a test it did not add, a check it did not run, a disclosure it did not make) is 0.

Delivering your verdicts:

- When you have judged everything, write a single JSON file to {{verdicts_path}} with exactly this shape:
  {{ '{' }}{% if has_solution %}"solution": {"level": <index of the level reached, from solution.json>, "reason": "<one-sentence evidence-backed justification>"},
   {% endif %}"verdicts": [{"id": 0, "verdict": 1, "reason": "<one-sentence evidence-backed justification>"}, ...]}
- Include every rubric id from rubrics.json exactly once; verdict must be the integer 1 or 0.
- After writing the file, end your turn."""

_JUDGE_PROMPT = """\
{% if has_solution %}Place the delivered solution on the graded rubric in {{workspace_dir}}/solution.json, then judge{% else %}Judge{% endif %} the {{num_rubrics}} rubric(s) in {{workspace_dir}}/rubrics.json against this environment's final state, and write your verdicts to {{verdicts_path}}.

The original task given to the doer:

<original_task>
{{task}}
</original_task>
"""


RUBRIC_COMBINE_MODES = ("programmatic", "rubric", "product", "weighted")
# How the graded solution score S folds with the binary mean B (see aggregate_rubric_scores).
SOLUTION_COMBINE_MODES = ("weighted", "product")
"""How the rubric score folds into the scalar reward when BOTH the dataset's
verifier and the judge ran (``judge_agent.combine`` in the yaml). The
verifier's and the judge's scores are always reported separately in
``reward_extra_info`` (``programmatic_reward`` / ``rubric_reward``); this only
decides the scalar:

* ``programmatic`` — scalar is the verifier's reward; the rubric score is
  informational. **Default**, so adding rubrics to a test-graded dataset never
  changes what it trains on unless asked to.
* ``rubric``       — scalar is the rubric mean; the verifier is informational.
* ``product``      — ``programmatic * rubric``: tests gate, rubrics scale.
* ``weighted``     — ``(1 - w) * programmatic + w * rubric`` with
  ``w = judge_agent.rubric_weight``.
"""


def parse_rubrics(instance: dict) -> list[dict] | None:
    """Read ``rubric.rubrics`` into a uniform ``[{"text", "category"?}, ...]``.

    Returns ``None`` when the instance carries no rubrics (no ``rubric`` block,
    or an empty/absent ``rubrics`` list) — rubrics are optional on every
    dataset, so absence is not an error. Malformed items DO raise: a producer
    that wrote rubrics wants them judged, and a silently dropped criterion
    would be a wrong reward, not a missing feature.

    Two authoring shapes are accepted per item, so a producer can say which
    evidence decides a criterion without every producer having to:

    * ``"the build succeeds"`` — a bare criterion, no category.
    * ``{"text": "...", "category": "process"}`` — categorised; ``category``
      names the primary evidence the judge should examine (see
      ``_JUDGE_SYSTEM``).
    * the same object may also carry ``id``/``label`` (producer's own name for
      the rubric), ``subcategory`` and ``how_to_judge`` (the judging
      procedure). Optional keys are kept only when set,
      so plain items parse exactly as before. Unknown sibling keys are ignored.
    """
    rubric_block = instance.get("rubric")
    if not rubric_block:
        return None
    if not isinstance(rubric_block, dict):
        raise ValueError(
            f"{instance.get('instance_id')}: 'rubric' must be an object with a 'rubrics' list; got {rubric_block!r}"
        )
    raw = rubric_block.get("rubrics")
    if not raw:
        return None
    if not isinstance(raw, list):
        raise ValueError(
            f"{instance.get('instance_id')}: 'rubric.rubrics' must be a list of strings or "
            f"{{text, category}} objects; got {raw!r}"
        )
    parsed: list[dict] = []
    for i, item in enumerate(raw):
        text = item if isinstance(item, str) else (item.get("text") if isinstance(item, dict) else None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{instance.get('instance_id')}: rubric.rubrics[{i}] must be a non-empty string "
                f"or an object with a non-empty 'text'; got {item!r}"
            )
        entry = {"text": text}
        if isinstance(item, dict):
            for key in ("category", "subcategory", "how_to_judge"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    entry[key] = val.strip()
            label = item.get("label", item.get("id"))
            if label is not None and str(label).strip():
                entry["label"] = str(label).strip()
        parsed.append(entry)
    return parsed


def parse_solution(instance: dict) -> dict | None:
    """Read the optional graded ``rubric.solution`` rubric.

    Returns ``{"text", "levels": [{"index", "score", "description", "how_to_judge"?}, ...]}``
    with levels in the producer's (best-first) order, or None when absent.
    Malformed blocks raise, for the same reason malformed rubrics do.
    """
    rubric_block = instance.get("rubric")
    if not isinstance(rubric_block, dict):
        return None
    sol = rubric_block.get("solution")
    if not sol:
        return None
    iid = instance.get("instance_id")
    if not isinstance(sol, dict):
        raise ValueError(f"{iid}: 'rubric.solution' must be an object with a 'levels' list; got {sol!r}")
    levels = sol.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"{iid}: 'rubric.solution.levels' must be a non-empty list")
    parsed_levels: list[dict] = []
    for i, lv in enumerate(levels):
        if not isinstance(lv, dict):
            raise ValueError(f"{iid}: rubric.solution.levels[{i}] must be an object; got {lv!r}")
        score = lv.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float) or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"{iid}: rubric.solution.levels[{i}].score must be a number in [0, 1]; got {score!r}")
        desc = lv.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(f"{iid}: rubric.solution.levels[{i}].description must be a non-empty string")
        entry = {"index": i, "score": float(score), "description": desc.strip()}
        htj = lv.get("how_to_judge")
        if isinstance(htj, str) and htj.strip():
            entry["how_to_judge"] = htj.strip()
        parsed_levels.append(entry)
    scores = [lv["score"] for lv in parsed_levels]
    if scores != sorted(scores, reverse=True):
        raise ValueError(
            f"{iid}: rubric.solution.levels must be listed best-first (scores non-increasing); got {scores}"
        )
    text = sol.get("text")
    return {"text": text.strip() if isinstance(text, str) else "", "levels": parsed_levels}


def parse_solution_weight(instance: dict, default: float) -> float:
    """Per-instance ``rubric.solution_weight`` override of the config default."""
    rubric_block = instance.get("rubric")
    if not isinstance(rubric_block, dict) or "solution_weight" not in rubric_block:
        return float(default)
    w = rubric_block["solution_weight"]
    if isinstance(w, bool) or not isinstance(w, int | float) or not 0.0 <= float(w) <= 1.0:
        raise ValueError(f"{instance.get('instance_id')}: rubric.solution_weight must be a number in [0, 1]; got {w!r}")
    return float(w)


def aggregate_rubric_scores(
    rubrics: list[dict],
    verdicts: dict[int, int],
    *,
    solution: dict | None = None,
    solution_level: int | None = None,
    solution_weight: float = 0.5,
    solution_combine: str = "weighted",
) -> dict:
    """Fold per-rubric verdicts (and the solution level) into the rubric reward.

    ``verdicts`` maps rubric index → 1 / 0; anything else (missing, null,
    wrong type) counts as 0. Rules:

    * ``B`` = mean of the verdicts (every rubric applies to every rollout);
    * with a solution rubric, ``S`` is the score of the reported level (0.0 if
      no valid level was delivered) and ``reward = w·S + (1−w)·B``
      (``solution_combine="weighted"``) or ``reward = S·B``
      (``solution_combine="product"``); without one ``reward = B``.

    Pure — usable offline to re-score a stored verdict matrix.
    """
    assert solution_combine in SOLUTION_COMBINE_MODES, (
        f"solution_combine must be one of {SOLUTION_COMBINE_MODES}, got {solution_combine!r}"
    )
    per_rubric: list[dict] = []
    for i in range(len(rubrics)):
        v = 1 if verdicts.get(i) == 1 and not isinstance(verdicts.get(i), bool) else 0
        per_rubric.append({"index": i, "verdict": v})
    satisfied = sum(p["verdict"] for p in per_rubric)
    behavior = satisfied / len(rubrics) if rubrics else 0.0
    out = {
        "behavior_score": behavior,
        "num_satisfied": satisfied,
        "num_rubrics": len(rubrics),
        "per_rubric": per_rubric,
    }
    if solution is None:
        out["rubric_reward"] = behavior
        return out
    levels = solution["levels"]
    valid = (
        isinstance(solution_level, int) and not isinstance(solution_level, bool) and 0 <= solution_level < len(levels)
    )
    s_score = levels[solution_level]["score"] if valid else 0.0
    w = float(solution_weight)
    reward = s_score * behavior if solution_combine == "product" else w * s_score + (1.0 - w) * behavior
    out.update(
        {
            "solution_level": solution_level if valid else None,
            "solution_score": s_score,
            "solution_weight": w,
            "solution_combine": solution_combine,
            "rubric_reward": reward,
        }
    )
    return out


@dataclass
class RubricJudgeConfig:
    """Config for the judge agent (yaml ``environment.judge_agent`` block).

    ``model`` is required: the judge always runs on its own model config (same
    shape as the top-level ``model:`` block), independent of the doer's.
    """

    model: dict[str, Any] = field(default_factory=dict)
    # Underlying DefaultAgent knobs. bash+read suffice: the judge runs tests
    # via bash and delivers verdicts as a file, it never edits code.
    tools: list[dict[str, Any]] = field(default_factory=lambda: [{"tool": "bash"}, {"tool": "read"}])
    step_limit: int = 100
    system_template: str = _JUDGE_SYSTEM
    prompt_template: str = _JUDGE_PROMPT
    # Private directory inside the environment holding the judge's inputs
    # (task/response/rubrics/patch/trajectory) and its verdicts.json output.
    workspace_dir: str = "/tmp/.mimo-rubric-judge"
    # Name used for the msg file (agent_msgs/<name>.log) and trajectory key.
    name: str = "judge"
    # Scalar-reward combination when the dataset's verifier ALSO ran (see
    # RUBRIC_COMBINE_MODES). Irrelevant in rubric-only runs.
    combine: str = "programmatic"
    rubric_weight: float = 0.5
    # Weight of the graded solution rubric against the binary rubrics' mean,
    # when the instance carries ``rubric.solution`` (see aggregate_rubric_scores).
    # ``rubric.solution_weight`` on the instance overrides it.
    solution_weight: float = 0.5
    # ``weighted``: w·S + (1−w)·B (solution_weight applies); ``product``: S·B
    # (solution_weight ignored). See SOLUTION_COMBINE_MODES / aggregate_rubric_scores.
    solution_combine: str = "weighted"

    @classmethod
    def from_dict(cls, raw: dict | None) -> RubricJudgeConfig:
        """Build + validate from the yaml block. Raises ValueError on misconfig
        so a bad judge block fails at env creation, not after a full rollout."""
        if not raw:
            raise ValueError(
                "rubric judging requires a judge agent config: set the `judge_agent:` key "
                "(with its own `model:`) inside the yaml `environment:` block."
            )
        cfg = cls(**filter_dataclass_kwargs(cls, dict(raw)))
        if not cfg.model:
            raise ValueError(
                "judge_agent requires its own model config: set the `model:` key inside the "
                "`environment.judge_agent:` block (same shape as the top-level `model:` block)."
            )
        if cfg.combine not in RUBRIC_COMBINE_MODES:
            raise ValueError(f"Invalid judge_agent.combine {cfg.combine!r}; expected one of {RUBRIC_COMBINE_MODES}")
        if not 0.0 <= float(cfg.rubric_weight) <= 1.0:
            raise ValueError(f"judge_agent.rubric_weight must be within [0, 1]; got {cfg.rubric_weight!r}")
        if not 0.0 <= float(cfg.solution_weight) <= 1.0:
            raise ValueError(f"judge_agent.solution_weight must be within [0, 1]; got {cfg.solution_weight!r}")
        if cfg.solution_combine not in SOLUTION_COMBINE_MODES:
            raise ValueError(
                f"Invalid judge_agent.solution_combine {cfg.solution_combine!r}; expected one of {SOLUTION_COMBINE_MODES}"
            )
        return cfg

    def combine_rewards(self, programmatic: float, rubric: float) -> float:
        """Fold the two scores into one scalar per ``self.combine``."""
        if self.combine == "programmatic":
            return programmatic
        if self.combine == "rubric":
            return rubric
        if self.combine == "product":
            return programmatic * rubric
        w = float(self.rubric_weight)
        return (1.0 - w) * programmatic + w * rubric


class RubricJudge:
    """One judge run against a finished rollout inside its dataset environment.

    Owns the whole reward-time flow — workspace assembly, running the judge
    agent, reading verdicts back, copying the workspace out, scoring. The
    judge delivers exactly once; retrying is the caller's concern, not this
    module's. It is constructed and invoked by
    :meth:`DatasetEnvironment.calculate_reward`; ``env`` is the dataset env
    wrapper (for ``execute`` / ``copy_text_to`` / ``repo_path`` / logging),
    ``env.env`` the underlying pod the judge agent is given.
    """

    def __init__(self, env: DatasetEnvironment, rubrics: list[dict], config: RubricJudgeConfig):
        if not rubrics:
            raise ValueError(f"{env.instance_id}: RubricJudge needs at least one rubric")
        self.env = env
        self.rubrics = rubrics
        self.cfg = config
        self.solution = parse_solution(env.instance)
        self.solution_weight = parse_solution_weight(env.instance, config.solution_weight)

    @property
    def logger(self):
        return self.env.logger

    @property
    def instance_id(self) -> str:
        return self.env.instance_id

    @property
    def verdicts_path(self) -> str:
        return f"{self.cfg.workspace_dir}/verdicts.json"

    def run(
        self, *, model_patch: str = "", task: str = "", doer_agent: Any = None, doer_result: str = ""
    ) -> tuple[float, str, dict]:
        """Judge every rubric; return ``(rubric_reward, summary, extra)``.

        ``task`` / ``doer_agent`` / ``doer_result`` come from the runner via
        :meth:`DatasetEnvironment.attach_rollout`; all are optional (recalc
        mode has no doer — the judge then works from task + patch + repo state).
        A judge that never delivers verdicts is an infra fault, not a doer
        fault: the result carries ``error_category`` so an RL trainer masks the
        sequence instead of training on a false 0.
        """
        task = task or self.env.instance.get("problem_statement") or ""
        workspace_files = self._build_workspace(model_patch, task, doer_agent, doer_result)
        exit_status = self._run_judge(workspace_files, task)
        payload, parse_error = self._read_verdicts()
        self._copy_out_workspace()

        if payload is None:
            from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED

            return (
                0.0,
                f"judge failed to deliver verdicts: {parse_error}",
                {
                    "error_category": REWARD_TESTBED_CORRUPTED,
                    "error": "judge_verdicts_missing",
                    "judge_exit_status": exit_status,
                    "num_rubrics": len(self.rubrics),
                },
            )

        return self._score(payload, exit_status)

    # --- workspace -------------------------------------------------------------

    def _build_workspace(self, model_patch: str, task: str, doer_agent: Any, doer_result: str) -> list[dict]:
        """Materialise the judge's inputs inside the pod.

        Returns ``[{"path": ..., "desc": ...}, ...]`` for the system prompt —
        only files that actually exist are listed, so the judge is never told
        about a trajectory that isn't there (recalc mode has no doer agent).
        """
        ws = self.cfg.workspace_dir
        res = self.env.execute(f"mkdir -p {shlex.quote(ws)}")
        if res.get("returncode", 1) != 0:
            raise RuntimeError(f"Failed to create judge workspace {ws}: {res.get('output', '')}")

        passthrough = ("label", "category", "subcategory", "how_to_judge")
        rubrics_json = json.dumps(
            [
                {"id": i, "rubric": r["text"], **{k: r[k] for k in passthrough if k in r}}
                for i, r in enumerate(self.rubrics)
            ],
            ensure_ascii=False,
            indent=2,
        )

        files: list[dict] = []
        self.env.copy_text_to(task, f"{ws}/task.md")
        files.append({"path": f"{ws}/task.md", "desc": "the original task given to the doer"})
        self.env.copy_text_to(rubrics_json, f"{ws}/rubrics.json")
        files.append(
            {
                "path": f"{ws}/rubrics.json",
                "desc": "the rubrics to judge, [{id, rubric, category?, subcategory?, how_to_judge?}, ...]",
            }
        )
        if self.solution is not None:
            self.env.copy_text_to(json.dumps(self.solution, ensure_ascii=False, indent=2), f"{ws}/solution.json")
            files.append(
                {
                    "path": f"{ws}/solution.json",
                    "desc": "the graded solution rubric: {text, levels: [{index, score, description, how_to_judge?}, ...]} best-first",
                }
            )
        if doer_result:
            self.env.copy_text_to(doer_result, f"{ws}/response.md")
            files.append({"path": f"{ws}/response.md", "desc": "the doer's final response message"})
        self.env.copy_text_to(model_patch or "", f"{ws}/model_patch.diff")
        files.append(
            {"path": f"{ws}/model_patch.diff", "desc": "the doer's captured diff vs the base commit (may be empty)"}
        )
        for p in self._publish_trajectory(ws, doer_agent):
            files.append({"path": p, "desc": "the doer's raw session trajectory (can be long; read selectively)"})
        return files

    def _publish_trajectory(self, workspace_dir: str, agent: Any) -> list[str]:
        """Make the doer's trajectory readable inside the environment.

        Same three-tier logic as ``UserAgentDriver._publish_trajectory``:
        blackbox scaffolds already write their session log in the pod (point at
        it in place); native agents get their host-side msg file mirrored in;
        bare Agent implementations get their raw ``messages`` dumped as JSON.
        No doer agent (recalc mode) → nothing to publish.
        """
        if agent is None:
            return []
        from mimoagent.agents.user_agent import env_trajectory_files

        in_env = env_trajectory_files(agent)
        if in_env:
            return in_env
        msg_path = getattr(agent, "msg_path", None)
        if msg_path and Path(msg_path).exists():
            remote_path = f"{workspace_dir}/{Path(msg_path).name}"
            self.env.env.copy_to(str(msg_path), remote_path)
            return [remote_path]
        messages = getattr(agent, "messages", None)
        if messages:
            remote_path = f"{workspace_dir}/trajectory.json"
            self.env.copy_text_to(json.dumps(messages, ensure_ascii=False, indent=2), remote_path)
            return [remote_path]
        return []

    # --- judge run -------------------------------------------------------------

    def _run_judge(self, workspace_files: list[dict], task: str) -> str:
        """Build the judge DefaultAgent (own model, shared pod), run it once, return its exit status."""
        from mimoagent.agents.default import DefaultAgent
        from mimoagent.models import get_model

        cfg = self.cfg
        model = get_model(config=copy.deepcopy(cfg.model))
        log_ctx = get_log_context()
        judge = DefaultAgent(
            model=model,
            env=self.env.env,
            msg_path=log_ctx.agent_msg_path(cfg.name) if log_ctx else None,
            system_template=cfg.system_template,
            instance_template="{{task}}",
            tools=cfg.tools,
            step_limit=cfg.step_limit,
        )
        if log_ctx:
            log_ctx.register_agent(cfg.name, judge)

        prompt = Template(cfg.prompt_template).render(
            task=task,
            num_rubrics=len(self.rubrics),
            workspace_dir=cfg.workspace_dir,
            verdicts_path=self.verdicts_path,
            has_solution=self.solution is not None,
        )
        # Extra template vars for the system template (rendered on first run()).
        exit_status, _ = judge.run(
            prompt,
            workspace_dir=cfg.workspace_dir,
            workspace_files=workspace_files,
            verdicts_path=self.verdicts_path,
            repo_path=self.env.repo_path,
            has_solution=self.solution is not None,
        )
        if exit_status != judge.IDLE_STATUS:
            self.logger.warning(f"{self.instance_id}: judge agent exited with {exit_status!r}")
        return exit_status

    def _read_verdicts(self) -> tuple[dict | None, str]:
        """Read verdicts.json back from the pod. Returns (payload, error); the
        payload is the whole delivered object (``verdicts`` list, optional ``solution``)."""
        # Some images print profile banners for every login shell. Frame the file
        # so those unrelated bytes cannot make otherwise valid JSON unparseable.
        token = os.urandom(16).hex()
        begin_marker = f"__MIMO_VERDICTS_BEGIN_{token}__"
        end_marker = f"__MIMO_VERDICTS_END_{token}__"
        res = self.env.execute(
            f"printf '%s\\n' {shlex.quote(begin_marker)}; "
            f"cat {shlex.quote(self.verdicts_path)}; "
            "verdicts_read_rc=$?; "
            f"printf '\\n%s\\n' {shlex.quote(end_marker)}; "
            'exit "$verdicts_read_rc"'
        )
        output = res.get("output") or ""
        if res.get("returncode", 1) != 0:
            return None, "file missing or empty"
        begin = output.find(begin_marker)
        end = output.find(end_marker, begin + len(begin_marker))
        if begin < 0 or end < 0:
            return None, "could not isolate verdicts file from command output"
        raw = output[begin + len(begin_marker) : end].strip()
        if not raw:
            return None, "file missing or empty"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}"
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if not isinstance(verdicts, list):
            return None, 'top-level object must have a "verdicts" list'
        return data, ""

    def _copy_out_workspace(self) -> None:
        """Copy the judge workspace out of the pod next to the instance logs.

        Best-effort: the verdicts were already read back via ``cat``, so a
        failed copy loses debuggability, not the reward. docker/local base envs
        have no ``copy_out`` — skip (same guard as the blackbox agents).
        """
        log_ctx = get_log_context()
        pod = self.env.env
        if log_ctx is None or not hasattr(pod, "copy_out"):
            return
        try:
            pod.copy_out(self.cfg.workspace_dir, str(log_ctx.dir / "judge_workspace"))
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: judge workspace copy_out failed (non-fatal): {repr(e)}")

    # --- scoring ---------------------------------------------------------------

    def _score(self, payload: dict, judge_exit_status: str) -> tuple[float, str, dict]:
        """Align verdicts to rubrics by id and aggregate (see ``aggregate_rubric_scores``).

        Missing / invalid / null entries score 0. With a solution rubric the
        delivered level index is read from ``payload["solution"]["level"]``.
        """
        by_id: dict[int, dict] = {}
        for v in payload.get("verdicts") or []:
            if isinstance(v, dict) and isinstance(v.get("id"), int):
                by_id.setdefault(v["id"], v)

        raw_verdicts: dict[int, int] = {}
        reasons: dict[int, str] = {}
        for i in range(len(self.rubrics)):
            v = by_id.get(i)
            if v is None:
                raw_verdicts[i] = 0
                reasons[i] = "no verdict delivered for this rubric"
                continue
            val = v.get("verdict")
            raw_verdicts[i] = 1 if val == 1 and not isinstance(val, bool) else 0
            reasons[i] = v.get("reason", "") or ""

        level: int | None = None
        level_reason = ""
        if self.solution is not None:
            sol = payload.get("solution")
            if isinstance(sol, dict):
                lv = sol.get("level")
                if isinstance(lv, int) and not isinstance(lv, bool):
                    level = lv
                level_reason = sol.get("reason", "") or ""
            if level is None or not 0 <= level < len(self.solution["levels"]):
                level = None
                level_reason = level_reason or "no valid solution level delivered"

        agg = aggregate_rubric_scores(
            self.rubrics,
            raw_verdicts,
            solution=self.solution,
            solution_level=level,
            solution_weight=self.solution_weight,
            solution_combine=self.cfg.solution_combine,
        )

        results: list[dict] = []
        lines: list[str] = []
        for i, rubric in enumerate(self.rubrics):
            pr = agg["per_rubric"][i]
            # "rubric" stays the criterion text so downstream readers of
            # reward_extra_info keep working; the producer's fields ride alongside.
            entry = {"id": i, "rubric": rubric["text"], "verdict": pr["verdict"], "reason": reasons[i]}
            for key in ("label", "category", "subcategory"):
                if rubric.get(key):
                    entry[key] = rubric[key]
            results.append(entry)
            mark = "PASS" if pr["verdict"] else "FAIL"
            name = f"rubric[{i}]" + (f" {rubric['label']}" if rubric.get("label") else "")
            lines.append(f"[{mark}] {name}: {reasons[i]}" if reasons[i] else f"[{mark}] {name}")

        # Legacy shape "rubric verdicts (k/n passed)" when nothing beyond plain
        # rubrics is in play; solution / reward details are appended otherwise.
        # Solution placement leads (it is judged first and delivered first);
        # the per-rubric lines follow.
        head = f"rubric verdicts ({agg['num_satisfied']}/{agg['num_rubrics']} passed"
        extra: dict[str, Any] = {}
        if self.solution is not None:
            fold = "S·B" if agg["solution_combine"] == "product" else f"w={agg['solution_weight']:.2f}"
            head += (
                f"; solution level {agg['solution_level']} → S={agg['solution_score']:.2f}, "
                f"{fold}; rubric_reward={agg['rubric_reward']:.4f}"
            )
            lines.insert(0, f"[S] solution level {agg['solution_level']}: {level_reason}")
            extra.update(
                {
                    "solution_level": agg["solution_level"],
                    "solution_score": agg["solution_score"],
                    "solution_weight": agg["solution_weight"],
                    "solution_combine": agg["solution_combine"],
                    "solution_reason": level_reason,
                }
            )
        head += "):"
        test_output = head + "\n" + "\n".join(lines)

        extra.update(
            {
                "rubric_verdicts": results,
                "num_rubrics": len(results),
                "num_satisfied": agg["num_satisfied"],
                "behavior_score": agg["behavior_score"],
                "judge_exit_status": judge_exit_status,
            }
        )
        return agg["rubric_reward"], test_output, extra
