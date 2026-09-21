"""Dataset environment registry.

To add a dataset:
1. Implement a :class:`DatasetEnvironment` subclass in its own module here.
2. Register it in ``DATASET_REGISTRY``.
3. Add a detection rule to ``detect_dataset_type`` (order matters — earlier
   rules win; explicit ``dataset_type`` fields beat image-substring matches).

Rubric judging is NOT a dataset: any registered dataset's instances may carry
``rubric.rubrics``, and the base class runs the judge at reward time when the
yaml also sets ``judge_agent`` (see ``base.REWARD_MODES`` /
``mimoagent.environments.rubric_judge``).
"""

from mimoagent import Environment
from mimoagent.environments.datasets.base import GIT_LEAK_PREVENTION_MODES, REWARD_MODES, DatasetEnvironment
from mimoagent.environments.datasets.arvo import ARVOEnvironment
from mimoagent.environments.datasets.deepswe import DeepSWEEnvironment
from mimoagent.environments.datasets.generic import GenericGitEnvironment
from mimoagent.environments.datasets.opensource_code import OpenSourceCodeEnvironment
from mimoagent.environments.rubric_judge import RubricJudgeConfig

DATASET_REGISTRY: dict[str, type[DatasetEnvironment]] = {
    "arvo": ARVOEnvironment,
    "deepswe": DeepSWEEnvironment,
    "generic": GenericGitEnvironment,
    "opensource-code": OpenSourceCodeEnvironment,
}


def detect_dataset_type(instance: dict) -> str:
    """Resolve the dataset type for an instance.

    An explicit ``instance['dataset_type']`` matching the registry always wins.
    Image-substring heuristics handle the well-known public image namespaces.
    Unknown instances raise — silently grading with the wrong verifier is worse
    than failing.
    """
    declared = instance.get("dataset_type")
    if declared in DATASET_REGISTRY:
        return declared

    image = instance.get("docker_image", "")
    if declared == "deepswe" or "swe-bench-202605" in image:
        return "deepswe"

    raise ValueError(
        f"Cannot detect dataset type for instance {instance.get('instance_id')!r} "
        f"(dataset_type={declared!r}, docker_image={image!r}). "
        f"Known types: {sorted(DATASET_REGISTRY)}"
    )


def create_from_registry(
    dataset_type: str,
    base_env: Environment,
    instance: dict,
    anti_hack_cleanup: bool | None = None,
    git_leak_prevention: str | None = None,
    judge_agent: dict | None = None,
    reward_mode: str | None = None,
) -> DatasetEnvironment:
    """Look up ``dataset_type`` in ``DATASET_REGISTRY`` and wrap ``base_env`` with it.

    Low-level constructor used by :func:`mimoagent.environments.utils.make_dataset_env`,
    which is the public entry point (it also builds the base env and detects the type).

    ``anti_hack_cleanup`` (from the yaml ``anti_hack_cleanup`` key) overrides each
    env's class default for the build-env anti-hack cleanup when not None.
    ``git_leak_prevention`` (from the yaml ``git_leak_prevention`` key; one of
    ``GIT_LEAK_PREVENTION_MODES``) overrides the git leak-prevention mode.
    ``judge_agent`` (from the yaml ``judge_agent`` key) is the rubric judge agent
    config; ``reward_mode`` (from the yaml ``reward_mode`` key; one of
    ``REWARD_MODES``) decides whether the judge and/or the dataset's verifier run
    at reward time. All are set after construction so the subclass constructors
    keep their uniform ``(base_env, instance)`` signature. The judge config and
    the reward plan are validated here so a misconfigured run fails before any
    pod is started.
    """
    try:
        env_class = DATASET_REGISTRY[dataset_type]
    except KeyError:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}. Known: {sorted(DATASET_REGISTRY)}")
    env = env_class(base_env, instance)
    if anti_hack_cleanup is not None:
        env.anti_hack_cleanup = anti_hack_cleanup
    if git_leak_prevention is not None:
        if git_leak_prevention not in GIT_LEAK_PREVENTION_MODES:
            raise ValueError(
                f"Invalid git_leak_prevention {git_leak_prevention!r}; expected one of {GIT_LEAK_PREVENTION_MODES}"
            )
        env.git_leak_prevention = git_leak_prevention
    if judge_agent is not None:
        RubricJudgeConfig.from_dict(judge_agent)  # validate the block up front
        env.judge_agent_config = judge_agent
    if reward_mode is not None:
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Invalid reward_mode {reward_mode!r}; expected one of {REWARD_MODES}")
        env.reward_mode = reward_mode
    env.resolve_reward_plan()  # explicit rubric/both modes fail loudly when their inputs are missing
    return env


__all__ = [
    "DATASET_REGISTRY",
    "ARVOEnvironment",
    "DeepSWEEnvironment",
    "GIT_LEAK_PREVENTION_MODES",
    "GenericGitEnvironment",
    "OpenSourceCodeEnvironment",
    "REWARD_MODES",
    "DatasetEnvironment",
    "create_from_registry",
    "detect_dataset_type",
]
