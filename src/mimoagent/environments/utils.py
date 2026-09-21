import inspect

from mimoagent import Environment
from mimoagent.environments import get_environment_class
from mimoagent.environments.datasets import (
    DATASET_REGISTRY,
    DatasetEnvironment,
    create_from_registry,
    detect_dataset_type,
)
from mimoagent.utils.dataclass_kwargs import filter_dataclass_kwargs


def filter_env_kwargs(base_env_class: type, env_kwargs: dict) -> dict:
    """Keep only kwargs the environment's config dataclass accepts.

    The base env classes take ``**kwargs`` and forward them into their config
    dataclass; passing a kwarg the dataclass doesn't define raises TypeError.
    Unknown keys are warned about and dropped — a typo in the yaml should be
    visible, not silently discarded.
    """
    # Convention: every environment constructor is (*, config_class=<Config>,
    # **kwargs) and forwards kwargs into that config dataclass. Resolve the
    # dataclass from the ``config_class`` keyword default.
    sig = inspect.signature(base_env_class.__init__)
    config_param = sig.parameters.get("config_class")
    config_cls = config_param.default if config_param is not None else None
    if config_cls is None:
        return env_kwargs  # cannot introspect — pass through unchanged
    return filter_dataclass_kwargs(config_cls, env_kwargs)


def resolve_docker_image(instance: dict, dataset_type: str, image_prefix: str | None = None) -> str:
    """Return the image to run ``instance`` in.

    The instance's own ``docker_image`` wins; otherwise the dataset class may
    derive one from other instance fields. ``image_prefix`` — a
    registry or mirror path such as ``registry.example.com/mirror`` — is
    prepended when the image does not already start with it.
    """
    docker_image = instance.get("docker_image") or DATASET_REGISTRY[dataset_type].default_docker_image(instance)
    if not docker_image:
        raise ValueError(
            f"Instance {instance.get('instance_id')!r} has no docker_image and dataset type "
            f"{dataset_type!r} cannot derive one"
        )
    if image_prefix:
        prefix = image_prefix.rstrip("/")
        if not docker_image.startswith(prefix + "/"):
            docker_image = f"{prefix}/{docker_image}"
    return docker_image


def make_dataset_env(
    instance: dict,
    *,
    anti_hack_cleanup: bool | None = None,
    git_leak_prevention: str | None = None,
    judge_agent: dict | None = None,
    reward_mode: str | None = None,
    image_prefix: str | None = None,
    **kwargs,
) -> DatasetEnvironment:
    """
    This create a base_env instance and a dataset-specific env instance
    Does not touch the actual env starting process, which is handled by
    dataset_env.setup_environment()

    ``anti_hack_cleanup`` (bool, build-env anti-hack cleanup),
    ``git_leak_prevention`` ("hide"/"strip"/"none", see base.GIT_LEAK_PREVENTION_MODES),
    ``judge_agent`` (dict, rubric judge agent config — runs on any dataset whose
    instances carry ``rubric.rubrics``), ``reward_mode`` ("auto"/"programmatic"/
    "rubric"/"both", see base.REWARD_MODES) and ``image_prefix`` (registry mirror
    prepended to every image, see :func:`resolve_docker_image`) are keyword-only
    overrides written in the yaml ``environment`` block. They are declared as
    explicit keyword params (not swept into ``**kwargs``) so that when callers
    splat the ``environment`` block (``make_dataset_env(instance, **env_config)``)
    they are captured here rather than reaching the base env's kwargs and getting
    dropped by ``filter_env_kwargs``.
    """
    # kwargs are forwarded to the selected base environment implementation
    dataset_type = detect_dataset_type(instance)
    docker_image = resolve_docker_image(instance, dataset_type, image_prefix)

    environment_spec = kwargs.pop("environment_class", None) or "kubernetes"

    base_env_class: type[Environment]
    if isinstance(environment_spec, str):
        base_env_class = get_environment_class(environment_spec)
    else:
        base_env_class = environment_spec

    if not callable(base_env_class):
        raise TypeError(f"Invalid environment_class: {environment_spec!r}")

    env_kwargs = {**kwargs, "image": docker_image}

    if "cwd" in instance:
        env_kwargs["cwd"] = instance["cwd"]
    elif dataset_type == "deepswe":
        env_kwargs["cwd"] = "/app"

    # Per-instance routing overrides (e.g. a dataset spread over several clusters).
    routing = instance.get("routing") or {}
    if routing.get("kubeconfig"):
        env_kwargs["kubeconfig"] = routing["kubeconfig"]
    if routing.get("node_selector"):
        env_kwargs["node_selector"] = routing["node_selector"]

    env_kwargs = filter_env_kwargs(base_env_class, env_kwargs)
    base_env = base_env_class(**env_kwargs)

    if hasattr(base_env, "instance_id"):
        base_env.instance_id = instance.get("instance_id")

    return create_from_registry(
        dataset_type,
        base_env,
        instance,
        anti_hack_cleanup,
        git_leak_prevention,
        judge_agent,
        reward_mode=reward_mode,
    )
