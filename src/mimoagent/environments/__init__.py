"""Environment implementations for mimoagent."""

import copy
import importlib

from mimoagent import Environment


class TransportError(RuntimeError):
    """Pod exec stream transport failure — unrecoverable infrastructure issue.

    Raised by KubernetesEnvironment.execute() when raise_on_transport_error=True
    and the exec stream cannot be established or is interrupted mid-transfer.
    """


_ENVIRONMENT_MAPPING = {
    "docker": "mimoagent.environments.docker.DockerEnvironment",
    "kubernetes": "mimoagent.environments.kubernetes.KubernetesEnvironment",
    "local": "mimoagent.environments.local.LocalEnvironment",
    "cube": "mimoagent.environments.cube.CubeEnvironment",
    "modal": "mimoagent.environments.modal.ModalEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown environment type: {spec} (resolved to {full_path}, available: {_ENVIRONMENT_MAPPING})"
        raise ValueError(msg)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    env = get_environment_class(environment_class)(**config)
    env.start()
    return env
