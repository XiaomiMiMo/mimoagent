"""``make_dataset_env`` image resolution and kwarg routing, without a backend."""

from dataclasses import dataclass, field

import pytest

from mimoagent.environments.utils import make_dataset_env, resolve_docker_image


@dataclass
class _FakeConfig:
    image: str
    cwd: str = "/testbed"
    kubeconfig: str | None = None
    node_selector: dict | None = None
    extra: dict = field(default_factory=dict)


class _FakeEnv:
    """Minimal base environment honouring the ``(*, config_class=..., **kwargs)`` contract."""

    created: list["_FakeEnv"] = []

    def __init__(self, *, config_class: type = _FakeConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.instance_id = None
        _FakeEnv.created.append(self)

    def start(self):
        pass

    def execute(self, command, cwd="", timeout=None):
        return {"output": "", "returncode": 0}

    def copy_to(self, src, dest, **kwargs):
        pass

    def get_template_vars(self):
        return {}


_HF_ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "version": "3.0",
    "FAIL_TO_PASS": "[]",
    "PASS_TO_PASS": "[]",
    "base_commit": "abc",
    "problem_statement": "...",
}


def test_example_row_without_docker_image_gets_the_official_image():
    assert resolve_docker_image(_HF_ROW, "example") == "example/test-image.django_1776_django-11099:latest"


def test_explicit_docker_image_is_used_verbatim():
    inst = {**_HF_ROW, "docker_image": "ghcr.io/me/custom:1"}
    assert resolve_docker_image(inst, "example") == "ghcr.io/me/custom:1"


def test_image_prefix_is_prepended_exactly_once():
    inst = {**_HF_ROW, "docker_image": "example/test-image.x:latest"}
    prefixed = resolve_docker_image(inst, "example", "registry.example.com/mirror/")
    assert prefixed == "registry.example.com/mirror/example/test-image.x:latest"
    assert (
        resolve_docker_image({**inst, "docker_image": prefixed}, "example", "registry.example.com/mirror") == prefixed
    )


def test_dataset_without_derivable_image_fails_loudly():
    with pytest.raises(ValueError, match="no docker_image"):
        resolve_docker_image({"instance_id": "x", "dataset_type": "generic"}, "generic")


def test_make_dataset_env_routes_kwargs_and_never_leaks_image_prefix():
    _FakeEnv.created.clear()
    env = make_dataset_env(
        _HF_ROW,
        environment_class=_FakeEnv,
        image_prefix="registry.example.com",
        git_leak_prevention="none",
        unknown_key="dropped-with-a-warning",
    )
    base = _FakeEnv.created[-1]
    assert base.config.image == "registry.example.com/example/test-image.django_1776_django-11099:latest"
    assert base.instance_id == "django__django-11099"
    assert env.git_leak_prevention == "none"
    assert not hasattr(base.config, "image_prefix")


def test_per_instance_routing_overrides_reach_the_base_env():
    _FakeEnv.created.clear()
    inst = {**_HF_ROW, "routing": {"kubeconfig": "/etc/k8s/other", "node_selector": {"pool": "b"}}}
    make_dataset_env(inst, environment_class=_FakeEnv)
    base = _FakeEnv.created[-1]
    assert base.config.kubeconfig == "/etc/k8s/other"
    assert base.config.node_selector == {"pool": "b"}
