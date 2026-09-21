"""The pod manifest KubernetesEnvironment submits, asserted without a cluster.

``_build_pod_body`` is pure: it reads the config and returns the dict that
``create_namespaced_pod`` receives, so the defaults (no scheduling constraints,
no host networking, no extra volumes) and the opt-in knobs can be pinned here.
"""

from unittest.mock import patch

import pytest

from mimoagent.environments.kubernetes import KubernetesEnvironment, KubernetesEnvironmentConfig


def _make_env(**kwargs) -> KubernetesEnvironment:
    with (
        patch("mimoagent.environments.kubernetes.config.load_kube_config"),
        patch("mimoagent.environments.kubernetes.client.ApiClient"),
        patch("mimoagent.environments.kubernetes.client.CoreV1Api"),
    ):
        env = KubernetesEnvironment(image="test:latest", **kwargs)
    env.pod_name = "mimoagent-test"
    return env


def test_default_spec_has_no_scheduling_constraints_or_extras():
    body = _make_env()._build_pod_body({})
    spec = body["spec"]
    assert body["metadata"]["labels"] == {"app": "mimoagent"}
    assert body["metadata"]["annotations"] == {}
    assert spec["hostNetwork"] is False
    assert spec["automountServiceAccountToken"] is False
    for absent in ("nodeSelector", "tolerations", "imagePullSecrets", "initContainers", "volumes"):
        assert absent not in spec
    main = spec["containers"][0]
    assert main["name"] == "main"
    assert main["image"] == "test:latest"
    assert "securityContext" not in main
    assert "volumeMounts" not in main


def test_env_vars_land_in_the_main_container():
    body = _make_env()._build_pod_body({"FOO": "bar", "N": 1})
    assert body["spec"]["containers"][0]["env"] == [{"name": "FOO", "value": "bar"}, {"name": "N", "value": "1"}]


def test_forwarded_host_env_loses_to_configured_env(monkeypatch):
    monkeypatch.setenv("MIMOAGENT_TEST_FWD", "from-host")
    monkeypatch.setenv("MIMOAGENT_TEST_BOTH", "from-host")
    env = _make_env(
        forward_env=["MIMOAGENT_TEST_FWD", "MIMOAGENT_TEST_BOTH", "MIMOAGENT_TEST_UNSET"],
        env={"MIMOAGENT_TEST_BOTH": "from-config"},
    )
    assert env._build_env_vars() == {"MIMOAGENT_TEST_FWD": "from-host", "MIMOAGENT_TEST_BOTH": "from-config"}


def test_scheduling_and_registry_knobs_are_passed_through():
    toleration = {"key": "dedicated", "operator": "Equal", "value": "agents", "effect": "NoSchedule"}
    env = _make_env(
        node_selector={"pool": "agents"},
        tolerations=[toleration],
        image_pull_secrets=["regcred"],
        host_network=True,
        labels={"team": "swe"},
        annotations={"owner": "ci"},
    )
    body = env._build_pod_body({})
    spec = body["spec"]
    assert spec["nodeSelector"] == {"pool": "agents"}
    assert spec["tolerations"] == [toleration]
    assert spec["imagePullSecrets"] == [{"name": "regcred"}]
    assert spec["hostNetwork"] is True
    assert body["metadata"]["labels"] == {"team": "swe"}
    assert body["metadata"]["annotations"] == {"owner": "ci"}


def test_resources_come_from_the_config():
    env = _make_env(cpu_request="2", memory_request="4Gi", cpu_limit="8", memory_limit="16Gi")
    resources = env._build_pod_body({})["spec"]["containers"][0]["resources"]
    assert resources == {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "8", "memory": "16Gi"}}


def test_namespace_defaults_from_the_environment(monkeypatch):
    monkeypatch.delenv("K8S_NAMESPACE", raising=False)
    assert KubernetesEnvironmentConfig(image="x").namespace == "default"
    monkeypatch.setenv("K8S_NAMESPACE", "agents")
    assert KubernetesEnvironmentConfig(image="x").namespace == "agents"
    assert KubernetesEnvironmentConfig(image="x", namespace="explicit").namespace == "explicit"


def test_invalid_label_values_are_sanitized():
    env = _make_env(labels={"instance": "django__django-11099/rollout 1"})
    assert env.config.labels["instance"] == "django__django-11099-rollout-1"


@pytest.mark.parametrize("value", ["a" * 70, "-leading", "trailing-", "mid dle"])
def test_sanitized_label_values_are_valid_k8s_labels(value):
    sanitized = KubernetesEnvironment._sanitize_label_value(value)
    assert len(sanitized) <= 63
    assert sanitized == "" or (sanitized[0].isalnum() and sanitized[-1].isalnum())
