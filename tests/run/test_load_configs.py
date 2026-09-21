"""``${VAR}`` expansion when batch.py loads yaml configs."""

import pytest

from mimoagent.config import expand_env_vars
from mimoagent.run.extra.batch import load_configs


def test_expands_nested_values_and_defaults(monkeypatch):
    monkeypatch.setenv("MIMOAGENT_TEST_KEY", "sk-test")
    monkeypatch.delenv("MIMOAGENT_TEST_UNSET", raising=False)
    config = {
        "model": {
            "model_kwargs": {
                "api_key": "${MIMOAGENT_TEST_KEY}",
                "base_url": "${MIMOAGENT_TEST_UNSET:-http://localhost:8000/v1}",
            },
            "names": ["${MIMOAGENT_TEST_KEY}-a", "plain"],
        },
        "agent": {"step_limit": 5, "system_template": "cost is $5 and ${literal} stays escaped as $${literal}"},
    }
    monkeypatch.setenv("literal", "expanded")
    out = expand_env_vars(config)
    assert out["model"]["model_kwargs"] == {"api_key": "sk-test", "base_url": "http://localhost:8000/v1"}
    assert out["model"]["names"] == ["sk-test-a", "plain"]
    assert out["agent"]["step_limit"] == 5
    assert out["agent"]["system_template"] == "cost is $5 and expanded stays escaped as ${literal}"


def test_unset_variable_without_default_names_the_config_path(monkeypatch):
    monkeypatch.delenv("MIMOAGENT_TEST_UNSET", raising=False)
    with pytest.raises(ValueError, match=r"MIMOAGENT_TEST_UNSET.*model\.model_kwargs\.api_key"):
        expand_env_vars({"model": {"model_kwargs": {"api_key": "${MIMOAGENT_TEST_UNSET}"}}})


def test_load_configs_expands_after_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMOAGENT_TEST_BASE", "http://gw:1/v1")
    cfg = tmp_path / "a.yaml"
    cfg.write_text("model:\n  model_name: m\n  model_kwargs:\n    base_url: ${MIMOAGENT_TEST_BASE}\n")
    [(name, config)] = load_configs(
        [cfg], override_config='{"model": {"model_kwargs": {"api_key": "${MIMOAGENT_TEST_BASE}"}}}'
    )
    assert name == "a"
    assert config["model"]["model_kwargs"] == {"base_url": "http://gw:1/v1", "api_key": "http://gw:1/v1"}
