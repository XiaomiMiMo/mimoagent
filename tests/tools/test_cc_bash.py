"""Tests for the Bash tool's per-call timeout resolution and config bounds.

The timeout-resolution logic gates pod lifetime during RL rollouts: too small
and a cold compile can never finish; too large and one call ties up a pod past
the training-throughput ceiling. It's pure arithmetic on a model-supplied value
(ms) against a seconds-based config, so it's tested in isolation rather than
through a live pod.
"""

import pytest

from mimoagent.tools.base import ToolException
from mimoagent.tools.cc.bash import BashTool, BashToolConfig

# ---- _resolve_timeout: omitted / malformed → config default -----------------


def test_resolve_timeout_omitted_uses_default():
    assert BashTool._resolve_timeout(None, default_s=60, max_s=300) == 60


@pytest.mark.parametrize("bad", ["abc", "", {}, [], object()])
def test_resolve_timeout_non_numeric_falls_back_to_default(bad):
    assert BashTool._resolve_timeout(bad, default_s=60, max_s=300) == 60


# ---- _resolve_timeout: ms→s conversion --------------------------------------


def test_resolve_timeout_converts_ms_to_s():
    # The schema documents milliseconds; 90000ms must become 90s, not 90000s.
    assert BashTool._resolve_timeout(90000, default_s=60, max_s=300) == 90


def test_resolve_timeout_accepts_numeric_string():
    assert BashTool._resolve_timeout("90000", default_s=60, max_s=300) == 90


# ---- _resolve_timeout: clamping to [1, max_s] -------------------------------


def test_resolve_timeout_clamps_to_ceiling():
    # The trajectory bug: model requested 300000ms (=300s) under a 300s ceiling
    # — allowed; 600000ms (=600s) must be capped at 300s.
    assert BashTool._resolve_timeout(300_000, default_s=60, max_s=300) == 300
    assert BashTool._resolve_timeout(600_000, default_s=60, max_s=300) == 300


def test_resolve_timeout_sub_second_floors_to_one():
    # A model emitting seconds-as-ms (e.g. 300 meaning "300s") would otherwise
    # round down to 0s and the in-pod `timeout 0` would never fire. Floor at 1s.
    assert BashTool._resolve_timeout(300, default_s=60, max_s=300) == 1
    assert BashTool._resolve_timeout(0, default_s=60, max_s=300) == 1


# ---- BashToolConfig: contradictory bounds fail fast -------------------------


def test_config_rejects_ceiling_below_default():
    with pytest.raises(ValueError, match="must be >="):
        BashToolConfig(timeout=300, max_timeout=60)


def test_config_defaults_are_consistent():
    cfg = BashToolConfig()
    assert cfg.max_timeout >= cfg.timeout


# ---- description / schema reflect the configured bounds ----------------------


def test_description_advertises_configured_bounds():
    tool = BashTool({"timeout": 60, "max_timeout": 300})
    desc = tool.description
    assert "300000ms" in desc  # ceiling, in ms
    assert "60000ms" in desc  # default, in ms
    # The stale hardcoded contract must be gone.
    assert "600000ms" not in desc


def test_schema_timeout_advertises_ceiling():
    tool = BashTool({"timeout": 60, "max_timeout": 300})
    timeout_desc = tool.get_function_parameters()["properties"]["timeout"]["description"]
    assert "300000" in timeout_desc


# ---- _extract_command (unchanged behavior, guards the happy path) ------------


def test_extract_command_strips_and_returns():
    assert BashTool._extract_command({"command": "  ls -la  "}) == "ls -la"


def test_extract_command_rejects_empty():
    with pytest.raises(ToolException):
        BashTool._extract_command({"command": "   "})
