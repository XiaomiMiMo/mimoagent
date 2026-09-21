"""The bash-only tool has an isolated prompt and shared bash execution behavior."""

from pathlib import Path

import yaml

from mimoagent.tools import ToolRegistry


class _Env:
    def __init__(self, result=None):
        self.result = result or {"output": "ok\n", "returncode": 0}
        self.calls = []

    def execute(self, command, cwd="", timeout=None):
        self.calls.append((command, cwd, timeout))
        return self.result


def test_bash_only_is_registered_with_its_own_name_and_prompt():
    registry = ToolRegistry.from_config([{"tool": "bash-only"}])

    assert registry.list_tools() == ["bash"]
    definition = registry.get_function_definitions()[0]["function"]
    assert definition["name"] == "bash"
    assert "the only available tool" in definition["description"]
    assert "Use read" not in definition["description"]
    assert "Use edit" not in definition["description"]
    assert "Use write" not in definition["description"]


def test_bash_only_reuses_bash_timeout_and_result_handling():
    registry = ToolRegistry.from_config([{"tool": "bash-only", "config": {"timeout": 45}}])
    env = _Env()

    result = registry.get("bash").execute({"command": "cat file.txt"}, {"env": env})

    assert result.to_dict() == {
        "output": "ok\n",
        "success": True,
        "metadata": {"returncode": 0},
        "media": [],
    }
    assert env.calls == [("cat file.txt", "", 45)]


def test_bash_only_defaults_match_the_profile():
    registry = ToolRegistry.from_config([{"tool": "bash-only"}])
    tool = registry.get("bash")

    assert tool.config.timeout == 60
    assert tool.config.max_timeout == 300


def test_regular_bash_prompt_and_registration_are_unchanged():
    registry = ToolRegistry.from_config([{"tool": "bash"}])
    definition = registry.get_function_definitions()[0]["function"]

    assert definition["name"] == "bash"
    # the regular tool keeps its full-catalogue prompt (concurrency note included)
    assert "Every tool call in a message runs concurrently" in definition["description"]


def test_bash_only_example_matches_the_default_profile():
    config_path = Path(__file__).parents[2] / "example_configs" / "default-bash-only.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["agent"]["type"] == "bashonly-agent"
    assert config["agent"]["tools"] == [{"tool": "bash-only", "config": {"timeout": 60, "max_timeout": 300}}]
    assert "You are an agent, your current working directory is {{cwd}}." in config["agent"]["system_template"]
    assert config["agent"]["instance_template"] == "Fix the following issue:\n\n{{task}}\n"
    assert config["agent"]["step_limit"] == 500
