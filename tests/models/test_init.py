import os
from unittest.mock import patch

import pytest

from mimoagent.models import GlobalModelStats, get_model, get_model_class, get_model_name
from mimoagent.models.test_models import DeterministicModel


class TestGetModelName:
    # Common config used across tests - model_name should be direct, not nested under "model"
    CONFIG_WITH_MODEL_NAME = {"model_name": "config-model"}

    def test_input_model_name_takes_precedence(self):
        """Test that explicit input_model_name overrides all other sources."""
        with patch.dict(os.environ, {"MIMOAGENT_MODEL_NAME": "env-model"}):
            assert get_model_name("input-model", self.CONFIG_WITH_MODEL_NAME) == "input-model"

    def test_config_takes_precedence_over_env(self):
        """Test that config takes precedence over environment variable."""
        with patch.dict(os.environ, {"MIMOAGENT_MODEL_NAME": "env-model"}):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_env_var_fallback(self):
        """Test that environment variable is used when no config provided."""
        with patch.dict(os.environ, {"MIMOAGENT_MODEL_NAME": "env-model"}):
            assert get_model_name(None, {}) == "env-model"

    def test_config_fallback(self):
        """Test that config model name is used when input and env are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_raises_error_when_no_model_configured(self):
        """Test that ValueError is raised when no model is configured anywhere."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No model configured"):
                get_model_name(None, {})

            with pytest.raises(ValueError, match="No model configured"):
                get_model_name(None, None)


class TestGetModelClass:
    def test_protocol_routing(self):
        """The ``protocol`` field is the only routing input."""
        from mimoagent.models.anthropic import AnthropicModel
        from mimoagent.models.openai_chat import OpenAIChatModel
        from mimoagent.models.openai_responses import OpenAIResponsesModel

        assert get_model_class("any-name", {"protocol": "chat"}) == OpenAIChatModel
        assert get_model_class("any-name", {"protocol": "anthropic"}) == AnthropicModel
        assert get_model_class("any-name", {"protocol": "responses"}) == OpenAIResponsesModel

    def test_model_name_plays_no_role(self):
        """claude-ish names do NOT sniff-route; only protocol decides."""
        from mimoagent.models.openai_chat import OpenAIChatModel

        assert get_model_class("claude-opus-5", {"protocol": "chat"}) == OpenAIChatModel

    def test_missing_protocol_defaults_to_chat(self):
        """Blackbox configs omit protocol; the model is just a config carrier."""
        from mimoagent.models.openai_chat import OpenAIChatModel

        assert get_model_class("gpt-4") == OpenAIChatModel
        assert get_model_class("claude-opus-5", {}) == OpenAIChatModel

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="model.protocol must be one of"):
            get_model_class("gpt-4", {"protocol": "openai"})


class TestGetModel:
    def test_config_deep_copy(self):
        """Test that get_model preserves original config via deep copy."""
        original_config = {"model_kwargs": {"api_key": "original"}, "outputs": ["test"]}

        with patch("mimoagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = lambda **kwargs: DeterministicModel(outputs=["test"], model_name="test")
            get_model("test-model", original_config)
            assert original_config["model_kwargs"]["api_key"] == "original"
            assert "model_name" not in original_config

    def test_integration_with_compatible_model(self):
        """Test get_model works end-to-end with a model that handles extra kwargs."""
        with patch("mimoagent.models.get_model_class") as mock_get_class:

            def compatible_model(**kwargs):
                # Filter to only what DeterministicModel accepts, provide defaults
                config_args = {k: v for k, v in kwargs.items() if k in ["outputs", "model_name"]}
                if "outputs" not in config_args:
                    config_args["outputs"] = ["default"]
                return DeterministicModel(**config_args)

            mock_get_class.return_value = compatible_model
            model = get_model("test-model", {"outputs": ["hello"]})
            assert isinstance(model, DeterministicModel)
            assert model.config.outputs == ["hello"]
            assert model.config.model_name == "test-model"

    def test_config_api_key_used_when_no_env_var(self):
        """Test that config api_key is used when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Capture the arguments passed to the model constructor
            captured_kwargs = {}

            def mock_model_constructor(**kwargs):
                captured_kwargs.update(kwargs)
                return DeterministicModel(
                    outputs=kwargs.get("outputs", ["test"]),
                    model_name=kwargs.get("model_name", "test"),
                )

            with patch("mimoagent.models.get_model_class") as mock_get_class:
                mock_get_class.return_value = mock_model_constructor

                config = {"model_kwargs": {"api_key": "config-key"}, "outputs": ["test"]}
                get_model("test-model", config)

                assert captured_kwargs["model_kwargs"]["api_key"] == "config-key"

    def test_no_api_key_when_none_provided(self):
        """Test that no api_key is set when neither env var nor config provide one."""
        with patch.dict(os.environ, {}, clear=True):
            # Capture the arguments passed to the model constructor
            captured_kwargs = {}

            def mock_model_constructor(**kwargs):
                captured_kwargs.update(kwargs)
                return DeterministicModel(
                    outputs=kwargs.get("outputs", ["test"]),
                    model_name=kwargs.get("model_name", "test"),
                )

            with patch("mimoagent.models.get_model_class") as mock_get_class:
                mock_get_class.return_value = mock_model_constructor

                config = {"outputs": ["test"]}
                get_model("test-model", config)
                model_kwargs = captured_kwargs.get("model_kwargs", {})
                assert "api_key" not in model_kwargs


class TestGlobalModelStats:
    def test_prints_call_limit_when_set(self, capsys):
        """Call limit is printed when MIMOAGENT_GLOBAL_CALL_LIMIT is set."""
        with patch.dict(os.environ, {"MIMOAGENT_GLOBAL_CALL_LIMIT": "10"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit: 10" in captured.out

    def test_no_print_when_silent_startup_set(self, capsys):
        """Limits are not printed when MIMOAGENT_SILENT_STARTUP is set."""
        with patch.dict(
            os.environ,
            {"MIMOAGENT_GLOBAL_CALL_LIMIT": "10", "MIMOAGENT_SILENT_STARTUP": "1"},
            clear=True,
        ):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit" not in captured.out

    def test_no_print_when_no_limit_set(self, capsys):
        """Nothing is printed when no limit is set."""
        with patch.dict(os.environ, {}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit" not in captured.out

    def test_call_limit_enforced(self):
        """Exceeding the call limit raises."""
        from mimoagent.models import TokenStats

        # The limit check fires on the call that *reaches* it: with limit=2 the
        # first add passes (1 < 2), the second raises (2 reaches the limit).
        with patch.dict(os.environ, {"MIMOAGENT_GLOBAL_CALL_LIMIT": "2", "MIMOAGENT_SILENT_STARTUP": "1"}, clear=True):
            stats = GlobalModelStats()
            stats.add(TokenStats())
            with pytest.raises(RuntimeError, match="Global call limit exceeded"):
                stats.add(TokenStats())
