import logging
import time

from mimoagent.models import GlobalModelStats
from mimoagent.models.test_models import DeterministicModel, DeterministicModelConfig


def test_basic_functionality_and_token_tracking():
    """Test basic model functionality, token tracking, and default configuration."""
    stats = GlobalModelStats()
    model = DeterministicModel(outputs=["Hello", "World"], shared_stats=stats)

    assert model.query([{"role": "user", "content": "test"}]) == {"content": "Hello"}
    assert model.n_calls == 1
    assert model.token_stats.input_tokens == model.config.tokens_per_call
    assert stats.n_calls == 1

    assert model.query([{"role": "user", "content": "test"}]) == {"content": "World"}
    assert model.n_calls == 2
    assert stats.n_calls == 2


def test_multiple_models_share_injected_stats():
    """An injected aggregator accumulates calls across model instances."""
    stats = GlobalModelStats()
    model1 = DeterministicModel(outputs=["Response1"], shared_stats=stats)
    model2 = DeterministicModel(outputs=["Response2"], shared_stats=stats)

    assert model1.query([{"role": "user", "content": "test"}]) == {"content": "Response1"}
    assert model2.query([{"role": "user", "content": "test"}]) == {"content": "Response2"}
    assert stats.n_calls == 2


def test_models_without_injected_stats_are_isolated():
    """Without injection, models share no aggregator state."""
    stats = GlobalModelStats()
    model_with = DeterministicModel(outputs=["A"], shared_stats=stats)
    model_without = DeterministicModel(outputs=["B"])

    assert model_with.query([{"role": "user", "content": "test"}]) == {"content": "A"}
    assert model_without.query([{"role": "user", "content": "test"}]) == {"content": "B"}
    assert stats.n_calls == 1
    assert model_without.n_calls == 1


def test_config_dataclass():
    """Test DeterministicModelConfig with custom values."""
    config = DeterministicModelConfig(outputs=["Test"], model_name="custom", tokens_per_call=42)

    assert config.tokens_per_call == 42
    assert config.model_name == "custom"

    model = DeterministicModel(**config.__dict__)
    assert model.config.tokens_per_call == 42


def test_sleep_and_warning_commands(caplog):
    """Test special /sleep and /warning command handling."""
    # Test sleep command - processes sleep then returns actual output (counts as 1 call)
    model = DeterministicModel(outputs=["/sleep0.1", "After sleep"])
    start_time = time.time()
    assert model.query([{"role": "user", "content": "test"}]) == {"content": "After sleep"}
    assert time.time() - start_time >= 0.1
    assert model.n_calls == 1  # Sleep no longer counts as separate call

    # Test warning command - processes warning then returns actual output (counts as 1 call)
    model2 = DeterministicModel(outputs=["/warningTest message", "After warning"])
    with caplog.at_level(logging.WARNING):
        assert model2.query([{"role": "user", "content": "test"}]) == {"content": "After warning"}
    assert model2.n_calls == 1  # Warning no longer counts as separate call
    assert "Test message" in caplog.text
