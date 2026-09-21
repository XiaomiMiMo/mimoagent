import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from mimoagent.models import GlobalModelStats, TokenStats


@dataclass
class DeterministicModelConfig:
    outputs: list[str]
    model_name: str = "deterministic"
    tokens_per_call: int = 100


class DeterministicModel:
    def __init__(self, *, shared_stats: GlobalModelStats | None = None, **kwargs):
        """
        Initialize with a list of outputs to return in sequence.
        """
        self.config = DeterministicModelConfig(**kwargs)
        self.shared_stats = shared_stats
        self.current_index = -1
        self.token_stats = TokenStats()
        self.n_calls = 0

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if "/sleep" in output:
            print("SLEEPING")
            time.sleep(float(output.split("/sleep")[1]))
            return self.query(messages, **kwargs)
        if "/warning" in output:
            logging.warning(output.split("/warning")[1])
            return self.query(messages, **kwargs)
        self.n_calls += 1
        # Simulate token usage
        token_count = TokenStats(
            input_tokens=self.config.tokens_per_call,
            output_tokens=self.config.tokens_per_call // 2,
        )
        self.token_stats.input_tokens += token_count.input_tokens
        self.token_stats.output_tokens += token_count.output_tokens
        if self.shared_stats is not None:
            self.shared_stats.add(token_count)
        return {"content": output}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "input_tokens": self.token_stats.input_tokens,
            "output_tokens": self.token_stats.output_tokens,
            "total_tokens": self.token_stats.total_tokens,
        }
