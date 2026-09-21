"""Configuration files and utilities for mimoagent."""

import os
import re
from pathlib import Path
from typing import Any

builtin_config_dir = Path(__file__).parent

# ``${NAME}`` or ``${NAME:-default}`` inside yaml string values. ``$${`` is a
# literal ``${``. Bare ``$NAME`` is deliberately NOT expanded: templates and
# shell snippets in configs use ``$`` freely.
_ENV_REF = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: Any, *, path: str = "") -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in config strings.

    Lets example configs reference credentials and endpoints from the
    environment (``api_key: "${OPENAI_API_KEY}"``) instead of embedding them.
    An unset variable without a default raises ``ValueError`` naming the
    config path (e.g. ``model.model_kwargs.api_key``) so a missing secret fails
    before any environment is started.
    """
    if isinstance(value, dict):
        return {k: expand_env_vars(v, path=f"{path}.{k}" if path else str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if not isinstance(value, str) or "${" not in value:
        return value

    def replace(match: re.Match) -> str:
        escaped, name, default = match.group(1), match.group(2), match.group(3)
        if escaped:
            return match.group(0)[1:]  # "$${X}" -> "${X}"
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        raise ValueError(
            f"config references the unset environment variable {name!r} at {path or '<root>'}; "
            f"export it or use ${{{name}:-default}}"
        )

    return _ENV_REF.sub(replace, value)


def get_config_path(config_spec: str | Path) -> Path:
    """Get the path to a config file."""
    config_spec = Path(config_spec)
    if config_spec.suffix != ".yaml":
        config_spec = config_spec.with_suffix(".yaml")
    candidates = [
        Path(config_spec),
        Path(os.getenv("MIMOAGENT_CONFIG_DIR", ".")) / config_spec,
        builtin_config_dir / config_spec,
        builtin_config_dir / "extra" / config_spec,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find config file for {config_spec} (tried: {candidates})")


__all__ = ["builtin_config_dir", "expand_env_vars", "get_config_path"]
