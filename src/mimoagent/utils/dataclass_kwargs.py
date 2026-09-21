"""Tolerant kwargs → config-dataclass construction.

Configs come from user-edited yaml that often outlives the code (old keys like
``cost_limit`` linger in launch scripts and override-config blobs). Unknown
keys should be warned about and dropped, not crash the run with a TypeError.
"""

import dataclasses

from mimoagent.utils.log import get_logger


def filter_dataclass_kwargs(config_class: type, kwargs: dict) -> dict:
    """Return ``kwargs`` restricted to ``config_class``'s dataclass fields.

    Warns (once per call) about any dropped keys so yaml typos stay visible.
    If ``config_class`` is not a dataclass, returns ``kwargs`` unchanged.
    """
    if not dataclasses.is_dataclass(config_class):
        return kwargs
    valid = {f.name for f in dataclasses.fields(config_class)}
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        get_logger().warning(f"Ignoring config keys not accepted by {config_class.__name__}: {dropped}")
    return {k: v for k, v in kwargs.items() if k in valid}
