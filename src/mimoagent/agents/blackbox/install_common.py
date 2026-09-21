"""Stage the shared installer library into a pod.

Every blackbox ``install-<name>.sh`` sources ``resources/install-common.sh``
(download helpers, libc detection, upstream fetchers, payload override). The
adapters copy it into the pod once per environment and prefix the install
command with ``MIMO_INSTALL_COMMON=<pod path>`` so the script finds it.

``installer_env_prefix`` turns the adapter's ``install_env`` mapping (mirror
URLs such as ``NPM_REGISTRY`` / ``PIP_INDEX_URL`` / ``GITHUB_BASE``) into the
same ``KEY=value`` prefix form.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path

_RESOURCES = Path(__file__).resolve().parent / "resources"
INSTALL_COMMON_SCRIPT = _RESOURCES / "install-common.sh"
POD_INSTALL_COMMON = "/tmp/mimo-install-common.sh"


def stage_install_common(env) -> str:
    """Copy ``install-common.sh`` into the pod (once per env object) and return
    the ``MIMO_INSTALL_COMMON=... `` prefix for the install command."""
    if not getattr(env, "_mimo_install_common_staged", False):
        env.copy_to(str(INSTALL_COMMON_SCRIPT), POD_INSTALL_COMMON)
        try:
            env._mimo_install_common_staged = True
        except Exception:
            pass
    return f"MIMO_INSTALL_COMMON={POD_INSTALL_COMMON} "


def installer_env_prefix(install_env: Mapping[str, str] | None) -> str:
    """``KEY=value `` pairs for the extra installer environment, shell-quoted."""
    if not install_env:
        return ""
    return "".join(f"{key}={shlex.quote(str(value))} " for key, value in install_env.items())
