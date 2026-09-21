"""Opt-in end-to-end smoke test for the blackbox installers.

Runs each ``install-<name>.sh`` from its upstream public source inside a
throwaway container and asserts the ``<PREFIX>_INSTALL_OK`` sentinel. It
downloads real runtimes and packages (hundreds of MB per harness), so it only
runs when explicitly requested:

    MIMOAGENT_INSTALLER_SMOKE=1 uv run pytest -m integration tests/agents/test_installers_docker_smoke.py

Restrict the set with ``MIMOAGENT_INSTALLER_SMOKE_HARNESSES=grok,opencode`` and
the images with ``MIMOAGENT_INSTALLER_SMOKE_IMAGES=python:3.12-slim,alpine:3.20``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from mimoagent.agents.blackbox.install_common import INSTALL_COMMON_SCRIPT, POD_INSTALL_COMMON

_RESOURCES = Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources"
_ALL = [
    p.name[len("install-") : -len(".sh")]
    for p in sorted(_RESOURCES.glob("install-*.sh"))
    if p.name != "install-common.sh"
]
_ENABLED = os.getenv("MIMOAGENT_INSTALLER_SMOKE") == "1"
_HARNESSES = [h for h in os.getenv("MIMOAGENT_INSTALLER_SMOKE_HARNESSES", ",".join(_ALL)).split(",") if h]
_IMAGES = [i for i in os.getenv("MIMOAGENT_INSTALLER_SMOKE_IMAGES", "python:3.12-slim,alpine:3.20").split(",") if i]


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ENABLED, reason="set MIMOAGENT_INSTALLER_SMOKE=1 to run the installer smoke test"),
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]


@pytest.mark.parametrize("image", _IMAGES)
@pytest.mark.parametrize("name", _HARNESSES)
def test_installer_completes_from_public_sources(name: str, image: str):
    from mimoagent.environments.docker import DockerEnvironment

    prefix = name.upper().replace("-", "_")
    # hosts behind an egress proxy: forward it so the container can download
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"]
    env = DockerEnvironment(image=image, cwd="/", timeout=1800, forward_env=proxy_vars)
    env.start()
    try:
        # alpine ships no bash; the installers are bash scripts
        if "alpine" in image:
            env.execute("apk add --no-cache bash >/dev/null 2>&1 || true", timeout=300)
        env.copy_to(str(INSTALL_COMMON_SCRIPT), POD_INSTALL_COMMON)
        env.copy_to(str(_RESOURCES / f"install-{name}.sh"), f"/tmp/mimo-install-{name}.sh")
        result = env.execute(
            f"MIMO_INSTALL_COMMON={POD_INSTALL_COMMON} bash {shlex.quote(f'/tmp/mimo-install-{name}.sh')} 2>&1",
            timeout=1800,
        )
        output = result.get("output", "")
        assert f"{prefix}_INSTALL_OK" in output, (
            f"{name} on {image} failed (rc={result.get('returncode')}):\n{output[-4000:]}"
        )
        # a second run must be a cheap no-op
        again = env.execute(
            f"MIMO_INSTALL_COMMON={POD_INSTALL_COMMON} bash {shlex.quote(f'/tmp/mimo-install-{name}.sh')} 2>&1",
            timeout=600,
        )
        assert "already installed" in again.get("output", ""), again.get("output", "")[-2000:]
    finally:
        env.cleanup()
