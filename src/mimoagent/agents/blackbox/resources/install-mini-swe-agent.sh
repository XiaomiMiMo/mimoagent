#!/bin/bash
# Install upstream mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# into the test pod from PyPI.
#
# A standalone CPython (astral-sh/python-build-standalone) plus a venv at the
# exact /opt prefix it lives at, `mini-swe-agent==<version>` from PyPI, and a
# static ripgrep. The adapter (mini_swe_agent.py) runs `mini` non-interactively
# with a config derived from the installed package's mini.yaml.
#
# Lands:
#   /opt/mimo-mini-swe-agent/bin/mini        wrapper -> venv/bin/mini
#   /opt/mimo-mini-swe-agent/{python,venv}   standalone CPython 3.12 + the app venv
#   /opt/mimo-mini-swe-agent/bin/rg          ripgrep
#   /opt/mimo-mini-swe-agent/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   MINI_SWE_AGENT_VERSION      PyPI version (default "2.4.6")
#   MINI_SWE_AGENT_PAYLOAD / MINI_SWE_AGENT_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   PIP_INDEX_URL, PBS_BASE, GITHUB_BASE   mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${MINI_SWE_AGENT_VERSION:-2.4.6}"
CPYTHON_VERSION="3.12.12"
CPYTHON_TAG="20260114"
DEST="/opt/mimo-mini-swe-agent"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    # No --version flag exists; the package attribute is the version surface.
    # The env prefix keeps the probe hermetic: minisweagent mkdir's its global
    # config dir at import time, which would otherwise land in $HOME.
    MSWEA_SILENT_STARTUP=1 MSWEA_GLOBAL_CONFIG_DIR=/tmp/mimo-mini-swe-agent-verify \
        venv/bin/python -c 'import minisweagent; print(minisweagent.__version__)'
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

# Upstream invariants the adapter depends on.
check_invariants() {
    local sp
    sp=$("$DEST/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # 1. cache_control breakpoint auto-enabled for claude* names
    grep -q 'config\["set_cache_control"\] = "default_end"' "$sp/minisweagent/models/__init__.py" \
        || { echo "mini-swe-agent: cache_control default changed" >&2; return 1; }
    # 2. submit sentinel unchanged
    grep -q 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' "$sp/minisweagent/environments/local.py" \
        || { echo "mini-swe-agent: submit sentinel changed" >&2; return 1; }
    # 3. limit==0 means unlimited
    grep -q '0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost' \
        "$sp/minisweagent/agents/default.py" \
        || { echo "mini-swe-agent: limit semantics changed" >&2; return 1; }
    # 4. model_kwargs splatted into litellm.completion
    grep -q '\*\*(self.config.model_kwargs | kwargs)' "$sp/minisweagent/models/litellm_model.py" \
        || { echo "mini-swe-agent: model_kwargs passthrough changed" >&2; return 1; }
}

install_public() {
    echo "Installing mini-swe-agent $VERSION from PyPI + standalone CPython $CPYTHON_VERSION ($MIMO_LIBC)..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$DEST/python"
    "$DEST/python/bin/python3" -m venv "$DEST/venv"
    mimo_pip_install "$DEST/venv/bin/python" "mini-swe-agent==$VERSION"
    check_invariants
    fetch_rg "$DEST/bin/rg"
    mimo_wrapper "$DEST/bin/mini" "$DEST/venv/bin/mini"
}

if has_wanted; then
    echo "mini-swe-agent $VERSION already installed at $DEST, skipping..."
    echo "MINI_SWE_AGENT_INSTALL_OK"
    exit 0
fi

payload_override MINI_SWE_AGENT "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "mini-swe-agent $VERSION installed at $DEST"
echo "MINI_SWE_AGENT_INSTALL_OK"
