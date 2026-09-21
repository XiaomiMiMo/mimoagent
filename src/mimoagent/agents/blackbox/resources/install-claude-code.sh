#!/bin/bash
# Install Claude Code into the test pod.
#
# Two pieces, both from public sources:
#   * the standalone Claude Code CLI binary, from the release bucket that the
#     official `claude.ai/install.sh` reads (pinned version, glibc or musl
#     variant auto-detected);
#   * `claude-agent-sdk` (the Python SDK the runner `run_claude_sdk.py` uses),
#     installed from PyPI into a standalone CPython 3.12 so the image's own
#     Python is never touched.
# The runner drives the pinned CLI via --cli-path, NOT the CLI bundled inside
# the wheel, so the scaffold version is controlled here.
#
# Lands:
#   /opt/mimo-python/python/bin/python3.12   standalone CPython 3.12
#   claude-agent-sdk (+ deps)                installed into that interpreter
#   /opt/mimo-claude/claude                  pinned Claude Code CLI binary
#
# Env knobs:
#   CLAUDE_VERSION         CLI version (x.y.z, or "latest" for the newest release)
#   CLAUDE_RELEASES_BASE   override the release bucket base URL
#   CLAUDE_SDK_SPEC        pip requirement for the SDK (default "claude-agent-sdk")
#   CLAUDE_CODE_PAYLOAD / CLAUDE_CODE_PAYLOAD_URL
#                          prebuilt payload tarball unpacked to / (offline mode;
#                          must lay down the paths above)
#   PIP_INDEX_URL, PBS_BASE  mirror overrides (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

CLAUDE_VERSION="${CLAUDE_VERSION:-2.1.269}"
CLAUDE_RELEASES_BASE="${CLAUDE_RELEASES_BASE:-https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases}"
CLAUDE_SDK_SPEC="${CLAUDE_SDK_SPEC:-claude-agent-sdk}"
CPYTHON_VERSION="3.12.12"
CPYTHON_TAG="20260114"

PYTHON_INSTALL_DIR="/opt/mimo-python"
PYTHON_BIN="$PYTHON_INSTALL_DIR/python/bin/python3.12"
CLAUDE_BIN_DIR="/opt/mimo-claude"
CLAUDE_BIN="$CLAUDE_BIN_DIR/claude"

mimo_detect_libc

# has_pinned <bin> <version>: binary exists and reports the pinned version.
# For "latest" there is no number to compare — just require that it runs.
has_pinned() {
    if [ "$2" = "latest" ]; then
        [ -x "$1" ] && "$1" -v >/dev/null 2>&1
    else
        [ -x "$1" ] && "$1" -v 2>/dev/null | grep -q "^$2 "
    fi
}

sdk_ok() {
    [ -x "$PYTHON_BIN" ] && "$PYTHON_BIN" -c "import claude_agent_sdk" 2>/dev/null
}

if sdk_ok && has_pinned "$CLAUDE_BIN" "$CLAUDE_VERSION"; then
    echo "claude-agent-sdk + pinned CLI already installed, skipping..."
    echo "CLAUDE_CODE_INSTALL_OK"
    exit 0
fi

# ----- offline payload override --------------------------------------------------
if payload_override CLAUDE_CODE /opt/mimo-claude-code-payload; then
    # The payload mirrors the absolute layout above under its root.
    cp -a /opt/mimo-claude-code-payload/. /
    rm -rf /opt/mimo-claude-code-payload
    sdk_ok || { echo "payload did not provide claude-agent-sdk at $PYTHON_BIN" >&2; exit 1; }
    has_pinned "$CLAUDE_BIN" "$CLAUDE_VERSION" || { echo "payload CLI version check failed (want $CLAUDE_VERSION)" >&2; exit 1; }
    echo "CLAUDE_CODE_INSTALL_OK"
    exit 0
fi

# ----- standalone Python + SDK ---------------------------------------------------
if ! sdk_ok; then
    echo "Installing standalone CPython $CPYTHON_VERSION ($MIMO_LIBC)..."
    fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$PYTHON_INSTALL_DIR/python"
    echo "Installing $CLAUDE_SDK_SPEC from PyPI..."
    mimo_pip_install "$PYTHON_BIN" "$CLAUDE_SDK_SPEC"
    sdk_ok || { echo "claude-agent-sdk import failed" >&2; exit 1; }
fi

# ----- pinned CLI binary ---------------------------------------------------------
if ! has_pinned "$CLAUDE_BIN" "$CLAUDE_VERSION"; then
    version="$CLAUDE_VERSION"
    if [ "$version" = "latest" ]; then
        tmp="$(mktemp)"
        mimo_fetch "$CLAUDE_RELEASES_BASE/latest" "$tmp"
        version="$(tr -d '[:space:]' < "$tmp")"
        rm -f "$tmp"
        [ -n "$version" ] || { echo "could not resolve the latest Claude Code version" >&2; exit 1; }
        echo "latest Claude Code release: $version"
    fi
    echo "Downloading Claude Code $version ($MIMO_PLATFORM)..."
    mkdir -p "$CLAUDE_BIN_DIR"
    mimo_fetch "$CLAUDE_RELEASES_BASE/$version/$MIMO_PLATFORM/claude" "$CLAUDE_BIN.tmp"
    chmod 0755 "$CLAUDE_BIN.tmp"
    mv -f "$CLAUDE_BIN.tmp" "$CLAUDE_BIN"
    has_pinned "$CLAUDE_BIN" "$CLAUDE_VERSION" || { echo "Claude Code version check failed (want $CLAUDE_VERSION): $("$CLAUDE_BIN" -v 2>&1 | tail -1)" >&2; exit 1; }
fi

echo "Claude Code $("$CLAUDE_BIN" -v) installed at $CLAUDE_BIN"
echo "CLAUDE_CODE_INSTALL_OK"
