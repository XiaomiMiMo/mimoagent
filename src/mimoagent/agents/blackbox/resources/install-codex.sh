#!/bin/bash
# Install the OpenAI Codex CLI into the test pod from the upstream GitHub
# release (https://github.com/openai/codex/releases, tag rust-v<version>).
#
# The release ships a `codex-package-<target>.tar.gz` bundle for the static
# musl target (runs on glibc and musl images alike) with this layout:
#   bin/codex                  the CLI
#   bin/codex-code-mode-host   the Code Mode host codex spawns for models whose
#                              tool_mode is code_mode_only; codex fails closed
#                              (no tools) when it is missing, so the bundle —
#                              not a bare binary — is what gets installed
#   codex-path/rg              ripgrep codex puts on the agent's PATH
#   codex-resources/{bwrap,zsh/bin/zsh}
#
# Lands:
#   /opt/mimo-codex/bin/{codex,codex-code-mode-host}
#   /opt/mimo-codex/codex-path/rg
#   /opt/mimo-codex/codex  -> bin/codex (compat symlink)
#
# Env knobs:
#   CODEX_VERSION       release version (x.y.z, or "latest" for the newest release)
#   CODEX_INSTALL_ROOT  install root (default /opt/mimo-codex)
#   CODEX_PAYLOAD / CODEX_PAYLOAD_URL
#                       prebuilt payload tarball with the bundle layout above
#   GITHUB_BASE         mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

CODEX_VERSION="${CODEX_VERSION:-0.154.0}"
CODEX_ROOT="${CODEX_INSTALL_ROOT:-/opt/mimo-codex}"
CODEX_BIN_DIR="$CODEX_ROOT/bin"
CODEX_BIN="$CODEX_BIN_DIR/codex"
HOST_BIN="$CODEX_BIN_DIR/codex-code-mode-host"
PACKAGE_ASSET="codex-package-x86_64-unknown-linux-musl.tar.gz"

# has_wanted_version <bin>: binary runs and matches CODEX_VERSION. For "latest"
# there is no number to compare — just require that the binary runs.
has_wanted_version() {
    if [ "$CODEX_VERSION" = "latest" ]; then
        [ -x "$1" ] && "$1" --version >/dev/null 2>&1
    else
        [ -x "$1" ] && "$1" --version 2>/dev/null | grep -q "codex-cli ${CODEX_VERSION}$"
    fi
}

# The bundle layout is complete only with the Code Mode host next to codex. An
# older single-binary install of the right version is deliberately *not*
# accepted here so it gets upgraded in place.
bundle_complete() {
    [ -x "$CODEX_BIN" ] && [ -x "$HOST_BIN" ]
}

# finish_bundle <stage_dir>: validate, set modes, atomically move into place.
finish_bundle() {
    local stage="$1" f
    for f in bin/codex bin/codex-code-mode-host; do
        [ -f "$stage/$f" ] || { echo "bundle is missing $f — not a codex-package tarball?" >&2; exit 1; }
    done
    chmod 0755 "$stage/bin/codex" "$stage/bin/codex-code-mode-host"
    [ -f "$stage/codex-path/rg" ] && chmod 0755 "$stage/codex-path/rg"
    [ -f "$stage/codex-resources/bwrap" ] && chmod 0755 "$stage/codex-resources/bwrap"
    [ -f "$stage/codex-resources/zsh/bin/zsh" ] && chmod 0755 "$stage/codex-resources/zsh/bin/zsh"
    ln -sfn "bin/codex" "$stage/codex"
    rm -rf "$CODEX_ROOT"
    mv "$stage" "$CODEX_ROOT"
    "$CODEX_BIN" --version
    has_wanted_version "$CODEX_BIN" || { echo "codex version check failed (want ${CODEX_VERSION})" >&2; exit 1; }
    bundle_complete || { echo "bundle install incomplete: $HOST_BIN missing" >&2; exit 1; }
    echo "installed: $CODEX_BIN + $HOST_BIN + $(ls "$CODEX_ROOT/codex-path" 2>/dev/null | tr '\n' ' ')"
}

if has_wanted_version "$CODEX_BIN" && bundle_complete; then
    echo "codex already installed ($("$CODEX_BIN" --version), code-mode host present), skipping..."
    echo "CODEX_INSTALL_OK"
    exit 0
fi

mkdir -p "$(dirname "$CODEX_ROOT")"
STAGE="${CODEX_ROOT}.staging.$$"
trap 'rm -rf "$STAGE"' EXIT

# ----- offline payload override --------------------------------------------------
if payload_override CODEX "$STAGE"; then
    finish_bundle "$STAGE"
    echo "CODEX_INSTALL_OK"
    exit 0
fi

# ----- upstream GitHub release ---------------------------------------------------
tag="rust-v${CODEX_VERSION}"
if [ "$CODEX_VERSION" = "latest" ]; then
    tmp="$(mktemp)"
    mimo_fetch "https://api.github.com/repos/openai/codex/releases/latest" "$tmp"
    tag="$(sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' "$tmp" | head -1)"
    rm -f "$tmp"
    [ -n "$tag" ] || { echo "could not resolve the latest codex release" >&2; exit 1; }
    echo "latest codex release: $tag"
fi
echo "Downloading Codex CLI bundle ($tag / $PACKAGE_ASSET)..."
ensure_tar_gz
TMP_TAR="$(mktemp)"
fetch_github_release "openai/codex" "$tag" "$PACKAGE_ASSET" "$TMP_TAR" || { echo "Failed to download $PACKAGE_ASSET" >&2; exit 1; }
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$TMP_TAR" -C "$STAGE"
rm -f "$TMP_TAR"
finish_bundle "$STAGE"

echo "CODEX_INSTALL_OK"
