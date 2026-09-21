#!/bin/bash
# Install opencode into the test pod from npm.
#
# Upstream publishes bun-compiled standalone binaries per platform:
# `opencode-linux-x64` (glibc) and `opencode-linux-x64-musl` (Alpine/musl).
# Everything else — provider, effort variant, per-agent tool whitelist — is
# config the adapter stages at run time.
#
# Lands:
#   /opt/mimo-opencode/bin/opencode
#   /opt/mimo-opencode/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   OPENCODE_VERSION      npm package version (default "1.18.18")
#   OPENCODE_PAYLOAD / OPENCODE_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY          mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${OPENCODE_VERSION:-1.18.18}"
DEST="/opt/mimo-opencode"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

# Report the installed version, run from $DEST.
probe() {
    bin/opencode --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

install_public() {
    local pkg="opencode-linux-x64"
    [ "$MIMO_LIBC" = "musl" ] && pkg="opencode-linux-x64-musl"
    echo "Installing opencode $VERSION from npm ($pkg)..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_npm_tarball "$pkg" "$VERSION" "$DEST/pkg"
    mv "$DEST/pkg/bin/opencode" "$DEST/bin/opencode"
    chmod 0755 "$DEST/bin/opencode"
    rm -rf "$DEST/pkg"
}

if has_wanted; then
    echo "opencode $VERSION already installed at $DEST, skipping..."
    echo "OPENCODE_INSTALL_OK"
    exit 0
fi

payload_override OPENCODE "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "opencode $VERSION installed at $DEST"
echo "OPENCODE_INSTALL_OK"
