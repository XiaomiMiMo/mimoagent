#!/bin/bash
# Install Kilo Code into the test pod from npm.
#
# Kilo Code is an opencode fork and likewise ships bun-compiled standalone
# binaries per platform. On glibc images the `-baseline` variant is used: it is
# the one that runs on both AVX2 and non-AVX2 hosts. On musl images the `-musl`
# variant links musl's loader. The whole bin/ tree is required, not just the
# binary: tree-sitter grammars, the bwrap + seccomp helpers and the sandbox
# workers live alongside it. The embedded web UI (bin/console) is dropped —
# dead weight in a headless rollout.
#
# Lands:
#   /opt/mimo-kilocode/bin/kilo (+ helpers)
#   /opt/mimo-kilocode/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   KILOCODE_VERSION      npm package version (default "7.4.22")
#   KILOCODE_PAYLOAD / KILOCODE_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY          mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${KILOCODE_VERSION:-7.4.22}"
DEST="/opt/mimo-kilocode"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/kilo --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

install_public() {
    local pkg="@kilocode/cli-linux-x64-baseline"
    [ "$MIMO_LIBC" = "musl" ] && pkg="@kilocode/cli-linux-x64-musl"
    echo "Installing kilocode $VERSION from npm ($pkg)..."
    rm -rf "$DEST"
    mkdir -p "$DEST"
    fetch_npm_tarball "$pkg" "$VERSION" "$DEST/pkg"
    mv "$DEST/pkg/bin" "$DEST/bin"
    [ -f "$DEST/pkg/LICENSE" ] && mv "$DEST/pkg/LICENSE" "$DEST/LICENSE"
    rm -rf "$DEST/pkg" "$DEST/bin/console"
    chmod 0755 "$DEST/bin/kilo"
}

if has_wanted; then
    echo "kilocode $VERSION already installed at $DEST, skipping..."
    echo "KILOCODE_INSTALL_OK"
    exit 0
fi

payload_override KILOCODE "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "kilocode $VERSION installed at $DEST"
echo "KILOCODE_INSTALL_OK"
