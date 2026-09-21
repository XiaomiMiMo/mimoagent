#!/bin/bash
# Install the xAI Grok Build CLI into the test pod from npm.
#
# Upstream publishes a brotli-compressed static binary in the platform package
# `@xai-official/grok-linux-x64` (bin/grok.br). It has no libc dependency, so
# one build serves glibc and musl images alike.
#
# Lands:
#   /opt/mimo-grok/bin/grok
#   /opt/mimo-grok/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   GROK_VERSION      npm package version (default "1.0.4")
#   GROK_PAYLOAD / GROK_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY      mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${GROK_VERSION:-1.0.4}"
DEST="/opt/mimo-grok"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

# Report the installed version, run from $DEST.
probe() {
    bin/grok --version
}

# has_wanted: the marker names the wanted version AND the unpacked tree agrees.
# Both halves matter — the marker alone would accept a half-unpacked tree, the
# probe alone re-runs a slow interpreter for nothing on every turn.
has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

# Decompress bin/grok.br in place. Prefer a brotli CLI; fall back to a
# throwaway Node runtime's zlib when the image has none.
brotli_decompress() {
    local src="$1" dst="$2"
    if ! command -v brotli >/dev/null 2>&1; then
        mimo_pkg_install brotli >/dev/null 2>&1 || true
    fi
    if command -v brotli >/dev/null 2>&1; then
        brotli -d -f -o "$dst" "$src"
        return
    fi
    echo "no brotli CLI available; using a temporary Node runtime to decompress"
    fetch_node "22.19.0" "$DEST/.node-tmp"
    "$DEST/.node-tmp/bin/node" -e '
const zlib = require("zlib"), fs = require("fs");
fs.writeFileSync(process.argv[2], zlib.brotliDecompressSync(fs.readFileSync(process.argv[1])));
' "$src" "$dst"
    rm -rf "$DEST/.node-tmp"
}

install_public() {
    echo "Installing grok $VERSION from npm (@xai-official/grok-linux-x64)..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_npm_tarball "@xai-official/grok-linux-x64" "$VERSION" "$DEST/pkg"
    if [ -f "$DEST/pkg/bin/grok" ]; then
        mv "$DEST/pkg/bin/grok" "$DEST/bin/grok"
    else
        brotli_decompress "$DEST/pkg/bin/grok.br" "$DEST/bin/grok"
    fi
    chmod 0755 "$DEST/bin/grok"
    rm -rf "$DEST/pkg"
}

if has_wanted; then
    echo "grok $VERSION already installed at $DEST, skipping..."
    echo "GROK_INSTALL_OK"
    exit 0
fi

payload_override GROK "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "grok $VERSION installed at $DEST"
echo "GROK_INSTALL_OK"
