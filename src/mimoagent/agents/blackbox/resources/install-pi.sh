#!/bin/bash
# Install a pinned Pi CLI (https://github.com/earendil-works/pi) into the test
# pod from its GitHub release: a standalone Bun binary plus the startup assets
# (package.json + theme/) that Pi expects next to it — no node/npm needed.
#
# Lands:
#   /opt/mimo-pi/standalone/{pi,package.json,theme/}
#   /opt/mimo-pi/bin/pi -> ../standalone/pi
#
# Env knobs:
#   PI_VERSION      release version (default "0.83.0"); verified with `pi --version`
#   PI_ROOT         install root (default /opt/mimo-pi)
#   PI_PAYLOAD / PI_PAYLOAD_URL
#                   prebuilt payload tarball with the standalone/ layout above
#   GITHUB_BASE     mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

PI_VERSION="${PI_VERSION:-0.83.0}"
PI_ROOT="${PI_ROOT:-/opt/mimo-pi}"
PI_BIN_DIR="${PI_BIN_DIR:-${PI_ROOT}/bin}"
STANDALONE_DIR="${PI_ROOT}/standalone"
STANDALONE_BIN="${STANDALONE_DIR}/pi"
PI_BIN="${PI_BIN_DIR}/pi"

standalone_ok() {
    [ -x "$STANDALONE_BIN" ] || return 1
    [ "$("$STANDALONE_BIN" --version 2>/dev/null)" = "$PI_VERSION" ] || return 1
}

if standalone_ok; then
    echo "Pi ${PI_VERSION} already installed at ${PI_ROOT}, skipping..."
else
    if ! payload_override PI "$STANDALONE_DIR"; then
        # The release tarball unpacks to pi/{pi,package.json,theme,...}; only
        # a glibc build is published for linux-x64.
        mimo_detect_libc
        if [ "$MIMO_LIBC" = "musl" ]; then
            echo "warning: Pi publishes no musl build; the glibc binary may not run on this image" >&2
        fi
        ASSET="pi-linux-x64.tar.gz"
        echo "Downloading Pi ${PI_VERSION} (${ASSET}) from GitHub..."
        ensure_tar_gz
        TMP_TAR="$(mktemp)"
        fetch_github_release "earendil-works/pi" "v${PI_VERSION}" "$ASSET" "$TMP_TAR" \
            || { echo "Failed to download ${ASSET}" >&2; exit 1; }
        rm -rf "$STANDALONE_DIR"
        mkdir -p "$STANDALONE_DIR"
        tar -xzf "$TMP_TAR" -C "$STANDALONE_DIR" --strip-components=1
        rm -f "$TMP_TAR"
    fi
    chmod 0755 "$STANDALONE_BIN"
    standalone_ok || {
        echo "Pi install verification failed (want Pi ${PI_VERSION}, got: $("$STANDALONE_BIN" --version 2>&1 || true))" >&2
        exit 1
    }
fi

mkdir -p "$PI_BIN_DIR"
ln -sfn "$STANDALONE_BIN" "$PI_BIN"

"$PI_BIN" --version | grep -Fx "$PI_VERSION" >/dev/null || {
    echo "Pi CLI version check failed" >&2
    exit 1
}

echo "Pi ${PI_VERSION} installed at ${PI_ROOT}"
echo "PI_INSTALL_OK"
