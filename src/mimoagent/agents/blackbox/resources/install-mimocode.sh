#!/bin/bash
# Install MiMo-Code (https://github.com/XiaomiMiMo/MiMo-Code) into the test pod
# from its GitHub release.
#
# MiMo-Code ships as a single Bun-compiled binary per platform (glibc + musl
# variants). The release tarball `mimocode-linux-x64[-musl].tar.gz` contains
# just the `mimo` binary; ripgrep is installed next to it because mimo's
# grep/glob tools resolve a sibling `rg` off their own executable path before
# consulting PATH or trying to download one at run time.
#
# Lands:
#   /opt/mimo-mimocode/bin/mimo   the mimocode binary the agent invokes
#   /opt/mimo-mimocode/bin/rg     ripgrep, next to `mimo`
#
# Env knobs:
#   MIMOCODE_VERSION    release version (default "0.1.12"); verified against
#                       `mimo --version` as a prefix, so a suffixed rebuild
#                       satisfies its pin while stale builds still fail.
#   MIMOCODE_BIN_DIR    install dir (default /opt/mimo-mimocode/bin)
#   MIMOCODE_PAYLOAD / MIMOCODE_PAYLOAD_URL
#                       prebuilt payload tarball (contains mimo [+ rg])
#   GITHUB_BASE         mirror override (see install-common.sh)
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

MIMOCODE_VERSION="${MIMOCODE_VERSION:-0.1.12}"
MIMOCODE_BIN_DIR="${MIMOCODE_BIN_DIR:-/opt/mimo-mimocode/bin}"
MIMOCODE_BIN="$MIMOCODE_BIN_DIR/mimo"
RG_BIN="$MIMOCODE_BIN_DIR/rg"

mimo_detect_libc

# has_wanted <bin>: binary exists, runs, and its --version starts with
# MIMOCODE_VERSION. A suffixed build satisfies its exact pin; a stale release
# or a "local" dev build does not.
has_wanted() {
    [ -x "$1" ] || return 1
    local out
    out="$("$1" --version 2>/dev/null)" || return 1
    case "$out" in
        "$MIMOCODE_VERSION"|"$MIMOCODE_VERSION"-*) return 0 ;;
        *) return 1 ;;
    esac
}

rg_ok() {
    [ -x "$RG_BIN" ] && "$RG_BIN" --version >/dev/null 2>&1
}

if has_wanted "$MIMOCODE_BIN" && rg_ok; then
    echo "mimocode already installed ($("$MIMOCODE_BIN" --version)), skipping..."
    echo "MIMOCODE_INSTALL_OK"
    exit 0
fi

if ! payload_override MIMOCODE "$MIMOCODE_BIN_DIR"; then
    ASSET="mimocode-${MIMO_PLATFORM}.tar.gz"
    echo "Downloading MiMo-Code ${MIMOCODE_VERSION} (${ASSET}) from GitHub..."
    ensure_tar_gz
    TMP_TAR="$(mktemp)"
    fetch_github_release "XiaomiMiMo/MiMo-Code" "v${MIMOCODE_VERSION}" "$ASSET" "$TMP_TAR" \
        || { echo "Failed to download ${ASSET}" >&2; exit 1; }
    rm -rf "$MIMOCODE_BIN_DIR"
    mkdir -p "$MIMOCODE_BIN_DIR"
    tar -xzf "$TMP_TAR" -C "$MIMOCODE_BIN_DIR"
    rm -f "$TMP_TAR"
fi
chmod 0755 "$MIMOCODE_BIN"

# mimo needs *a* working `rg` beside it. Use the payload's bundled one when it
# runs here; otherwise (or on the public path) drop in a fully static ripgrep.
if rg_ok; then
    echo "ripgrep bundled with the payload: $("$RG_BIN" --version | head -1)"
else
    rm -f "$RG_BIN"
    echo "fetching static ripgrep next to mimo..."
    fetch_rg "$RG_BIN" || { echo "could not obtain a working rg for mimo" >&2; exit 1; }
    echo "ripgrep: $("$RG_BIN" --version | head -1)"
fi

# `|| true`: a binary built for the wrong libc dies with rc=127 ("No such file
# or directory" — missing ELF interpreter). Under set -e that would abort here
# with a bare 127; swallow it so has_wanted below reports the real problem.
VERSION_OUT="$("$MIMOCODE_BIN" --version 2>&1 || true)"
has_wanted "$MIMOCODE_BIN" \
    || { echo "mimocode version check failed (want ${MIMOCODE_VERSION}, got: ${VERSION_OUT})" >&2; exit 1; }
echo "mimocode ${VERSION_OUT} installed at ${MIMOCODE_BIN}"

echo "MIMOCODE_INSTALL_OK"
