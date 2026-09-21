#!/bin/bash
# Install Kimi CLI into the test pod from PyPI.
#
# Python >= 3.12 app: a standalone CPython (astral-sh/python-build-standalone)
# plus a venv at the exact /opt prefix it lives at (venvs are not relocatable),
# and a static ripgrep so the Grep tool never dials a CDN at run time.
#
# Three small source patches, all gated on env vars the adapter sets
# (kimi_cli.py). The first two make xhigh/max adaptive thinking reachable
# (upstream only ever sends effort=high and its model-name list predates the
# families this harness runs with); the third unpins the output ceiling:
#   1. kosong: treat opus-5/fable ids as adaptive-capable, allow xhigh
#   2. kimi_cli: read the effort level from $KIMI_CLI_THINKING_EFFORT
#   3. kimi_cli: read max_tokens from $KIMI_CLI_MAX_TOKENS (upstream pins 50000)
#
# Lands:
#   /opt/mimo-kimi-cli/bin/kimi       wrapper -> venv/bin/kimi
#   /opt/mimo-kimi-cli/{python,venv}  standalone CPython 3.12 + the app venv
#   /opt/mimo-kimi-cli/bin/rg         ripgrep for the Grep tool
#   /opt/mimo-kimi-cli/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   KIMI_CLI_VERSION      PyPI version (default "1.49.0")
#   KIMI_CLI_PAYLOAD / KIMI_CLI_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   PIP_INDEX_URL, PBS_BASE, GITHUB_BASE      mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${KIMI_CLI_VERSION:-1.49.0}"
CPYTHON_VERSION="3.12.12"
CPYTHON_TAG="20260114"
DEST="/opt/mimo-kimi-cli"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/kimi --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

# patch_once <file> <grep-anchor> <sed-expression>: require exactly one anchor
# (or an already-patched file) before rewriting.
patch_once() {
    local file="$1" anchor="$2" expr="$3"
    local n
    n="$(grep -c -- "$anchor" "$file" || true)"
    if [ "$n" = "0" ]; then
        return 0  # already patched (anchor gone)
    fi
    [ "$n" = "1" ] || { echo "kimi-cli: patch anchor '$anchor' matched $n times in $file" >&2; return 1; }
    sed -i "$expr" "$file"
}

apply_patches() {
    local sp
    sp=$("$DEST/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    local kosong="$sp/kosong/contrib/chat_provider/anthropic.py" llm="$sp/kimi_cli/llm.py"
    # 1a: adaptive marker for unversioned opus-5/fable ids
    patch_once "$kosong" '_ADAPTIVE_MARKERS_NO_VERSION: tuple\[str, \.\.\.\] = ("mythos",)' \
        's/_ADAPTIVE_MARKERS_NO_VERSION: tuple\[str, \.\.\.\] = ("mythos",)/_ADAPTIVE_MARKERS_NO_VERSION: tuple[str, ...] = ("mythos", "opus-5", "fable")/'
    # 1b: adaptive family accepts xhigh too
    patch_once "$kosong" 'return frozenset({"low", "medium", "high", "max"})' \
        's/return frozenset({"low", "medium", "high", "max"})/return frozenset({"low", "medium", "high", "xhigh", "max"})/'
    # 2: effort level from env (llm.py already imports os)
    patch_once "$llm" 'chat_provider.with_thinking("high")' \
        's/chat_provider.with_thinking("high")/chat_provider.with_thinking(os.environ.get("KIMI_CLI_THINKING_EFFORT", "high"))/'
    # 3: output ceiling from env
    patch_once "$llm" 'default_max_tokens=50000' \
        's/default_max_tokens=50000/default_max_tokens=int(os.environ.get("KIMI_CLI_MAX_TOKENS") or 50000)/'
    grep -q 'KIMI_CLI_THINKING_EFFORT' "$llm" || { echo "kimi-cli: effort patch did not apply" >&2; return 1; }
    grep -q 'KIMI_CLI_MAX_TOKENS' "$llm" || { echo "kimi-cli: max_tokens patch did not apply" >&2; return 1; }
    "$DEST/venv/bin/python" -c "import kimi_cli, kosong"
}

install_public() {
    echo "Installing kimi-cli $VERSION from PyPI + standalone CPython $CPYTHON_VERSION ($MIMO_LIBC)..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$DEST/python"
    "$DEST/python/bin/python3" -m venv "$DEST/venv"
    mimo_pip_install "$DEST/venv/bin/python" "kimi-cli==$VERSION"
    apply_patches
    fetch_rg "$DEST/bin/rg"
    mimo_wrapper "$DEST/bin/kimi" "$DEST/venv/bin/kimi"
}

if has_wanted; then
    echo "kimi-cli $VERSION already installed at $DEST, skipping..."
    echo "KIMI_CLI_INSTALL_OK"
    exit 0
fi

payload_override KIMI_CLI "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "kimi-cli $VERSION installed at $DEST"
echo "KIMI_CLI_INSTALL_OK"
