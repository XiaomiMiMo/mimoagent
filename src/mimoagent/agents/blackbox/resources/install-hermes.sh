#!/bin/bash
# Install the NousResearch Hermes Agent into the test pod from its GitHub tag.
#
# Python app that is not on PyPI: the tagged source tree is downloaded from
# GitHub, kept under the install prefix and installed *editable* into a venv
# on a standalone CPython. (Recent releases refuse wheel builds outside their
# own packaging because a wheel drops bundled assets — locales, skills,
# plugin manifests — that are resolved from the source layout at run time.)
# The [anthropic] extra is mandatory: without it the SDK is lazily
# pip-installed at run time.
#
# Two small source patches, both gated on env vars the adapter sets
# (hermes.py):
#   1. HERMES_FORCE_NATIVE_ANTHROPIC=1 — hermes treats any base_url without
#      an "anthropic.com" substring as a third-party endpoint and strips all
#      historic thinking blocks, which breaks signature replay through an
#      Anthropic-compatible gateway;
#   2. HERMES_KEEP_ALL_THINKING=1 — keep every signed thinking block instead
#      of only the latest assistant message's (prompt cache + trajectory).
#
# Lands:
#   /opt/mimo-hermes/bin/hermes            wrapper -> venv/bin/hermes
#   /opt/mimo-hermes/{python,venv,src}     standalone CPython 3.12, the app venv, the source tree
#   /opt/mimo-hermes/bin/rg                ripgrep
#   /opt/mimo-hermes/models_dev_cache.json pre-seeded catalog (best effort; skips a slow fetch)
#   /opt/mimo-hermes/.mimo-payload-version installed-version marker (idempotency)
#
# Env knobs:
#   HERMES_VERSION      GitHub tag without the leading v (default "2026.8.13")
#   HERMES_PAYLOAD / HERMES_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   PIP_INDEX_URL, PBS_BASE, GITHUB_BASE  mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${HERMES_VERSION:-2026.8.13}"
CPYTHON_VERSION="3.12.12"
CPYTHON_TAG="20260114"
DEST="/opt/mimo-hermes"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/hermes --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

apply_patches() {
    local adapter="$DEST/src/agent/anthropic_adapter.py"
    # Patch 1: env escape hatch on the third-party heuristic.
    if ! grep -q 'HERMES_FORCE_NATIVE_ANTHROPIC' "$adapter"; then
        [ "$(grep -c '^def _is_third_party_anthropic_endpoint' "$adapter")" = "1" ] \
            || { echo "hermes: _is_third_party_anthropic_endpoint anchor moved" >&2; return 1; }
        sed -i '/^def _is_third_party_anthropic_endpoint/a\    if os.environ.get("HERMES_FORCE_NATIVE_ANTHROPIC"):\n        return False  # gateway declared native: replay signed thinking blocks' \
            "$adapter"
    fi
    # Patch 2: keep every signed thinking block when asked to.
    "$DEST/venv/bin/python" - "$adapter" <<'PYPATCH'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
anchor = "        elif _is_third_party or idx != last_assistant_idx:"
patched = ('        elif _is_third_party or (\n'
           '            idx != last_assistant_idx\n'
           '            and not os.environ.get("HERMES_KEEP_ALL_THINKING")\n'
           '        ):')
if patched in text:
    sys.exit(0)
assert text.count(anchor) == 1, f"hermes: thinking-retention anchor moved ({text.count(anchor)} matches)"
path.write_text(text.replace(anchor, patched))
PYPATCH
    # Self-check: both patches must be effective and the defaults unchanged.
    "$DEST/venv/bin/python" - <<'PYCHECK'
import os
os.environ["HERMES_FORCE_NATIVE_ANTHROPIC"] = "1"
from agent.anthropic_adapter import _is_third_party_anthropic_endpoint as f
from agent.anthropic_adapter import _manage_thinking_signatures as trim
assert f("http://gateway") is False, "patch ineffective"
del os.environ["HERMES_FORCE_NATIVE_ANTHROPIC"]
assert f("http://gateway") is True, "patch broke the default path"
os.environ["HERMES_FORCE_NATIVE_ANTHROPIC"] = "1"


def conversation():
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t1", "signature": "S1"},
            {"type": "tool_use", "id": "a", "name": "x", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "r"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "t2", "signature": "S2"},
            {"type": "text", "text": "done"}]},
    ]


def signed(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for b in m["content"]
               if isinstance(b, dict) and b.get("type") == "thinking" and b.get("signature"))


msgs = conversation()
trim(msgs, "http://gateway", "claude-opus-5")
assert signed(msgs) == 1, f"upstream default changed: {signed(msgs)} signed blocks"
os.environ["HERMES_KEEP_ALL_THINKING"] = "1"
msgs = conversation()
trim(msgs, "http://gateway", "claude-opus-5")
assert signed(msgs) == 2, f"keep-all patch ineffective: {signed(msgs)} signed blocks"
print("hermes: patches OK")
PYCHECK
}

install_public() {
    echo "Installing hermes $VERSION from GitHub (NousResearch/hermes-agent v$VERSION) + standalone CPython..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    ensure_tar_gz
    local archive
    archive="$(mktemp)"
    mimo_fetch "${GITHUB_BASE:-https://github.com}/NousResearch/hermes-agent/archive/refs/tags/v${VERSION}.tar.gz" "$archive"
    mkdir -p "$DEST/src"
    tar -xzf "$archive" -C "$DEST/src" --strip-components=1
    rm -f "$archive"
    [ -f "$DEST/src/pyproject.toml" ] || { echo "hermes source extract failed" >&2; return 1; }
    # docs + marketing website are a third of the tree and unused headless.
    rm -rf "$DEST/src/website" "$DEST/src/docs" "$DEST/src/.git"
    fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$DEST/python"
    "$DEST/python/bin/python3" -m venv "$DEST/venv"
    mimo_pip_install "$DEST/venv/bin/python" -e "$DEST/src[anthropic]"
    apply_patches
    # models.dev catalog cache: a pre-seeded file skips a slow network fetch on
    # startup; without it the pod just takes the timeout path.
    mimo_fetch "https://models.dev/api.json" "$DEST/models_dev_cache.json" 2>/dev/null \
        || { rm -f "$DEST/models_dev_cache.json"; echo "warn: no models.dev catalog available; startup falls back to the timeout path" >&2; }
    fetch_rg "$DEST/bin/rg"
    mimo_wrapper "$DEST/bin/hermes" "$DEST/venv/bin/hermes"
}

if has_wanted; then
    echo "hermes $VERSION already installed at $DEST, skipping..."
    echo "HERMES_INSTALL_OK"
    exit 0
fi

payload_override HERMES "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "hermes $VERSION installed at $DEST"
echo "HERMES_INSTALL_OK"
