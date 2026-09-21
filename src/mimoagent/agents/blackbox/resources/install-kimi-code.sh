#!/bin/bash
# Install Kimi Code into the test pod from npm.
#
# `@moonshot-ai/kimi-code` ships dist/main.mjs as a self-contained bundle
# (only optional deps are native); it runs on a pinned Node 22 downloaded from
# nodejs.org (musl build from unofficial-builds.nodejs.org on Alpine). A static
# ripgrep is placed on PATH so the Grep tool never downloads one at run time.
#
# No source patch: the adapter overrides the built-in default agent profile
# through a discovered agent file named "agent" with `override: true`
# (kimi_code.py `_agent_md`). The self-check below fails the install if the
# three things that hook depends on stop holding.
#
# Lands:
#   /opt/mimo-kimi-code/bin/kimi      wrapper -> node app/dist/main.mjs
#   /opt/mimo-kimi-code/{app,node}    the app bundle + pinned node 22
#   /opt/mimo-kimi-code/bin/rg        ripgrep for the Grep tool
#   /opt/mimo-kimi-code/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   KIMI_CODE_VERSION      npm package version (default "0.36.1")
#   KIMI_CODE_PAYLOAD / KIMI_CODE_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY, NODE_DIST, NODE_MUSL_DIST, GITHUB_BASE   mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${KIMI_CODE_VERSION:-0.36.1}"
NODE_VERSION="22.19.0"
DEST="/opt/mimo-kimi-code"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/kimi --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

# The adapter's agent-file override hook depends on these three anchors.
check_override_hook() {
    local src="$DEST/app/dist/main.mjs"
    [ "$(grep -c 'DEFAULT_AGENT_PROFILE_NAME$1 = "agent"' "$src" || true)" -ge 1 ] \
        || { echo "kimi-code: default profile is no longer named 'agent'" >&2; return 1; }
    [ "$(grep -c 'definition.override || definition.source === "explicit"' "$src" || true)" -ge 2 ] \
        || { echo "kimi-code: file-defined profile override is no longer honoured" >&2; return 1; }
    [ "$(grep -c 'vars\["base_prompt"\] = basePrompt(context)' "$src" || true)" -ge 1 ] \
        || { echo "kimi-code: \${base_prompt} no longer embeds the builtin system prompt" >&2; return 1; }
    echo "kimi-code: agent-file override hook OK"
}

install_public() {
    echo "Installing kimi-code $VERSION from npm (@moonshot-ai/kimi-code) + node $NODE_VERSION..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_npm_tarball "@moonshot-ai/kimi-code" "$VERSION" "$DEST/app"
    check_override_hook
    fetch_node "$NODE_VERSION" "$DEST/node"
    fetch_rg "$DEST/bin/rg"
    mimo_wrapper "$DEST/bin/kimi" "$DEST/node/bin/node $DEST/app/dist/main.mjs"
}

if has_wanted; then
    echo "kimi-code $VERSION already installed at $DEST, skipping..."
    echo "KIMI_CODE_INSTALL_OK"
    exit 0
fi

payload_override KIMI_CODE "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "kimi-code $VERSION installed at $DEST"
echo "KIMI_CODE_INSTALL_OK"
