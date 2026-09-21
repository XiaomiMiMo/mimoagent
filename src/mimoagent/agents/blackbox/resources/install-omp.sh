#!/bin/bash
# Install Oh My Pi (omp) into the test pod from npm.
#
# `@oh-my-pi/pi-coding-agent` targets Bun (engines.bun >= 1.3.14): a pinned Bun
# runtime is downloaded from the oven-sh/bun GitHub release, the package's
# production dependencies are installed with it, and omp runs as
# `bun dist/cli.js`. Two small source patches gate request fields that only an
# Anthropic-native route accepts behind env vars the adapter controls (omp.py):
#   1. the context-management beta (`context_management: {edits: [...]}`)
#      is sent only when OMP_CONTEXT_MANAGEMENT is set;
#   2. `strict: true` on tool definitions only when OMP_TOOL_STRICT is set.
# A prebuilt payload (see scripts/harness_payloads/) is considerably smaller
# than the in-pod dependency install; prefer it where install time matters.
#
# Lands:
#   /opt/mimo-omp/bin/omp        wrapper -> bun app/dist/cli.js
#   /opt/mimo-omp/{app,bun}      the app tree (+ node_modules) and the bun runtime
#   /opt/mimo-omp/natives/       pi_natives .node files (staged into $HOME/.omp by probe)
#   /opt/mimo-omp/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   OMP_VERSION       npm package version (default "17.3.4")
#   OMP_BUN_VERSION   bun release (default "1.4.2")
#   OMP_PAYLOAD / OMP_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY, GITHUB_BASE       mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${OMP_VERSION:-17.3.4}"
BUN_VERSION="${OMP_BUN_VERSION:-1.4.2}"
DEST="/opt/mimo-omp"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    # The binary loads pi_natives from $HOME/.omp/natives/<ver>/, so stage them
    # before probing (idempotent, and the run needs them too).
    mkdir -p "$HOME/.omp/natives/$VERSION"
    cp -n natives/*.node "$HOME/.omp/natives/$VERSION/" 2>/dev/null || true
    bin/omp --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

fetch_bun() {
    local asset="bun-linux-x64.zip" tmp dir
    [ "$MIMO_LIBC" = "musl" ] && asset="bun-linux-x64-musl.zip"
    command -v unzip >/dev/null 2>&1 || mimo_pkg_install unzip
    tmp="$(mktemp)"
    fetch_github_release "oven-sh/bun" "bun-v${BUN_VERSION}" "$asset" "$tmp"
    dir="$(mktemp -d)"
    unzip -q "$tmp" -d "$dir"
    mkdir -p "$DEST/bun"
    mv "$dir"/bun-linux-x64*/bun "$DEST/bun/bun"
    chmod 0755 "$DEST/bun/bun"
    rm -rf "$tmp" "$dir"
    "$DEST/bun/bun" --version >/dev/null
}

write_patch_script() {
    cat > "$DEST/patch-omp.mjs" <<'PATCH'
import fs from "node:fs";

const path = process.argv[2];
let src = fs.readFileSync(path, "utf8");
const count = (s) => src.split(s).length - 1;

// context_management -> $OMP_CONTEXT_MANAGEMENT
{
  const old = '?{edits:[{type:"clear_thinking_20251015",keep:"all"}]}:void 0';
  const next = "&&process.env.OMP_CONTEXT_MANAGEMENT" + old;
  if (count(next) === 1) console.log("omp: context_management already patched");
  else {
    if (count(old) !== 1) throw new Error(`omp: context_management anchor moved (${count(old)} hits)`);
    src = src.replace(old, next);
    console.log("omp: context_management -> $OMP_CONTEXT_MANAGEMENT");
  }
}
// tools[].strict -> $OMP_TOOL_STRICT
{
  const old = "...c.strict?{strict:!0}:{}";
  const next = "...c.strict&&process.env.OMP_TOOL_STRICT?{strict:!0}:{}";
  if (count(next) === 1) console.log("omp: tool strict already patched");
  else {
    if (count(old) !== 1) throw new Error(`omp: tool strict anchor moved (${count(old)} hits)`);
    src = src.replace(old, next);
    console.log("omp: tools[].strict -> $OMP_TOOL_STRICT");
  }
}
fs.writeFileSync(path, src);
PATCH
}

install_public() {
    echo "Installing omp $VERSION from npm (@oh-my-pi/pi-coding-agent) + bun $BUN_VERSION..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin" "$DEST/natives"
    fetch_npm_tarball "@oh-my-pi/pi-coding-agent" "$VERSION" "$DEST/app"
    fetch_bun
    write_patch_script
    "$DEST/bun/bun" "$DEST/patch-omp.mjs" "$DEST/app/dist/cli.js"
    (
        export HOME="$DEST/.home"; mkdir -p "$HOME"
        # Optional dependencies stay in: the platform native package
        # (@oh-my-pi/pi-natives-linux-x64) that omp dlopens is one of them.
        cd "$DEST/app" && "$DEST/bun/bun" install --production --no-progress \
            --registry "${NPM_REGISTRY:-https://registry.npmjs.org}" >/dev/null
    )
    cp "$DEST"/app/node_modules/@oh-my-pi/pi-natives-linux-x64/pi_natives.linux-x64-*.node "$DEST/natives/" 2>/dev/null || true
    mimo_wrapper "$DEST/bin/omp" "$DEST/bun/bun $DEST/app/dist/cli.js"
}

if has_wanted; then
    echo "omp $VERSION already installed at $DEST, skipping..."
    echo "OMP_INSTALL_OK"
    exit 0
fi

payload_override OMP "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "omp $VERSION installed at $DEST"
echo "OMP_INSTALL_OK"
