#!/bin/bash
# Install OpenClaw into the test pod from npm.
#
# No published binary: the `openclaw` package is unpacked, its production
# dependencies installed with a pinned Node 24 (downloaded from nodejs.org),
# and three small source patches applied. All three are the same class of
# fix — upstream allow-lists that predate the model families this harness is
# run with — and each is gated on an env var the adapter sets (openclaw.py):
#   1. the adaptive/xhigh thinking whitelist gains `opus-5`, so
#      `--thinking xhigh|max` reaches the wire as adaptive + output_config.effort;
#   2. `OPENCLAW_USE_MODEL_MAX_TOKENS=1` honours the model's declared maxTokens
#      instead of clamping non-Sonnet-5 models to 32000;
#   3. `shouldPreserveThinkingBlocks()` recognises `claude-opus-5`, so prior
#      turns keep their signed thinking blocks (prompt cache + trajectory).
#
# Lands:
#   /opt/mimo-openclaw/bin/openclaw   wrapper -> node app/openclaw.mjs
#   /opt/mimo-openclaw/{app,node}     the npm app tree + pinned node 24
#   /opt/mimo-openclaw/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   OPENCLAW_VERSION      npm package version (default "2026.7.1-2")
#   OPENCLAW_PAYLOAD / OPENCLAW_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY, NODE_DIST, NODE_MUSL_DIST   mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${OPENCLAW_VERSION:-2026.7.1-2}"
NODE_VERSION="24.15.0"
DEST="/opt/mimo-openclaw"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/openclaw --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

# Source patches, applied with the fetched node. Idempotent: each patch
# recognises its own output and skips; a moved anchor fails the install.
write_patch_script() {
    cat > "$DEST/patch-openclaw.mjs" <<'PATCH'
import fs from "node:fs";
import path from "node:path";

const root = process.argv[2];
const files = [
  ...fs.readdirSync(path.join(root, "dist")).filter((f) => f.endsWith(".js")).map((f) => path.join(root, "dist", f)),
];
const aiDist = path.join(root, "node_modules/@openclaw/ai/dist");
if (fs.existsSync(aiDist)) {
  files.push(...fs.readdirSync(aiDist).filter((f) => f.endsWith(".mjs")).map((f) => path.join(aiDist, f)));
}
const read = (f) => fs.readFileSync(f, "utf8");

// Patch 1: adaptive/xhigh whitelists gain opus-5 (three regex spellings).
const whitelist = [
  ["fable-5|mythos-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)", "fable-5|mythos-5|opus-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)"],
  ["fable-5|mythos-5|opus-4-(?:7|8)|sonnet-5", "fable-5|mythos-5|opus-5|opus-4-(?:7|8)|sonnet-5"],
  ["fable-5|mythos-(?:5|preview)|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)", "fable-5|mythos-(?:5|preview)|opus-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)"],
];
let touched = 0;
for (const f of files) {
  let text = read(f);
  if (!text.includes("opus-4-(?:6|7|8)") && !text.includes("opus-4-(?:7|8)")) continue;
  let next = text;
  for (const [from, to] of whitelist) next = next.split(from).join(to);
  if (!next.includes("opus-5|opus-4")) throw new Error(`openclaw whitelist patch failed on ${f}`);
  if (next !== text) { fs.writeFileSync(f, next); touched++; }
}
console.log(`openclaw: thinking whitelist patched in ${touched} file(s)`);

// Patch 3: thinking-preservation allowlist.
{
  const anchor = 'id.includes("fable-5") || id.includes("opus-4")';
  const patched = 'id.includes("fable-5") || id.includes("opus-5") || id.includes("opus-4")';
  const hits = files.filter((f) => read(f).includes(anchor));
  const done = files.filter((f) => read(f).includes(patched));
  if (!hits.length && !done.length) throw new Error("openclaw: thinking-preservation allowlist anchor moved");
  for (const f of hits) {
    const text = read(f);
    if (text.split(anchor).length - 1 !== 1) throw new Error(`${path.basename(f)}: unexpected anchor count`);
    fs.writeFileSync(f, text.replace(anchor, patched));
    console.log(`openclaw: patched thinking-preservation allowlist in ${path.basename(f)}`);
  }
}

// Patch 2: max_tokens clamp behind OPENCLAW_USE_MODEL_MAX_TOKENS.
{
  const anchor = "useModelDefault: resolveClaudeSonnet5ModelIdentity(model) !== void 0";
  const patched = "useModelDefault: !!process.env.OPENCLAW_USE_MODEL_MAX_TOKENS || resolveClaudeSonnet5ModelIdentity(model) !== void 0";
  const hits = files.filter((f) => read(f).includes(anchor) || read(f).includes(patched));
  if (!hits.length) throw new Error("openclaw: max_tokens clamp anchor moved");
  for (const f of hits) {
    const text = read(f);
    if (text.includes(patched)) continue;
    if (text.split(anchor).length - 1 !== 1) throw new Error(`${path.basename(f)}: unexpected anchor count`);
    fs.writeFileSync(f, text.replace(anchor, patched));
    console.log(`openclaw: patched max_tokens clamp in ${path.basename(f)}`);
  }
}
PATCH
}

install_public() {
    echo "Installing openclaw $VERSION from npm + node $NODE_VERSION..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin"
    fetch_npm_tarball "openclaw" "$VERSION" "$DEST/app"
    fetch_node "$NODE_VERSION" "$DEST/node"
    (
        export PATH="$DEST/node/bin:$PATH" npm_config_userconfig=/dev/null
        cd "$DEST/app" && npm install --omit=dev --no-audit --no-fund --loglevel=error \
            --registry "${NPM_REGISTRY:-https://registry.npmjs.org}"
    )
    write_patch_script
    "$DEST/node/bin/node" "$DEST/patch-openclaw.mjs" "$DEST/app"
    mimo_wrapper "$DEST/bin/openclaw" "$DEST/node/bin/node $DEST/app/openclaw.mjs"
}

if has_wanted; then
    echo "openclaw $VERSION already installed at $DEST, skipping..."
    echo "OPENCLAW_INSTALL_OK"
    exit 0
fi

payload_override OPENCLAW "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "openclaw $VERSION installed at $DEST"
echo "OPENCLAW_INSTALL_OK"
