#!/bin/bash
# Install the DeepSeek Harness (dsh) into the test pod from npm.
#
# No published binary and the bundle is a large plugin tree, so
# `@deepseek-ai/dsh@<version>` is installed with a pinned Node 22 (downloaded
# from nodejs.org) and run from its lib/bin.js. The search tool execs the rg
# that ships in @vscode/ripgrep's platform package.
#
# Two small source patches:
#   1. the launcher re-creates the HMR plugin and registers chokidar watchers
#      on user patch files; the watch error escapes its try/catch and kills the
#      run when the host's inotify budget is exhausted. A one-shot pod turn
#      never reloads its composition, so the block is disabled.
#   2. the headless runner always mints a fresh session id, so there is no way
#      to continue a session; honour $DSH_SESSION_ID and resume it when
#      $DSH_RESUME_SESSION=1 (set by the adapter, dsh.py).
#
# Lands:
#   /opt/mimo-dsh/bin/dsh        wrapper -> node app/node_modules/@deepseek-ai/dsh/lib/bin.js
#   /opt/mimo-dsh/{app,node}     the installed plugin tree + pinned node 22
#   /opt/mimo-dsh/.mimo-payload-version   installed-version marker (idempotency)
#
# Env knobs:
#   DSH_VERSION      npm package version (default "0.1.0-rc.6")
#   DSH_PAYLOAD / DSH_PAYLOAD_URL   prebuilt payload tarball (offline mode)
#   NPM_REGISTRY, NODE_DIST, NODE_MUSL_DIST   mirror overrides
set -euo pipefail
. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"

VERSION="${DSH_VERSION:-0.1.0-rc.6}"
NODE_VERSION="22.19.0"
DEST="/opt/mimo-dsh"
MARKER="$DEST/.mimo-payload-version"
mimo_detect_libc

probe() {
    bin/dsh --version
}

has_wanted() {
    mimo_marker_matches "$MARKER" "$VERSION" || return 1
    (cd "$DEST" && probe 2>&1) | grep -qF "$VERSION"
}

write_patch_script() {
    cat > "$DEST/patch-dsh.mjs" <<'PATCH'
import fs from "node:fs";
import path from "node:path";

const root = process.argv[2]; // .../node_modules/@deepseek-ai
const read = (f) => fs.readFileSync(f, "utf8");

// patch 1: drop the launcher's HMR + user-patch watchers
{
  const libDir = path.join(root, "dsh/lib");
  const needle = 'ctx.get("loader") !== void 0) try {';
  const boots = fs.readdirSync(libDir).filter((f) => /^profile-boot-.*\.js$/.test(f))
    .map((f) => path.join(libDir, f)).filter((f) => read(f).includes(needle));
  const done = fs.readdirSync(libDir).filter((f) => /^profile-boot-.*\.js$/.test(f))
    .map((f) => path.join(libDir, f)).filter((f) => read(f).includes("if (false) try {"));
  if (boots.length === 0 && done.length > 0) {
    console.log("dsh: launcher watchers already removed");
  } else {
    if (boots.length !== 1) throw new Error(`dsh: expected one profile-boot chunk with the watcher block, got ${boots.length}`);
    const anchor = 'if (!signalShutdown.signal.aborted && ctx.fiber.state === 2 && ctx.get("loader") !== void 0) try {';
    const text = read(boots[0]);
    if (text.split(anchor).length - 1 !== 1) throw new Error("dsh: watcher block anchor moved");
    fs.writeFileSync(boots[0], text.replace(anchor, "if (false) try {"));
    console.log("dsh: launcher watchers removed");
  }
}

// patch 2: resumable headless sessions
{
  const runner = path.join(root, "dsh-headless/lib/index.js");
  const text = read(runner);
  if (text.includes("DSH_RESUME_SESSION")) {
    console.log("dsh: headless resume already enabled");
  } else {
    const old = [
      "\tconst { agent } = await agents.create({",
      "\t\tsessionId: SessionId(`session-${randomUUID()}`),",
      "\t\tmeta: { cwd: process.cwd() },",
      "\t\tagentOptions: {",
      "\t\t\tprovider: selection.provider,",
      "\t\t\tmodel: selection.model",
      "\t\t},",
      "\t\tsetup: (agentCtx) => {",
      "\t\t\tinstallModelSelection(agentCtx, {",
      "\t\t\t\tcurrent: selection,",
      "\t\t\t\tassembled: void 0",
      "\t\t\t});",
      "\t\t}",
      "\t});",
    ].join("\n");
    if (text.split(old).length - 1 !== 1) throw new Error("dsh: headless agent-creation anchor moved");
    const next = [
      "\tconst dshEnvSessionId = process.env.DSH_SESSION_ID;",
      "\tconst dshSessionId = SessionId(dshEnvSessionId && dshEnvSessionId.length > 0 ? dshEnvSessionId : `session-${randomUUID()}`);",
      "\tconst dshAgentOptions = {",
      "\t\tprovider: selection.provider,",
      "\t\tmodel: selection.model",
      "\t};",
      "\tconst dshSetup = (agentCtx) => {",
      "\t\tinstallModelSelection(agentCtx, {",
      "\t\t\tcurrent: selection,",
      "\t\t\tassembled: void 0",
      "\t\t});",
      "\t};",
      '\tconst { agent } = process.env.DSH_RESUME_SESSION === "1" ? await agents.resume({',
      "\t\tresumeSessionId: dshSessionId,",
      "\t\tagentOptions: dshAgentOptions,",
      "\t\tsetup: dshSetup",
      "\t}) : await agents.create({",
      "\t\tsessionId: dshSessionId,",
      "\t\tmeta: { cwd: process.cwd() },",
      "\t\tagentOptions: dshAgentOptions,",
      "\t\tsetup: dshSetup",
      "\t});",
    ].join("\n");
    fs.writeFileSync(runner, text.replace(old, next));
    console.log("dsh: headless resume enabled");
  }
}
PATCH
}

install_public() {
    echo "Installing dsh $VERSION from npm (@deepseek-ai/dsh) + node $NODE_VERSION..."
    rm -rf "$DEST"
    mkdir -p "$DEST/bin" "$DEST/app"
    fetch_node "$NODE_VERSION" "$DEST/node"
    (
        export PATH="$DEST/node/bin:$PATH" npm_config_userconfig=/dev/null
        cd "$DEST/app" && npm install --omit=dev --no-audit --no-fund --loglevel=error \
            --registry "${NPM_REGISTRY:-https://registry.npmjs.org}" "@deepseek-ai/dsh@$VERSION"
    )
    write_patch_script
    "$DEST/node/bin/node" "$DEST/patch-dsh.mjs" "$DEST/app/node_modules/@deepseek-ai"
    # the search tool execs the rg that ships in @vscode/ripgrep-linux-x64
    "$DEST/app/node_modules/@vscode/ripgrep-linux-x64/bin/rg" --version >/dev/null
    mimo_wrapper "$DEST/bin/dsh" "$DEST/node/bin/node $DEST/app/node_modules/@deepseek-ai/dsh/lib/bin.js"
}

if has_wanted; then
    echo "dsh $VERSION already installed at $DEST, skipping..."
    echo "DSH_INSTALL_OK"
    exit 0
fi

payload_override DSH "$DEST" || install_public

(cd "$DEST" && mimo_verify_version "$VERSION" probe) || exit 1
mimo_marker_write "$MARKER" "$VERSION"

echo "dsh $VERSION installed at $DEST"
echo "DSH_INSTALL_OK"
