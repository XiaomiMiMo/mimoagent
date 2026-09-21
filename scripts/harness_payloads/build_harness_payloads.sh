#!/bin/bash
# Build self-contained payload tarballs for the blackbox harness adapters.
#
# The installers (src/mimoagent/agents/blackbox/resources/install-<name>.sh)
# install each harness from its upstream public source by default. This
# builder is for the OFFLINE path: it produces a tarball per harness that
# unpacks into /opt/mimo-<name> and carries everything the harness needs —
# a single upstream binary where one exists, otherwise a bundled runtime
# (node/bun/python) plus the installed app tree, with the same source patches
# the installers apply. Point an adapter at the result with
# ``payload_path`` (uploaded into the pod) or ``payload_url`` (your mirror).
#
# Usage: build_harness_payloads.sh [name ...]   (default: all)
# Artifacts land in $BBH_OUT as <name>-<version>-linux-x64.tar.gz.
#
# Sources are cached under $BBH_CACHE (npm tarballs, runtimes). Re-running a
# build reuses the cache; delete it to force a re-download. Requires node,
# python3, bun (for omp) and curl on the build host; builds glibc (linux-x64)
# payloads — build on an Alpine host with the same script for musl images.
set -euo pipefail

BBH_CACHE="${BBH_CACHE:-/tmp/bbh}"
BBH_OUT="${BBH_OUT:-./dist/harness_payloads}"
PKGS="$BBH_CACHE/pkgs"
RUNTIMES="$BBH_CACHE/runtimes"
STAGE="$BBH_CACHE/stage"
NPM_REG="${NPM_REG:-https://registry.npmjs.org}"
NODE_DIST="${NODE_DIST:-https://nodejs.org/dist}"
PBS_BASE="${PBS_BASE:-https://github.com/astral-sh/python-build-standalone/releases/download}"
GITHUB_BASE="${GITHUB_BASE:-https://github.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
CPYTHON_VERSION=3.12.12
CPYTHON_TAG=20260114

GROK_VERSION=1.0.4
OPENCODE_VERSION=1.18.18
KIMI_CODE_VERSION=0.36.1
OPENCLAW_VERSION=2026.7.1-2
OMP_VERSION=17.3.4
KIMI_CLI_VERSION=1.49.0
HERMES_VERSION=2026.8.13
KILOCODE_VERSION=7.4.22
MSWEA_VERSION=2.4.6
# payload revision: bumped when a build-time patch changes the binary but the
# upstream version does not (rev 3 = prompt-cache breakpoint fix)
DSH_VERSION=0.1.0-rc.6
NODE22=22.19.0
NODE24=24.15.0
RG_VERSION=15.1.0

mkdir -p "$PKGS" "$RUNTIMES" "$STAGE" "$BBH_OUT"

fetch_npm() { # name version outdir -> extracts package/ under outdir
    local name="$1" ver="$2" out="$PKGS/$3"
    # The cache is keyed by name only, so a tree left over from an earlier
    # version pin has to be discarded rather than silently reused.
    if [ -f "$out/package/package.json" ]; then
        local have
        have=$(python3 -c "import json;print(json.load(open('$out/package/package.json')).get('version',''))")
        [ "$have" = "$ver" ] && return 0
        echo "  cache holds $name@$have, want $ver — refetching"
        rm -rf "$out"
    fi
    mkdir -p "$out"
    local tarball
    tarball=$(curl -fsSL "$NPM_REG/$name" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['versions']['$ver']['dist']['tarball'])")
    curl -fsSL "$tarball" -o "$out/pkg.tgz"
    tar -xzf "$out/pkg.tgz" -C "$out"
}

fetch_node() { # version
    local v="$1" tarball="$RUNTIMES/node-v$1-linux-x64.tar.xz"
    [ -d "$RUNTIMES/node-v$v-linux-x64" ] && return 0
    curl -fsSL --retry 3 -o "$tarball" "$NODE_DIST/v$v/node-v$v-linux-x64.tar.xz"
    tar -xf "$tarball" -C "$RUNTIMES"
}

fetch_cpython() { # -> $RUNTIMES/cpython-<ver>.tar.gz (astral-sh/python-build-standalone)
    local tarball="$RUNTIMES/cpython-$CPYTHON_VERSION.tar.gz"
    [ -f "$tarball" ] && return 0
    curl -fsSL --retry 3 -o "$tarball" \
        "$PBS_BASE/$CPYTHON_TAG/cpython-$CPYTHON_VERSION+$CPYTHON_TAG-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
}

fetch_rg() { # -> $RUNTIMES/rg (static musl build of ripgrep)
    [ -x "$RUNTIMES/rg" ] && return 0
    local tarball="$RUNTIMES/ripgrep-$RG_VERSION.tar.gz" dir
    curl -fsSL --retry 3 -o "$tarball" \
        "$GITHUB_BASE/BurntSushi/ripgrep/releases/download/$RG_VERSION/ripgrep-$RG_VERSION-x86_64-unknown-linux-musl.tar.gz"
    dir="$(mktemp -d)"
    tar -xzf "$tarball" -C "$dir" --strip-components=1
    mv "$dir/rg" "$RUNTIMES/rg"
    rm -rf "$dir"
    chmod +x "$RUNTIMES/rg"
}

pack() { # stagedir outfile — tar the *contents* of stagedir
    local dir="$1" out="$BBH_OUT/$2"
    tar -czf "$out.tmp" -C "$dir" .
    mv "$out.tmp" "$out"
    echo "built $out ($(du -h "$out" | cut -f1))"
}

wrapper() { # path shebang-lines...
    local path="$1"; shift
    mkdir -p "$(dirname "$path")"
    printf '#!/bin/sh\nexec %s "$@"\n' "$*" > "$path"
    chmod +x "$path"
}

build_grok() {
    # Upstream ships a brotli-compressed static binary (no glibc deps at all).
    fetch_npm "@xai-official/grok-linux-x64" "$GROK_VERSION" grok-bin
    local s="$STAGE/grok"; rm -rf "$s"; mkdir -p "$s/bin"
    if [ ! -x "$PKGS/grok-bin/package/bin/grok" ]; then
        node -e '
const zlib=require("zlib"),fs=require("fs");
fs.createReadStream(process.argv[1]).pipe(zlib.createBrotliDecompress())
  .pipe(fs.createWriteStream(process.argv[2])).on("finish",()=>{});
' "$PKGS/grok-bin/package/bin/grok.br" "$PKGS/grok-bin/package/bin/grok"
        sleep 1; chmod +x "$PKGS/grok-bin/package/bin/grok"
    fi
    cp "$PKGS/grok-bin/package/bin/grok" "$s/bin/grok"
    "$s/bin/grok" --version | grep -F "$GROK_VERSION"
    pack "$s" "grok-$GROK_VERSION-linux-x64.tar.gz"
}

build_opencode() {
    # Upstream ships a bun-compiled standalone binary per platform.
    fetch_npm "opencode-linux-x64" "$OPENCODE_VERSION" opencode-bin
    local s="$STAGE/opencode"; rm -rf "$s"; mkdir -p "$s/bin"
    cp "$PKGS/opencode-bin/package/bin/opencode" "$s/bin/opencode"
    chmod +x "$s/bin/opencode"
    "$s/bin/opencode" --version | grep -F "$OPENCODE_VERSION"
    pack "$s" "opencode-$OPENCODE_VERSION-linux-x64.tar.gz"
}

build_kimi_code() {
    # dist/main.mjs is a self-contained bundle (only optionalDeps are native);
    # ship it with a pinned node runtime.
    #
    # NO SOURCE PATCH from 0.36.1 on: the tool list is no longer taken from the
    # compiled-in default profile literal (patching it changed nothing on the
    # wire), and 0.36.1 gained a supported hook instead — a discovered agent
    # file named "agent" with `override: true` replaces the builtin default
    # profile for every session, resumes included. The adapter stages that file
    # (kimi_code.py `_agent_md`). The self-check below fails the build if the
    # three things that hook depends on stop holding.
    fetch_npm "@moonshot-ai/kimi-code" "$KIMI_CODE_VERSION" kimi-code
    fetch_node "$NODE22"
    fetch_rg
    local s="$STAGE/kimi-code"; rm -rf "$s"; mkdir -p "$s/app"
    cp -r "$PKGS/kimi-code/package/." "$s/app/"
    python3 - "$s/app/dist/main.mjs" <<'PY'
import sys

src = open(sys.argv[1], encoding="utf8").read()
checks = {
    # 1. the default profile the adapter's agent file overrides is still "agent"
    'DEFAULT_AGENT_PROFILE_NAME$1 = "agent"': 1,
    # 2. `override` is still honoured for file-defined profiles
    'definition.override || definition.source === "explicit"': 2,
    # 3. ${base_prompt} still embeds the builtin system prompt
    'vars["base_prompt"] = basePrompt(context)': 1,
}
missing = [k for k, n in checks.items() if src.count(k) < n]
assert not missing, f"kimi-code: agent-file override hook changed: {missing}"
print("kimi-code: agent-file override hook OK")
PY
    cp -r "$RUNTIMES/node-v$NODE22-linux-x64" "$s/node"
    # rg on PATH keeps the Grep tool from downloading one at runtime
    cp "$RUNTIMES/rg" "$s/bin/rg" 2>/dev/null || { mkdir -p "$s/bin"; cp "$RUNTIMES/rg" "$s/bin/rg"; }
    wrapper "$s/bin/kimi" "/opt/mimo-kimi-code/node/bin/node /opt/mimo-kimi-code/app/dist/main.mjs"
    "$s/node/bin/node" "$s/app/dist/main.mjs" --version | grep -F "$KIMI_CODE_VERSION"
    pack "$s" "kimi-code-$KIMI_CODE_VERSION-linux-x64.tar.gz"
}

build_openclaw() {
    # No published binary; install prod deps locally and ship the whole tree
    # with a pinned node 24.
    #
    # Build-time patch: openclaw's Claude adaptive/xhigh whitelists predate
    # claude-opus-5 (fable-5|mythos-5|opus-4-(6|7|8)|sonnet-...), so xhigh/max
    # would clamp to budget_tokens. Extending them with opus-5 makes
    # `--thinking xhigh|max` reach the wire as adaptive + output_config.effort
    # (verified against a local mock gateway). Two copies exist: the CLI dist
    # chunk and @openclaw/ai (the one the transport actually imports).
    fetch_npm openclaw "$OPENCLAW_VERSION" openclaw
    fetch_node "$NODE24"
    local app="$PKGS/openclaw/package"
    if [ ! -d "$app/node_modules" ]; then
        (export PATH="$RUNTIMES/node-v$NODE24-linux-x64/bin:$PATH" \
                npm_config_userconfig=/dev/null
         cd "$app" && npm install --omit=dev --no-audit --no-fund \
             --loglevel=error --registry "$NPM_REG")
    fi
    for f in "$app/dist/src-"*.js "$app"/node_modules/@openclaw/ai/dist/src-*.mjs; do
        grep -q "opus-4-(?:6|7|8)" "$f" 2>/dev/null || continue
        sed -i \
            -e 's/fable-5|mythos-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)/fable-5|mythos-5|opus-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)/g' \
            -e 's/fable-5|mythos-5|opus-4-(?:7|8)|sonnet-5/fable-5|mythos-5|opus-5|opus-4-(?:7|8)|sonnet-5/g' \
            -e 's/fable-5|mythos-(?:5|preview)|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)/fable-5|mythos-(?:5|preview)|opus-5|opus-4-(?:6|7|8)|sonnet-(?:5|4-6)/g' \
            "$f"
        grep -q "opus-5|opus-4" "$f" || { echo "openclaw whitelist patch failed on $f" >&2; return 1; }
    done
    # Patch 3: `shouldPreserveThinkingBlocks()` decides whether prior turns keep
    # their thinking blocks, and it recognises new Claude families only as
    # `claude-[5-9]...` — in `claude-opus-5` the char after "claude-" is "o", so
    # it returns false and `dropThinkingBlocks` strips the signed thinking of
    # every assistant turn but the latest. That rewrites the cached prefix each
    # turn (measured: cache_read flat across 3 requests) and loses the reasoning
    # from the trajectory. Upstream's own comment says dropping "breaks replay
    # and prompt caching" — the model just is not on the list.
    python3 - "$app" <<'PYPATCH'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
anchor = 'id.includes("fable-5") || id.includes("opus-4")'
patched = 'id.includes("fable-5") || id.includes("opus-5") || id.includes("opus-4")'
files = list(root.glob("dist/*.js")) + list(root.glob("node_modules/@openclaw/ai/dist/*.mjs"))
hits = [f for f in files if anchor in f.read_text(errors="replace")]
done = [f for f in files if patched in f.read_text(errors="replace")]
assert hits or done, "openclaw: thinking-preservation allowlist anchor moved"
for f in hits:
    text = f.read_text()
    assert text.count(anchor) == 1, f"{f.name}: {text.count(anchor)} matches"
    f.write_text(text.replace(anchor, patched))
    print(f"patched thinking-preservation allowlist in {f.name}")
PYPATCH
    # Patch 2: the Anthropic transport clamps the model's declared maxTokens to
    # 32000 unless the model is recognised as Claude Sonnet 5
    # (`Math.min(modelMax, 32e3)` with `useModelDefault` gated on a
    # Sonnet-5-only identity check), so agent.output_limit above 32000 is
    # silently dropped on the wire. OPENCLAW_USE_MODEL_MAX_TOKENS=1 (set by the
    # adapter) honours the declared value instead — claude-opus-5 serves up to
    # 128000 on this channel.
    python3 - "$app" <<'PYPATCH'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
anchor = "useModelDefault: resolveClaudeSonnet5ModelIdentity(model) !== void 0"
patched = ('useModelDefault: !!process.env.OPENCLAW_USE_MODEL_MAX_TOKENS '
           '|| resolveClaudeSonnet5ModelIdentity(model) !== void 0')
hits = [f for f in list(root.glob("dist/*.js")) + list(root.glob("node_modules/@openclaw/ai/dist/*.mjs"))
        if anchor in f.read_text(errors="replace") or patched in f.read_text(errors="replace")]
assert hits, "openclaw: max_tokens clamp anchor moved"
for f in hits:
    text = f.read_text()
    if patched in text:
        continue
    assert text.count(anchor) == 1, f"{f.name}: {text.count(anchor)} matches"
    f.write_text(text.replace(anchor, patched))
    print(f"patched max_tokens clamp in {f.name}")
PYPATCH
    local s="$STAGE/openclaw"; rm -rf "$s"; mkdir -p "$s/app"
    cp -r "$app/." "$s/app/"
    cp -r "$RUNTIMES/node-v$NODE24-linux-x64" "$s/node"
    wrapper "$s/bin/openclaw" "/opt/mimo-openclaw/node/bin/node /opt/mimo-openclaw/app/openclaw.mjs"
    "$s/node/bin/node" "$s/app/openclaw.mjs" --version | grep -F "$OPENCLAW_VERSION"
    pack "$s" "openclaw-$OPENCLAW_VERSION-linux-x64.tar.gz"
}

build_omp() {
    # dist/cli.js is NOT a full bundle (pi-natives, transformers, readability
    # etc. are externalized) and the node_modules tree runs to ~1GB, so
    # compile a single binary with bun instead (same as the official release,
    # minus the monorepo-only embedding of natives). The pi_natives .node
    # files ship alongside and are pre-staged by the adapter into
    # $HOME/.omp/natives/<version>/ where the binary expects them.
    #
    # Build-time patch: omp sends two fields that only an Anthropic-native
    # route accepts, and a Bedrock-backed one rejects the whole request over
    # either of them (HTTP 400 "Extra inputs are not permitted"), taking the
    # arm down entirely on gateways that fan out that way:
    #   1. the context-management beta
    #      (`context_management: {edits:[{type: clear_thinking_20251015, keep: all}]}`),
    #      attached to every adaptive/enabled-thinking request, gated upstream
    #      only on three hardcoded provider names;
    #   2. `strict: true` on tool definitions whose schema qualifies (`edit`).
    # Neither has a config or CLI hook, so both go behind an env var and the
    # adapter decides (default: off, i.e. omitted — see omp.py).
    fetch_npm "@oh-my-pi/pi-coding-agent" "$OMP_VERSION" omp
    local app="$PKGS/omp/package"
    python3 - "$app/dist/cli.js" <<'PY'
import sys

path = sys.argv[1]
patches = [
    ('?{edits:[{type:"clear_thinking_20251015",keep:"all"}]}:void 0',
     '&&process.env.OMP_CONTEXT_MANAGEMENT',
     "context_management -> $OMP_CONTEXT_MANAGEMENT"),
    ('...c.strict?{strict:!0}:{}',
     None,  # rewritten below (the gate goes inside the ternary condition)
     "tools[].strict -> $OMP_TOOL_STRICT"),
]
src = open(path, encoding="utf8").read()
done = []

old, prefix, label = patches[0]
new = prefix + old
if src.count(new) == 1:
    done.append(f"{label} (already patched)")
else:
    assert src.count(old) == 1, f"omp: context_management anchor moved ({src.count(old)} hits)"
    src = src.replace(old, new, 1)
    done.append(label)

old, _, label = patches[1]
new = "...c.strict&&process.env.OMP_TOOL_STRICT?{strict:!0}:{}"
if src.count(new) == 1:
    done.append(f"{label} (already patched)")
else:
    assert src.count(old) == 1, f"omp: tool strict anchor moved ({src.count(old)} hits)"
    src = src.replace(old, new, 1)
    done.append(label)

open(path, "w", encoding="utf8").write(src)
for line in done:
    print(f"omp: {line}")
PY
    (export HOME="$BBH_CACHE/fakehome"; mkdir -p "$HOME"
     cd "$app" && bun install --production --no-progress \
         --registry "$NPM_REG" >/dev/null)
    local s="$STAGE/omp"; rm -rf "$s"; mkdir -p "$s/bin" "$s/natives"
    (export HOME="$BBH_CACHE/fakehome"
     cd "$app" && bun build --compile dist/cli.js \
        --external omp-legacy-pi-modules --external fastembed \
        --external onnxruntime-node \
        --outfile "$s/bin/omp" >/dev/null)
    cp "$app"/node_modules/@oh-my-pi/pi-natives-linux-x64/pi_natives.linux-x64-*.node "$s/natives/"
    (export HOME="$BBH_CACHE/fakehome"
     mkdir -p "$HOME/.omp/natives/$OMP_VERSION"
     cp "$s/natives/"*.node "$HOME/.omp/natives/$OMP_VERSION/"
     "$s/bin/omp" --version | grep -F "$OMP_VERSION")
    pack "$s" "omp-$OMP_VERSION-linux-x64.tar.gz"
}

build_kimi_cli() {
    # Python >=3.12 app: standalone CPython + venv built at the exact /opt
    # prefix it will live at in the pod (venvs are not relocatable), plus a
    # static rg so the Grep tool never dials cdn.kimi.com at runtime.
    #
    # Three build-time patches. The first two gate xhigh/max adaptive thinking
    # (upstream only ever sends effort=high, and its model-name regex predates
    # claude-opus-5); the third unpins the output ceiling:
    #   1. kosong: treat opus-5/fable ids as adaptive-capable, allow xhigh
    #   2. kimi_cli: read the effort level from $KIMI_CLI_THINKING_EFFORT
    #   3. kimi_cli: read max_tokens from $KIMI_CLI_MAX_TOKENS (upstream pins
    #      the anthropic provider at 50000)
    fetch_cpython
    fetch_rg
    local prefix=/opt/mimo-kimi-cli
    rm -rf "$prefix"; mkdir -p "$prefix"
    tar -xzf "$RUNTIMES/cpython-$CPYTHON_VERSION.tar.gz" -C "$prefix"
    "$prefix/python/bin/python3.12" -m venv "$prefix/venv"
    PIP_CONFIG_FILE=/dev/null "$prefix/venv/bin/pip" install --quiet \
        -i "$PIP_INDEX_URL" \
        "kimi-cli==$KIMI_CLI_VERSION"

    local sp
    sp=$("$prefix/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # patch 1a: adaptive marker for unversioned opus-5/fable ids
    grep -c '_ADAPTIVE_MARKERS_NO_VERSION: tuple\[str, \.\.\.\] = ("mythos",)' \
        "$sp/kosong/contrib/chat_provider/anthropic.py" | grep -qx 1
    sed -i 's/_ADAPTIVE_MARKERS_NO_VERSION: tuple\[str, \.\.\.\] = ("mythos",)/_ADAPTIVE_MARKERS_NO_VERSION: tuple[str, ...] = ("mythos", "opus-5", "fable")/' \
        "$sp/kosong/contrib/chat_provider/anthropic.py"
    # patch 1b: adaptive family accepts xhigh too (upstream pins it to 4.7)
    grep -c 'return frozenset({"low", "medium", "high", "max"})' \
        "$sp/kosong/contrib/chat_provider/anthropic.py" | grep -qx 1
    sed -i 's/return frozenset({"low", "medium", "high", "max"})/return frozenset({"low", "medium", "high", "xhigh", "max"})/' \
        "$sp/kosong/contrib/chat_provider/anthropic.py"
    # patch 2: effort level from env (llm.py already imports os)
    grep -c 'chat_provider.with_thinking("high")' "$sp/kimi_cli/llm.py" | grep -qx 1
    sed -i 's/chat_provider.with_thinking("high")/chat_provider.with_thinking(os.environ.get("KIMI_CLI_THINKING_EFFORT", "high"))/' \
        "$sp/kimi_cli/llm.py"
    # patch 3: the anthropic chat provider is constructed with a hardcoded
    # default_max_tokens=50000, so agent.output_limit could never reach the
    # wire (claude-opus-5 serves up to 128000 on this channel).
    grep -c 'default_max_tokens=50000' "$sp/kimi_cli/llm.py" | grep -qx 1
    sed -i 's/default_max_tokens=50000/default_max_tokens=int(os.environ.get("KIMI_CLI_MAX_TOKENS") or 50000)/' \
        "$sp/kimi_cli/llm.py"
    "$prefix/venv/bin/python" -c "import kimi_cli, kosong"

    mkdir -p "$prefix/bin"
    cp "$RUNTIMES/rg" "$prefix/bin/rg"
    wrapper "$prefix/bin/kimi" "/opt/mimo-kimi-cli/venv/bin/kimi"
    "$prefix/bin/kimi" --version | grep -F "$KIMI_CLI_VERSION"
    pack "$prefix" "kimi-cli-$KIMI_CLI_VERSION-linux-x64.tar.gz"
}

fetch_hermes_src() { # -> $PKGS/hermes holds the v$HERMES_VERSION source tree
    local out="$PKGS/hermes" tag="v$HERMES_VERSION"
    local archive="$PKGS/hermes-$tag.tar.gz"
    # The tag is a date while pyproject carries a semver, so the extracted tree
    # is stamped with the tag it came from instead.
    if [ -f "$out/.mimo-src-tag" ] && grep -qxF "$tag" "$out/.mimo-src-tag"; then
        return 0
    fi
    if [ ! -s "$archive" ]; then
        # ~64 MB over a slow route: resume across retries rather than restart.
        curl -fL --retry 5 --retry-all-errors -C - --max-time 3600 \
            -o "$archive" \
            "$GITHUB_BASE/NousResearch/hermes-agent/archive/refs/tags/$tag.tar.gz"
    fi
    rm -rf "$out"; mkdir -p "$out"
    tar -xzf "$archive" -C "$out" --strip-components=1
    [ -f "$out/pyproject.toml" ] || { echo "hermes source extract failed" >&2; return 1; }
    echo "$tag" > "$out/.mimo-src-tag"
}

build_hermes() {
    # Python app from the GitHub tag (not on PyPI). Same standalone-CPython
    # scheme as kimi-cli. The [anthropic] extra is mandatory (the SDK is not
    # a core dep; without it the pod would lazy-pip-install at runtime).
    #
    # Build-time patch: hermes strips ALL historic thinking blocks whenever
    # the base_url lacks an "anthropic.com" substring (third-party
    # heuristic), which breaks signature replay against internal gateways.
    # Add an env escape hatch (HERMES_FORCE_NATIVE_ANTHROPIC=1, set by the
    # adapter) — auth headers / betas / fast-mode are unaffected for
    # non-OAuth-shaped keys (verified against all three call sites).
    fetch_cpython
    fetch_rg
    fetch_hermes_src
    local src="$PKGS/hermes"
    local prefix=/opt/mimo-hermes
    rm -rf "$prefix"; mkdir -p "$prefix"
    tar -xzf "$RUNTIMES/cpython-$CPYTHON_VERSION.tar.gz" -C "$prefix"
    "$prefix/python/bin/python3.12" -m venv "$prefix/venv"
    # 2026.8.13 (0.20.1) added a setup.py guard that refuses wheel/sdist builds
    # outside Nix, because a wheel silently drops the bundled assets (locales,
    # skills, optional-mcps, web_dist, tui_dist, plugin manifests) that are
    # resolved from the source-checkout layout at runtime. So install editable
    # from a source tree shipped *inside* the payload prefix: the path is
    # identical in the pod, and hermes reports it as its install directory.
    mkdir -p "$prefix/src"
    cp -r "$src/." "$prefix/src/"
    # docs + marketing website are ~57 MB of the 162 MB tree and nothing in the
    # headless path reads them.
    rm -rf "$prefix/src/website" "$prefix/src/docs" "$prefix/src/.git"
    PIP_CONFIG_FILE=/dev/null "$prefix/venv/bin/pip" install --quiet \
        -i "$PIP_INDEX_URL" \
        -e "$prefix/src[anthropic]"

    # editable install: the live source tree is what gets patched
    grep -c '^def _is_third_party_anthropic_endpoint' "$prefix/src/agent/anthropic_adapter.py" | grep -qx 1
    sed -i '/^def _is_third_party_anthropic_endpoint/a\    if os.environ.get("HERMES_FORCE_NATIVE_ANTHROPIC"):\n        return False  # gateway declared native: replay signed thinking blocks' \
        "$prefix/src/agent/anthropic_adapter.py"
    # Patch 2: on the native path hermes replays thinking blocks only from the
    # LATEST assistant message and strips the rest. That rewrites the cached
    # prefix on every turn (one full prompt-cache miss measured mid-run) and
    # drops earlier reasoning from the trajectory. HERMES_KEEP_ALL_THINKING=1
    # keeps every signed block; the channel accepts an accumulating history
    # (the other harnesses send 3-4 blocks and get 200).
    python3 - "$prefix/src/agent/anthropic_adapter.py" <<'PYPATCH'
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
    "$prefix/venv/bin/python" - <<'PYCHECK'
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
PYCHECK

    # models.dev catalog cache: a pre-seeded file (touched fresh by the
    # adapter) skips the 15s network fetch on startup entirely. models.dev is
    # not always reachable from build hosts, so a good copy is cached and
    # reused; without either the pod just takes the timeout path.
    if curl -fsSL -m 60 -o "$RUNTIMES/models_dev.json.tmp" https://models.dev/api.json; then
        mv -f "$RUNTIMES/models_dev.json.tmp" "$RUNTIMES/models_dev.json"
    fi
    rm -f "$RUNTIMES/models_dev.json.tmp"
    if [ -s "$RUNTIMES/models_dev.json" ]; then
        cp "$RUNTIMES/models_dev.json" "$prefix/models_dev_cache.json"
    else
        echo "warn: no models.dev catalog available; pods fall back to the 15s timeout path" >&2
    fi

    mkdir -p "$prefix/bin"
    cp "$RUNTIMES/rg" "$prefix/bin/rg"
    wrapper "$prefix/bin/hermes" "/opt/mimo-hermes/venv/bin/hermes"
    "$prefix/bin/hermes" --version | grep -F "$HERMES_VERSION"
    pack "$prefix" "hermes-$HERMES_VERSION-linux-x64.tar.gz"
}

build_kilocode() {
    # Kilo Code is an opencode fork and likewise ships bun-compiled standalone
    # binaries per platform — but three linux-x64 variants. Ship -baseline: it
    # is the only one that runs everywhere (-musl links /lib/ld-musl-x86_64.so.1
    # and dies instantly on a glibc pod; the plain build requires AVX2;
    # -baseline uses the same glibc interpreter as plain, minus that
    # requirement). No source patch is needed: effort reaches the wire as
    # adaptive + output_config.effort natively, unlike opencode/openclaw.
    fetch_npm "@kilocode/cli-linux-x64-baseline" "$KILOCODE_VERSION" kilocode-bin
    local s="$STAGE/kilocode"; rm -rf "$s"; mkdir -p "$s"
    # The whole bin/ tree is required, not just the binary: tree-sitter/*.wasm
    # grammars, the bwrap + kilo-sandbox-seccomp helpers and the
    # kilo-sandbox-*.js workers all live alongside it.
    cp -a "$PKGS/kilocode-bin/package/bin" "$s/bin"
    cp -a "$PKGS/kilocode-bin/package/LICENSE" "$s/" 2>/dev/null || true
    # Embedded web UI: dead weight in a headless rollout (~and the adapter sets
    # KILO_DISABLE_EMBEDDED_WEB_UI anyway).
    rm -rf "$s/bin/console"
    "$s/bin/kilo" --version | grep -F "$KILOCODE_VERSION"
    pack "$s" "kilocode-$KILOCODE_VERSION-linux-x64.tar.gz"
}

build_mini_swe_agent() {
    # Own stripped cpython + venv prune (adapter uses --agent-class default,
    # anthropic transport only). Payload ~92 MB gz.
    fetch_cpython
    fetch_rg
    local prefix=/opt/mimo-mini-swe-agent
    rm -rf "$prefix"; mkdir -p "$prefix"
    tar -xzf "$RUNTIMES/cpython-$CPYTHON_VERSION.tar.gz" -C "$prefix"
    "$prefix/python/bin/python3.12" -m venv "$prefix/venv"
    # ensurepip must go AFTER venv creation.
    rm -rf "$prefix"/python/lib/python3.12/{idlelib,turtledemo,tkinter,ensurepip} \
           "$prefix"/python/lib/{tcl9.0,tk9.0,itcl4.3.5,tcl9,thread3.0.4,pkgconfig} \
           "$prefix"/python/lib/lib{tcl9,tk9,itcl}* \
           "$prefix"/python/include \
           "$prefix"/python/share
    PIP_CONFIG_FILE=/dev/null "$prefix/venv/bin/pip" install --quiet \
        -i "$PIP_INDEX_URL" \
        "mini-swe-agent==$MSWEA_VERSION"

    local sp
    sp=$("$prefix/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # Prune deps the adapter never loads. Smoke test below catches false negatives.
    (
        cd "$sp"
        for pat in \
            datasets datasets-*.dist-info \
            pyarrow pyarrow.libs pyarrow-*.dist-info \
            pandas pandas-*.dist-info \
            textual textual-*.dist-info \
            PIL Pillow-*.dist-info pillow-*.dist-info \
            multiprocess multiprocess-*.dist-info \
            dill dill-*.dist-info \
            fsspec fsspec-*.dist-info \
            xxhash xxhash-*.dist-info \
            botocore botocore-*.dist-info \
            boto3 boto3-*.dist-info \
            s3transfer s3transfer-*.dist-info \
            jmespath jmespath-*.dist-info \
            hf_xet hf_xet-*.dist-info \
            pip pip-*.dist-info; do
            rm -rf $pat
        done
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    )
    # Upstream invariants the adapter depends on:
    # 1. cache_control breakpoint auto-enabled for claude* names
    grep -c 'config\["set_cache_control"\] = "default_end"' \
        "$sp/minisweagent/models/__init__.py" | grep -qx 1
    # 2. submit sentinel unchanged
    grep -c 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' \
        "$sp/minisweagent/environments/local.py" | grep -qx 1
    # 3. limit==0 means unlimited
    grep -c '0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost' \
        "$sp/minisweagent/agents/default.py" | grep -qx 1
    # 4. model_kwargs splatted into litellm.completion
    grep -c '\*\*(self.config.model_kwargs | kwargs)' \
        "$sp/minisweagent/models/litellm_model.py" | grep -qx 1
    # 5. effort xhigh/max reaches wire as adaptive+output_config; one trailing cache breakpoint.
    LITELLM_LOCAL_MODEL_COST_MAP=True MSWEA_SILENT_STARTUP=1 MSWEA_CONFIGURED=1 \
    MSWEA_GLOBAL_CONFIG_DIR="$prefix/.cfgcheck" \
    "$prefix/venv/bin/python" - <<'PYCHECK'
import json
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from minisweagent.models.utils.cache_control import set_cache_control

cfg, m = AnthropicConfig(), "claude-opus-5"
assert AnthropicConfig._is_adaptive_thinking_model(m, "anthropic"), \
    "opus-5 is no longer an adaptive-thinking model in this litellm"
for eff in ("high", "xhigh", "max"):
    op = cfg.map_openai_params(
        non_default_params={"reasoning_effort": eff, "max_tokens": 32000},
        optional_params={}, model=m, drop_params=False)
    d = cfg.transform_request(model=m, messages=[{"role": "user", "content": "x"}],
                              optional_params=dict(op), litellm_params={}, headers={})
    assert d.get("thinking") == {"type": "adaptive"}, (eff, d.get("thinking"))
    assert d.get("output_config") == {"effort": eff}, \
        f"effort {eff} clamped to {d.get('output_config')}"
    assert d.get("max_tokens") == 32000, d.get("max_tokens")

msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
out = set_cache_control(msgs, mode="default_end")
n = json.dumps(out).count('"cache_control"')
assert n == 1, f"expected 1 cache_control breakpoint, got {n}"
assert "cache_control" in json.dumps(out[-1]), "breakpoint is not on the last message"
print("PYCHECK OK")
PYCHECK
    rm -rf "$prefix/.cfgcheck"

    mkdir -p "$prefix/bin"
    cp "$RUNTIMES/rg" "$prefix/bin/rg"
    wrapper "$prefix/bin/mini" "/opt/mimo-mini-swe-agent/venv/bin/mini"
    # `mini` has no --version; probe the package attribute.
    MSWEA_GLOBAL_CONFIG_DIR="$prefix/.vercheck" \
        "$prefix/venv/bin/python" -c "import minisweagent; print(minisweagent.__version__)" \
        | grep -Fx "$MSWEA_VERSION"
    MSWEA_GLOBAL_CONFIG_DIR="$prefix/.vercheck" \
        "$prefix/bin/mini" --help > /dev/null < /dev/null
    # Reach the LLM call via fake api; catches prune-induced lazy import breakage.
    # Spool then grep because pipefail propagates mini's non-zero exit.
    local smoke_log="$prefix/.smoke.log"
    MSWEA_CONFIGURED=1 MSWEA_SILENT_STARTUP=1 MSWEA_COST_TRACKING=ignore_errors \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    MSWEA_GLOBAL_CONFIG_DIR="$prefix/.smokecheck" \
    MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 \
    ANTHROPIC_API_BASE=http://127.0.0.1:9 \
    ANTHROPIC_API_KEY=dummy \
    timeout 30 "$prefix/venv/bin/mini" --agent-class default -y --exit-immediately \
        -c "$sp/minisweagent/config/mini.yaml" \
        -c model.model_name=anthropic/claude-opus-5 \
        -c model.model_kwargs.reasoning_effort=xhigh \
        -c model.model_kwargs.max_tokens=4096 \
        -c agent.step_limit=1 \
        -c agent.cost_limit=0 \
        -c environment.environment_class=local \
        -t "smoke" > "$smoke_log" 2>&1 || true
    if ! grep -qE "Connection refused|InternalServerError" "$smoke_log"; then
        echo "SLIM SMOKE FAILED: expected connection-refused, got:" >&2
        tail -20 "$smoke_log" >&2
        exit 1
    fi
    rm -rf "$prefix/.vercheck" "$prefix/.smokecheck" "$smoke_log"
    pack "$prefix" "mini-swe-agent-$MSWEA_VERSION-slim-linux-x64.tar.gz"
}

build_dsh() {
    # No published binary and the bundle is a 60-package plugin tree, so install
    # prod deps locally and ship the whole thing with a pinned node 22 (the
    # @vscode/ripgrep platform package already carries its own rg).
    #
    # Two build-time patches:
    #   1. the launcher re-creates the HMR plugin and registers chokidar
    #      watchers on the user patch files; the watch error escapes the
    #      launcher's try/catch and kills the run when the host's inotify
    #      budget is exhausted. A one-shot pod turn never reloads its
    #      composition, so the whole block goes.
    #   2. the headless runner always mints a fresh session id, so there is no
    #      way to continue a session; honour $DSH_SESSION_ID and resume it when
    #      $DSH_RESUME_SESSION=1 (agents.resume loads the persisted session).
    fetch_node "$NODE22"
    local app="$PKGS/dsh"
    if [ ! -d "$app/node_modules/@deepseek-ai/dsh" ]; then
        mkdir -p "$app"
        (export PATH="$RUNTIMES/node-v$NODE22-linux-x64/bin:$PATH" \
                npm_config_userconfig=/dev/null
         cd "$app" && npm install --omit=dev --no-audit --no-fund \
             --loglevel=error --registry "$NPM_REG" "@deepseek-ai/dsh@$DSH_VERSION")
    fi
    local s="$STAGE/dsh"; rm -rf "$s"; mkdir -p "$s/app"
    cp -r "$app/." "$s/app/"
    python3 - "$s/app/node_modules/@deepseek-ai" <<'PY'
import glob, sys, pathlib

root = pathlib.Path(sys.argv[1])

# patch 1: drop the launcher's HMR + user-patch watchers
boots = [
    p for p in glob.glob(str(root / "dsh/lib/profile-boot-*.js"))
    if 'ctx.get("loader") !== void 0) try {' in open(p, encoding="utf8").read()
]
assert len(boots) == 1, f"dsh: expected one profile-boot chunk with the watcher block, got {boots}"
text = open(boots[0], encoding="utf8").read()
anchor = ('if (!signalShutdown.signal.aborted && ctx.fiber.state === 2 '
          '&& ctx.get("loader") !== void 0) try {')
assert text.count(anchor) == 1, "dsh: watcher block anchor moved"
open(boots[0], "w", encoding="utf8").write(text.replace(anchor, "if (false) try {", 1))

# patch 2: resumable headless sessions
runner = root / "dsh-headless/lib/index.js"
text = runner.read_text(encoding="utf8")
old = """	const { agent } = await agents.create({
		sessionId: SessionId(`session-${randomUUID()}`),
		meta: { cwd: process.cwd() },
		agentOptions: {
			provider: selection.provider,
			model: selection.model
		},
		setup: (agentCtx) => {
			installModelSelection(agentCtx, {
				current: selection,
				assembled: void 0
			});
		}
	});"""
assert text.count(old) == 1, "dsh: headless agent-creation anchor moved"
new = """	const dshEnvSessionId = process.env.DSH_SESSION_ID;
	const dshSessionId = SessionId(dshEnvSessionId && dshEnvSessionId.length > 0 ? dshEnvSessionId : `session-${randomUUID()}`);
	const dshAgentOptions = {
		provider: selection.provider,
		model: selection.model
	};
	const dshSetup = (agentCtx) => {
		installModelSelection(agentCtx, {
			current: selection,
			assembled: void 0
		});
	};
	const { agent } = process.env.DSH_RESUME_SESSION === "1" ? await agents.resume({
		resumeSessionId: dshSessionId,
		agentOptions: dshAgentOptions,
		setup: dshSetup
	}) : await agents.create({
		sessionId: dshSessionId,
		meta: { cwd: process.cwd() },
		agentOptions: dshAgentOptions,
		setup: dshSetup
	});"""
runner.write_text(text.replace(old, new, 1), encoding="utf8")
print("dsh: launcher watchers removed, headless resume enabled")
PY
    # the search tool execs the rg that ships in @vscode/ripgrep-linux-x64
    "$s/app/node_modules/@vscode/ripgrep-linux-x64/bin/rg" --version >/dev/null
    cp -r "$RUNTIMES/node-v$NODE22-linux-x64" "$s/node"
    wrapper "$s/bin/dsh" \
        "/opt/mimo-dsh/node/bin/node /opt/mimo-dsh/app/node_modules/@deepseek-ai/dsh/lib/bin.js"
    "$s/node/bin/node" "$s/app/node_modules/@deepseek-ai/dsh/lib/bin.js" --version \
        | grep -F "$DSH_VERSION"
    pack "$s" "dsh-$DSH_VERSION-linux-x64.tar.gz"
}

ALL=(grok opencode kimi-code openclaw omp kimi-cli hermes kilocode mini-swe-agent dsh)
targets=("${@:-${ALL[@]}}")
for t in "${targets[@]}"; do
    echo "=== building $t"
    "build_${t//-/_}"
done
echo PAYLOADS_DONE
