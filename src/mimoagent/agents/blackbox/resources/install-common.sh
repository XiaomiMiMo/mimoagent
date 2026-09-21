#!/bin/bash
# Shared helpers for the blackbox harness installers (install-<name>.sh).
#
# Every installer sources this file first:
#
#     . "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"
#
# The adapters stage it into the pod as /tmp/mimo-install-common.sh and pass
# MIMO_INSTALL_COMMON so the installer finds it; running an installer from the
# repository checkout works too because the file sits next to it.
#
# Design rules:
#   * Sourcing has no side effects beyond defining functions and reading env.
#   * Every download goes through mimo_fetch, which needs curl or wget and can
#     bootstrap curl from the image's package manager (apk/apt/dnf/yum).
#   * Public upstream sources are the default; every base URL is overridable
#     so users behind a mirror can point at their own copies:
#       NPM_REGISTRY  (default https://registry.npmjs.org)
#       NODE_DIST     (default https://nodejs.org/dist)
#       NODE_MUSL_DIST(default https://unofficial-builds.nodejs.org/download/release)
#       PBS_BASE      (default https://github.com/astral-sh/python-build-standalone/releases/download)
#       GITHUB_BASE   (default https://github.com)
#       PIP_INDEX_URL (pip's own knob; unset = PyPI)
#   * A prebuilt payload tarball (see scripts/harness_payloads/) bypasses the
#     public path entirely: <PREFIX>_PAYLOAD (a path already in the pod) or
#     <PREFIX>_PAYLOAD_URL (fetched). payload_override handles both.
#
# Linux x86_64 only, glibc and musl.

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

# Sets MIMO_LIBC (glibc|musl) and MIMO_PLATFORM (linux-x64|linux-x64-musl).
mimo_detect_libc() {
    local arch
    arch="$(uname -m 2>/dev/null || echo unknown)"
    case "$arch" in
        x86_64|amd64) ;;
        *) echo "unsupported architecture: $arch (the harness installers support linux x86_64 only)" >&2; return 1 ;;
    esac
    if [ -f /lib/libc.musl-x86_64.so.1 ] || ldd /bin/ls 2>&1 | grep -q musl; then
        MIMO_LIBC="musl"
        MIMO_PLATFORM="linux-x64-musl"
    else
        MIMO_LIBC="glibc"
        MIMO_PLATFORM="linux-x64"
    fi
    export MIMO_LIBC MIMO_PLATFORM
}

# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

# Old Debian releases moved to archive.debian.org; repoint apt when needed.
_mimo_apt_repoint_archive() {
    if apt-get update 2>&1 | grep -q "does not have a Release file"; then
        local codename
        codename=$(. /etc/os-release 2>/dev/null && echo "$VERSION_CODENAME" || echo "")
        if [ -n "$codename" ]; then
            cat > /etc/apt/sources.list <<EOF
deb http://archive.debian.org/debian ${codename} main
deb http://archive.debian.org/debian-security ${codename}/updates main
EOF
            apt-get update || true
        fi
    fi
}

# Install a package with whatever package manager the image has.
mimo_pkg_install() {
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    elif command -v apt-get >/dev/null 2>&1; then
        _mimo_apt_repoint_archive
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y "$@"
    elif command -v microdnf >/dev/null 2>&1; then
        microdnf install -y "$@"
    elif command -v yum >/dev/null 2>&1; then
        yum install -y "$@"
    else
        echo "no supported package manager found to install: $*" >&2
        return 1
    fi
}

# Make sure curl or wget exists (bootstrapping curl if neither does).
ensure_downloader() {
    if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
        return 0
    fi
    echo "neither curl nor wget found; installing curl..."
    mimo_pkg_install curl ca-certificates || {
        echo "cannot install curl; provide a payload via <PREFIX>_PAYLOAD instead" >&2
        return 1
    }
}

# Make sure tar can unpack .tar.gz (and .tar.xz when xz is requested).
ensure_tar_gz() {
    local need=()
    command -v tar >/dev/null 2>&1 || need+=(tar)
    command -v gzip >/dev/null 2>&1 || need+=(gzip)
    if [ "${1:-}" = "xz" ]; then
        command -v xz >/dev/null 2>&1 || need+=(xz)
    fi
    if [ "${#need[@]}" -gt 0 ]; then
        mimo_pkg_install "${need[@]}" || { echo "cannot install ${need[*]}" >&2; return 1; }
    fi
}

# mimo_fetch <url> <dest>: download with retries; curl first, wget as fallback.
mimo_fetch() {
    local url="$1" dest="$2"
    local max_retries="${MIMO_FETCH_RETRIES:-3}" delay=5 i
    ensure_downloader || return 1
    for i in $(seq 1 "$max_retries"); do
        if command -v curl >/dev/null 2>&1; then
            if curl -fSL --retry 2 --max-time "${MIMO_FETCH_MAX_TIME:-1800}" -o "$dest" "$url"; then
                return 0
            fi
        elif wget -q -O "$dest" "$url"; then
            return 0
        fi
        rm -f "$dest"
        echo "download failed (attempt $i/$max_retries): $url; retrying in ${delay}s..." >&2
        sleep "$delay"
        delay=$((delay * 2))
    done
    echo "download failed after $max_retries attempts: $url" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Upstream sources
# ---------------------------------------------------------------------------

# fetch_npm_tarball <package> <version> <dest_dir>: unpack the npm tarball's
# package/ directory into <dest_dir>. The tarball URL is deterministic, so no
# JSON parsing (and therefore no node/python) is needed.
fetch_npm_tarball() {
    local pkg="$1" ver="$2" dest="$3"
    local base="${pkg##*/}"  # @scope/name -> name
    local url="${NPM_REGISTRY:-https://registry.npmjs.org}/${pkg}/-/${base}-${ver}.tgz"
    local tmp
    tmp="$(mktemp)"
    ensure_tar_gz || return 1
    mimo_fetch "$url" "$tmp" || { rm -f "$tmp"; return 1; }
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tmp" -C "$dest" --strip-components=1
    rm -f "$tmp"
}

# fetch_node <version> <dest_dir>: unpack a Node.js runtime matching the pod's
# libc so that <dest_dir>/bin/node exists.
fetch_node() {
    local ver="$1" dest="$2" url tmp
    [ -n "${MIMO_LIBC:-}" ] || mimo_detect_libc || return 1
    if [ "$MIMO_LIBC" = "musl" ]; then
        url="${NODE_MUSL_DIST:-https://unofficial-builds.nodejs.org/download/release}/v${ver}/node-v${ver}-linux-x64-musl.tar.gz"
    else
        url="${NODE_DIST:-https://nodejs.org/dist}/v${ver}/node-v${ver}-linux-x64.tar.gz"
    fi
    tmp="$(mktemp)"
    ensure_tar_gz || return 1
    mimo_fetch "$url" "$tmp" || { rm -f "$tmp"; return 1; }
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tmp" -C "$dest" --strip-components=1
    rm -f "$tmp"
    "$dest/bin/node" --version >/dev/null
}

# fetch_cpython <version> <pbs_tag> <dest_dir>: unpack a standalone CPython
# (astral-sh/python-build-standalone, install_only_stripped flavour) so that
# <dest_dir>/bin/python3 exists.
fetch_cpython() {
    local ver="$1" tag="$2" dest="$3" triple url tmp
    [ -n "${MIMO_LIBC:-}" ] || mimo_detect_libc || return 1
    if [ "$MIMO_LIBC" = "musl" ]; then
        triple="x86_64-unknown-linux-musl"
    else
        triple="x86_64-unknown-linux-gnu"
    fi
    url="${PBS_BASE:-https://github.com/astral-sh/python-build-standalone/releases/download}/${tag}/cpython-${ver}+${tag}-${triple}-install_only_stripped.tar.gz"
    tmp="$(mktemp)"
    ensure_tar_gz || return 1
    mimo_fetch "$url" "$tmp" || { rm -f "$tmp"; return 1; }
    rm -rf "$dest"
    mkdir -p "$dest"
    # The archive has a single top-level "python/" directory.
    tar -xzf "$tmp" -C "$dest" --strip-components=1
    rm -f "$tmp"
    "$dest/bin/python3" --version >/dev/null
}

# fetch_github_release <owner/repo> <tag> <asset> <dest_file>
fetch_github_release() {
    local repo="$1" tag="$2" asset="$3" dest="$4"
    mimo_fetch "${GITHUB_BASE:-https://github.com}/${repo}/releases/download/${tag}/${asset}" "$dest"
}

# fetch_rg <dest_bin>: a fully static ripgrep (musl build runs on glibc too).
MIMO_RG_VERSION="${MIMO_RG_VERSION:-15.1.0}"
fetch_rg() {
    local dest="$1" tmp dir
    tmp="$(mktemp)"
    ensure_tar_gz || return 1
    fetch_github_release "BurntSushi/ripgrep" "$MIMO_RG_VERSION" \
        "ripgrep-${MIMO_RG_VERSION}-x86_64-unknown-linux-musl.tar.gz" "$tmp" || { rm -f "$tmp"; return 1; }
    dir="$(mktemp -d)"
    tar -xzf "$tmp" -C "$dir" --strip-components=1
    mkdir -p "$(dirname "$dest")"
    mv -f "$dir/rg" "$dest"
    chmod 0755 "$dest"
    rm -rf "$tmp" "$dir"
    "$dest" --version >/dev/null
}

# mimo_pip_install <python> <pip args...>: pip with the image's pip.conf
# ignored (it may point at an unreachable index) and PIP_INDEX_URL honoured.
mimo_pip_install() {
    local python="$1"; shift
    PIP_CONFIG_FILE=/dev/null "$python" -m pip install --quiet --no-cache-dir \
        ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} "$@"
}

# ---------------------------------------------------------------------------
# Payload override, markers, wrappers
# ---------------------------------------------------------------------------

# payload_override <PREFIX> <dest_dir>: when <PREFIX>_PAYLOAD (a tarball already
# in the pod) or <PREFIX>_PAYLOAD_URL is set, unpack it into <dest_dir> and
# return 0. Returns 1 when neither is set so the caller takes the public path.
payload_override() {
    local prefix="$1" dest="$2"
    local path_var="${prefix}_PAYLOAD" url_var="${prefix}_PAYLOAD_URL"
    local payload="${!path_var:-}" url="${!url_var:-}"
    if [ -z "$payload" ] && [ -z "$url" ]; then
        return 1
    fi
    ensure_tar_gz || return 1
    if [ -z "$payload" ]; then
        payload="$(mktemp)"
        echo "downloading payload from $url"
        mimo_fetch "$url" "$payload" || { rm -f "$payload"; echo "payload download failed" >&2; exit 1; }
    fi
    [ -f "$payload" ] || { echo "payload missing: $payload" >&2; exit 1; }
    # Replace the tree wholesale: a same-version repack must not leave stale
    # files behind, and a failed unpack must not look installed.
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$payload" -C "$dest"
    rm -f "$payload"
    return 0
}

# mimo_marker_matches <marker_file> <version>
mimo_marker_matches() {
    [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

# mimo_marker_write <marker_file> <version>
mimo_marker_write() {
    mkdir -p "$(dirname "$1")"
    printf '%s' "$2" > "$1"
}

# mimo_wrapper <path> <command...>: a tiny exec wrapper script.
mimo_wrapper() {
    local path="$1"; shift
    mkdir -p "$(dirname "$path")"
    printf '#!/bin/sh\nexec %s "$@"\n' "$*" > "$path"
    chmod 0755 "$path"
}

# mimo_verify_version <want> <command...>: run a probe and require the wanted
# version string in its output; print the probe tail on failure.
mimo_verify_version() {
    local want="$1"; shift
    local out
    out="$("$@" 2>&1 || true)"
    if ! printf '%s' "$out" | grep -qF "$want"; then
        echo "version check failed (want $want):" >&2
        printf '%s\n' "$out" | tail -5 >&2
        return 1
    fi
}
