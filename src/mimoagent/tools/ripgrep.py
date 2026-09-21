"""Locate (or download) the static ripgrep binary the Grep tools stage into
environments that do not ship ``rg``.

Resolution order:

1. ``MIMOAGENT_RG_PATH`` — an explicit host path to an ``rg`` binary.
2. The user cache (``platformdirs.user_cache_dir("mimoagent")/ripgrep/<version>/rg``).
3. Download ``MIMOAGENT_RG_URL`` (default: the ``x86_64-unknown-linux-musl``
   tarball of the pinned ripgrep release on GitHub; a fully static build that
   runs on glibc and musl images alike) into that cache.

The download is guarded by a file lock so parallel workers do not race, and
the binary is verified with ``rg --version`` before it is handed out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir

RG_VERSION = "15.1.0"
RG_ASSET = f"ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz"
DEFAULT_RG_URL = f"https://github.com/BurntSushi/ripgrep/releases/download/{RG_VERSION}/{RG_ASSET}"
RG_PATH_ENV = "MIMOAGENT_RG_PATH"
RG_URL_ENV = "MIMOAGENT_RG_URL"


class RipgrepUnavailable(RuntimeError):
    """No usable ``rg`` binary could be resolved for staging."""


def _cache_dir() -> Path:
    return Path(os.getenv("MIMOAGENT_CACHE_DIR") or user_cache_dir("mimoagent")) / "ripgrep" / RG_VERSION


def _is_runnable(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        return subprocess.run([str(path), "--version"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _download(url: str, dest: Path) -> None:
    """Fetch the release tarball and extract ``rg`` to ``dest`` (atomic rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "rg.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        extracted = Path(tmp) / "extracted"
        with tarfile.open(archive) as tar:
            tar.extractall(extracted, filter="data")
        candidates = list(extracted.rglob("rg"))
        if not candidates:
            raise RipgrepUnavailable(f"{url} does not contain an rg binary")
        staged = Path(tmp) / "rg"
        shutil.copy2(candidates[0], staged)
        staged.chmod(0o755)
        os.replace(staged, dest)


def resolve_host_rg(*, download: bool = True) -> Path:
    """Return a runnable ``rg`` binary on the host, downloading it if needed.

    Raises :class:`RipgrepUnavailable` with instructions when no binary can be
    obtained (offline hosts: set ``MIMOAGENT_RG_PATH``).
    """
    explicit = os.getenv(RG_PATH_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if _is_runnable(path):
            return path
        raise RipgrepUnavailable(f"{RG_PATH_ENV}={explicit} is not a runnable rg binary")

    cached = _cache_dir() / "rg"
    if _is_runnable(cached):
        return cached
    if not download:
        raise RipgrepUnavailable(f"no cached rg at {cached}")

    url = os.getenv(RG_URL_ENV) or DEFAULT_RG_URL
    lock_path = cached.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        _lock(lock)
        try:
            if not _is_runnable(cached):
                try:
                    _download(url, cached)
                except (OSError, tarfile.TarError) as e:
                    raise RipgrepUnavailable(
                        f"could not download ripgrep from {url}: {e}. Set {RG_PATH_ENV} to a local static rg "
                        f"binary, or {RG_URL_ENV} to a mirror of {RG_ASSET}."
                    ) from e
        finally:
            _unlock(lock)
    if not _is_runnable(cached):
        raise RipgrepUnavailable(f"downloaded rg at {cached} is not runnable on this host")
    return cached


def _lock(handle) -> None:
    try:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX)
    except ImportError:  # non-POSIX: best effort, no lock
        pass


def _unlock(handle) -> None:
    try:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)
    except ImportError:
        pass
