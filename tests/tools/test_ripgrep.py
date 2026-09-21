"""Host-side ripgrep resolution for the Grep tools (no network: the download is faked)."""

import os
import tarfile

import pytest

from mimoagent.tools import ripgrep


def _fake_rg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho 'ripgrep 15.1.0'\n")
    path.chmod(0o755)
    return path


def test_explicit_path_wins(monkeypatch, tmp_path):
    rg = _fake_rg(tmp_path / "custom" / "rg")
    monkeypatch.setenv(ripgrep.RG_PATH_ENV, str(rg))
    assert ripgrep.resolve_host_rg(download=False) == rg


def test_explicit_path_must_be_runnable(monkeypatch, tmp_path):
    monkeypatch.setenv(ripgrep.RG_PATH_ENV, str(tmp_path / "missing"))
    with pytest.raises(ripgrep.RipgrepUnavailable, match=ripgrep.RG_PATH_ENV):
        ripgrep.resolve_host_rg(download=False)


def test_cache_hit_skips_the_download(monkeypatch, tmp_path):
    monkeypatch.delenv(ripgrep.RG_PATH_ENV, raising=False)
    monkeypatch.setenv("MIMOAGENT_CACHE_DIR", str(tmp_path))
    cached = _fake_rg(tmp_path / "ripgrep" / ripgrep.RG_VERSION / "rg")
    monkeypatch.setattr(ripgrep, "_download", lambda url, dest: pytest.fail("download must not run"))
    assert ripgrep.resolve_host_rg() == cached


def test_download_extracts_rg_into_the_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(ripgrep.RG_PATH_ENV, raising=False)
    monkeypatch.setenv("MIMOAGENT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(ripgrep.RG_URL_ENV, "https://mirror.invalid/rg.tar.gz")
    # a release-shaped tarball: <dir>/rg
    src = _fake_rg(tmp_path / "ripgrep-x/rg")
    archive = tmp_path / "rg.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname="ripgrep-x/rg")
    seen = {}

    def fake_urlopen(url, timeout=0):
        seen["url"] = url
        return archive.open("rb")

    monkeypatch.setattr(ripgrep.urllib.request, "urlopen", fake_urlopen)
    resolved = ripgrep.resolve_host_rg()
    assert seen["url"] == "https://mirror.invalid/rg.tar.gz"
    assert resolved == tmp_path / "cache" / "ripgrep" / ripgrep.RG_VERSION / "rg"
    assert os.access(resolved, os.X_OK)


def test_no_cache_and_no_download_fails_with_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv(ripgrep.RG_PATH_ENV, raising=False)
    monkeypatch.setenv("MIMOAGENT_CACHE_DIR", str(tmp_path))
    with pytest.raises(ripgrep.RipgrepUnavailable, match="no cached rg"):
        ripgrep.resolve_host_rg(download=False)
