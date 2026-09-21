# Harness payload builder

The blackbox adapters (`agent.type: claude-code`, `codex`, `opencode`, ...) install
their scaffold inside the task container from the upstream public source (npm,
GitHub releases, PyPI) at a pinned version. That needs outbound network access
from the container at setup time.

For containers without internet access, or to avoid re-downloading runtimes for
every rollout, build a self-contained payload tarball once and hand it to the
adapter:

```bash
# build every harness (or pass names: grok opencode kimi-cli ...)
bash scripts/harness_payloads/build_harness_payloads.sh
ls dist/harness_payloads/
```

Then, in the agent block of your config:

```yaml
agent:
  type: opencode
  version: 1.18.18
  payload_path: dist/harness_payloads/opencode-1.18.18-linux-x64.tar.gz   # uploaded into the container
  # or: payload_url: https://mirror.example.com/harness/opencode-1.18.18-linux-x64.tar.gz
```

The installer unpacks the payload into `/opt/mimo-<name>` and skips the public
download path. The payload must match the pinned `version`: the installer
verifies the harness reports it.

Requirements on the build host: `curl`, `tar`, `node` and `python3` on PATH,
`bun` for `omp`. The script builds glibc (`linux-x64`) payloads; run it on an
Alpine host to produce payloads for musl-based images.

Mirrors: every download honours `NPM_REG`, `NODE_DIST`, `PBS_BASE`,
`GITHUB_BASE` and `PIP_INDEX_URL`. The same overrides (with `NPM_REGISTRY`
instead of `NPM_REG`) apply to the in-container installers through the
adapter's `install_env` mapping.
