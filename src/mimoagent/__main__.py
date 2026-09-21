#!/usr/bin/env python3
"""Entrypoint for `python -m mimoagent` — runs the batch runner.

Single-instance runs: pass `--filter '<instance_id>'`.
"""

from mimoagent.run.extra.batch import app

if __name__ == "__main__":
    app()
