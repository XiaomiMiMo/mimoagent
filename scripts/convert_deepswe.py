#!/usr/bin/env python3
"""Convert a checkout of the public DeepSWE task repository into a mimoagent JSONL.

DeepSWE (https://github.com/datacurve-ai/deep-swe) publishes every task in
Harbor layout::

    tasks/<task_id>/task.toml               # image, base commit, verifier timeout
    tasks/<task_id>/instruction.md          # the prompt the agent sees
    tasks/<task_id>/tests/test.sh           # verifier entry point
    tasks/<task_id>/tests/test.patch        # held-out tests
    tasks/<task_id>/tests/grader.py         # shared grader (v1.1)
    tasks/<task_id>/tests/config.json       # f2p / p2p whitelists (v1.1)
    tasks/<task_id>/solution/solution.patch # reference solution (for --recalc-input gt)

This script flattens each task into one row of the ``dataset_type: deepswe``
schema consumed by ``DeepSWEEnvironment``.

Usage:
  git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git
  python scripts/convert_deepswe.py --src deep-swe/tasks --out deepswe.jsonl
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path


def _read(path: Path, required: bool = True) -> str:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return ""
    return path.read_text(encoding="utf-8")


def build_instance(task_dir: Path) -> dict:
    meta = tomllib.loads(_read(task_dir / "task.toml"))
    metadata = meta.get("metadata", {})
    environment = meta.get("environment", {})
    verifier = meta.get("verifier", {})

    tests = task_dir / "tests"
    grader_py = _read(tests / "grader.py", required=False)
    config_json = _read(tests / "config.json", required=False)
    is_v11 = bool(grader_py and config_json)

    instance = {
        "instance_id": metadata.get("task_id") or task_dir.name,
        "ext_id": metadata.get("ext_id", ""),
        "dataset_type": "deepswe",
        "schema_version": "1.1" if is_v11 else "1.0",
        "docker_image": environment["docker_image"],
        "base_commit": metadata["base_commit_hash"],
        "problem_statement": _read(task_dir / "instruction.md"),
        "test_sh": _read(tests / "test.sh"),
        "test_patch": _read(tests / "test.patch"),
        "grader_py": grader_py,
        "config_json": config_json,
        "patch": _read(task_dir / "solution" / "solution.patch", required=False),
        "verifier_timeout_sec": verifier.get("timeout_sec"),
        "language": metadata.get("language", ""),
        "repository_url": metadata.get("repository_url", ""),
        "category": metadata.get("category", ""),
    }
    if not instance["test_sh"].strip():
        raise ValueError(f"{task_dir.name}: tests/test.sh is empty")
    return instance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="the deep-swe/tasks directory")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL path")
    args = parser.parse_args()

    task_dirs = sorted(p for p in args.src.iterdir() if (p / "task.toml").is_file())
    if not task_dirs:
        print(f"no task.toml found under {args.src}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for task_dir in task_dirs:
            f.write(json.dumps(build_instance(task_dir), ensure_ascii=False) + "\n")
    print(f"wrote {len(task_dirs)} instances to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
