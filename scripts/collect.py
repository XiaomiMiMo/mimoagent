#!/usr/bin/env python3
"""
Collect trajectories from output folders with a given prefix.
Skips trajectories with errors.
Outputs statistics on reward 1/0 and saves valid trajectories to a jsonl file.

Usage:
    python scripts/collect.py <prefix> [--output <output_file>]
    python scripts/collect.py outputs/0107-itp-lyc  # also works with path format
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def load_single_traj(traj_json_path: Path):
    """Load and process a single trajectory file."""
    try:
        with open(traj_json_path) as f:
            traj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"error": str(e), "path": str(traj_json_path)}

    # Skip trajectories with errors
    exit_status = traj.get("info", {}).get("exit_status", "")
    if "error" in exit_status.lower():
        return {"skipped": True, "reason": "error_status"}

    instance_id = traj_json_path.parent.name
    payload = traj.get("traj", {})
    payload["reward"] = traj.get("info", {}).get("reward", 0)
    payload["instance_id"] = instance_id

    return {"payload": payload}


def process_traj(traj):
    """Validate and clean a single trajectory."""
    if len(traj.get("messages", [])) == 0:
        return None

    try:
        for message in traj["messages"]:
            if message["role"] == "assistant":
                for tool_call in message.get("tool_calls", []):
                    if tool_call["type"] == "function":
                        if isinstance(tool_call["function"]["arguments"], str):
                            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
                        tool_call["function"]["arguments"] = {
                            k: v for k, v in tool_call["function"]["arguments"].items() if v != ""
                        }
                if message.get("content"):
                    message["content"] = message["content"].strip()
                if message.get("reasoning_content"):
                    message["reasoning_content"] = message["reasoning_content"].strip()
                    if not message["reasoning_content"]:
                        return None
        return traj
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories from output folders")
    parser.add_argument("prefix", type=str, help="Prefix to match output folders (can include path like outputs/xxx)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output jsonl file path")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Base outputs directory")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    prefix_path = Path(args.prefix)

    # Support both "outputs/xxx" and "xxx" formats
    if prefix_path.parent.name and prefix_path.parent.exists():
        # User provided path like "outputs/0107-itp-lyc"
        outputs_dir = prefix_path.parent
        prefix = prefix_path.name
    else:
        outputs_dir = Path(args.outputs_dir)
        prefix = args.prefix

    # Find all matching output folders
    matching_folders = [f for f in outputs_dir.iterdir() if f.is_dir() and f.name.startswith(prefix)]
    print(f"Found {len(matching_folders)} folders matching prefix '{prefix}' in {outputs_dir}")

    # Collect all traj.json paths first
    traj_paths = []
    for folder in matching_folders:
        for subfolder in folder.iterdir():
            if not subfolder.is_dir():
                continue
            instance_id = subfolder.name
            traj_json = subfolder / f"{instance_id}.traj.json"
            if traj_json.exists():
                traj_paths.append(traj_json)

    print(f"Found {len(traj_paths)} trajectory files to process")

    # Load trajectories in parallel with progress bar
    trajs = []
    error_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_single_traj, p): p for p in traj_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading trajectories"):
            result = future.result()
            if "error" in result:
                error_count += 1
            elif "skipped" in result:
                skipped_count += 1
            elif "payload" in result:
                trajs.append(result["payload"])

    print(f"\nTotal trajectories collected: {len(trajs)}")
    print(f"Trajectories with errors (skipped): {skipped_count}")
    print(f"Load errors: {error_count}")

    # Process and validate trajectories in parallel
    valid = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_traj, t): t for t in trajs}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            result = future.result()
            if result is not None:
                valid.append(result)

    print(f"Valid trajectories after processing: {len(valid)}")

    # Count reward statistics
    reward_1_count = sum(1 for t in valid if t.get("reward") == 1)
    reward_0_count = sum(1 for t in valid if t.get("reward") == 0)
    other_reward_count = len(valid) - reward_1_count - reward_0_count

    print("\nReward statistics:")
    print(f"  Reward = 1: {reward_1_count}")
    print(f"  Reward = 0: {reward_0_count}")
    if other_reward_count > 0:
        print(f"  Other rewards: {other_reward_count}")

    # Save to jsonl
    output_file = args.output or f"{prefix}_collected.jsonl"
    with open(output_file, "w") as f:
        for traj in valid:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(valid)} trajectories to {output_file}")


if __name__ == "__main__":
    main()
