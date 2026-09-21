import json
from pathlib import Path

import typer
from tqdm import tqdm


def main(
    output_dir: Path = typer.Argument(..., help="Directory containing trajectory outputs"),
    output_file: Path = typer.Argument(..., help="Output JSONL file path"),
) -> None:
    trajs = []
    for subfolder in tqdm(output_dir.iterdir(), desc="Collecting trajectories"):
        if not subfolder.is_dir():
            continue
        instance_id = subfolder.name
        traj_json = subfolder / f"{instance_id}.traj.json"
        if not traj_json.exists():
            continue
        traj = json.loads(traj_json.read_text())
        trajs.append(traj["traj"])

    typer.echo(f"Total trajectories: {len(trajs)}")

    valid = []
    for traj in tqdm(trajs, desc="Validating trajectories"):
        if len(traj.get("messages", [])) == 0:
            continue
        try:
            for message in traj["messages"]:
                if message["role"] == "assistant":
                    for tool_call in message.get("tool_calls", []):
                        if tool_call["type"] == "function":
                            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
            valid.append(traj)
        except:
            continue

    typer.echo(f"Valid trajectories: {len(valid)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for traj in tqdm(valid, desc="Saving trajectories"):
            f.write(json.dumps(traj) + "\n")

    typer.echo(f"Saved to {output_file}")


if __name__ == "__main__":
    typer.run(main)
