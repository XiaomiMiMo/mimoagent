import json

import pytest
import typer

from mimoagent.run.extra.batch import load_jsonl_instances


def test_load_jsonl_instances_allows_heterogeneous_rows(tmp_path):
    path = tmp_path / "instances.jsonl"
    rows = [
        {"instance_id": "one", "routing": {"node_selector": {"pool": "a"}}},
        {"instance_id": "two", "verdict": {"valuable": True}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert load_jsonl_instances(path) == rows


def test_load_jsonl_instances_reports_bad_line(tmp_path):
    path = tmp_path / "instances.jsonl"
    path.write_text('{"instance_id": "one"}\nnot-json\n')

    with pytest.raises(typer.BadParameter, match=r"line 2"):
        load_jsonl_instances(path)
