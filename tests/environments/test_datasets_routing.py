"""Routing tests for the dataset environment registry."""

import pytest

from mimoagent.environments.datasets import DATASET_REGISTRY, detect_dataset_type


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ({"dataset_type": "generic", "docker_image": "registry.example.com/foo:1"}, "generic"),
        ({"dataset_type": "deepswe", "docker_image": "whatever"}, "deepswe"),
        ({"dataset_type": "opensource-code", "docker_image": "registry.example.com/x:1"}, "opensource-code"),
        ({"dataset_type": "arvo", "docker_image": "registry.example.com/arvo:1"}, "arvo"),
        ({"docker_image": "registry.example.com/swe-bench-202605/foo:1"}, "deepswe"),
    ],
)
def test_detect_dataset_type(instance, expected):
    assert detect_dataset_type(instance) == expected


def test_unknown_dataset_type_raises():
    with pytest.raises(ValueError, match="Cannot detect dataset type"):
        detect_dataset_type({"instance_id": "x", "docker_image": "registry.example.com/random:1"})


def test_registry_covers_all_detected_types():
    """Every type detect_dataset_type can return must be constructible."""
    for t in ("deepswe", "generic", "opensource-code", "arvo"):
        assert t in DATASET_REGISTRY


def test_rubric_is_not_a_dataset_type():
    """Rubric judging stacks on any dataset; a stale dataset_type='rubric' fails with a pointer."""
    assert "rubric" not in DATASET_REGISTRY
    with pytest.raises(ValueError, match="no longer a dataset type"):
        detect_dataset_type(
            {"instance_id": "x", "dataset_type": "rubric", "docker_image": "registry.example.com/swe/dev:latest"}
        )
