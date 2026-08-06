from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracker.association import CandidateGraphConfig, build_candidate_graph
from biohub_tracker.association.learned import (
    LinearAssociationArtifact,
    LinearAssociationScorer,
    candidate_feature_matrix,
    fit_linear_association_model,
)
from biohub_tracker.models import DetectionCandidate
from biohub_tracker.preprocessing import PreprocessingConfig
from biohub_tracker.training.association import (
    AssociationTrainingConfig,
    train_association_model,
)
from biohub_tracker.training.config import load_training_config
from biohub_tracker.training.data import CentroidPatchDataset, discover_training_pairs


def _write_training_pair(root: Path) -> None:
    zarr = pytest.importorskip("zarr")
    train = root / "train"
    train.mkdir(parents=True)
    image_group = zarr.open_group(str(train / "demo.zarr"), mode="w")
    image = np.zeros((3, 4, 16, 16), dtype=np.uint16)
    image[0, 1, 7, 7] = 100
    image[1, 1, 8, 7] = 100
    image[2, 1, 9, 7] = 100
    image_group.create_array("0", data=image, chunks=(1, 4, 16, 16))
    image_group.attrs["multiscales"] = [
        {
            "axes": [{"name": axis} for axis in ("t", "z", "y", "x")],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 2.0, 0.5, 0.5]}],
                }
            ],
        }
    ]
    lineage = zarr.open_group(str(train / "demo.geff"), mode="w")
    lineage.attrs["geff"] = {"geff_version": "1.1", "directed": True}
    nodes = lineage.create_group("nodes")
    nodes.create_array("ids", data=np.asarray([10, 11, 12], dtype=np.uint64))
    props = nodes.create_group("props")
    for name, values in {
        "t": [0, 1, 2],
        "z": [1, 1, 1],
        "y": [7, 8, 9],
        "x": [7, 7, 7],
    }.items():
        prop = props.create_group(name)
        prop.create_array("values", data=np.asarray(values, dtype=np.int64))
    edges = lineage.create_group("edges")
    edges.create_array("ids", data=np.asarray([[10, 11], [11, 12]], dtype=np.uint64))


def test_centroid_patch_dataset_reads_paired_zarr_and_geff(tmp_path: Path) -> None:
    _write_training_pair(tmp_path)
    assert [pair.dataset for pair in discover_training_pairs(tmp_path)] == ["demo"]
    dataset = CentroidPatchDataset(
        tmp_path,
        patch_shape_zyx=(4, 16, 16),
        sigma_um=1.0,
        preprocessing=PreprocessingConfig(lower_percentile=0, upper_percentile=100),
        jitter_voxels_zyx=(0, 0, 0),
    )
    image, target = dataset[0]
    assert len(dataset) == 3
    assert image.shape == target.shape == (1, 4, 16, 16)
    assert image.dtype == target.dtype == np.float32
    assert target.max() == 1.0


def test_association_training_exports_runtime_artifact(tmp_path: Path) -> None:
    _write_training_pair(tmp_path)
    destination = tmp_path / "artifacts" / "association.json"
    artifact = train_association_model(
        tmp_path,
        destination,
        AssociationTrainingConfig(
            candidate_graph=CandidateGraphConfig(
                max_neighbors=3,
                max_gap=2,
                max_speed_um_per_frame=5.0,
                divisions_enabled=False,
            )
        ),
    )
    assert destination.is_file()
    assert LinearAssociationArtifact.load(destination) == artifact


def test_linear_association_scorer_learns_and_scores_candidates(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, 14))
    labels = (features[:, 2] < 0).astype(np.int8)
    artifact = fit_linear_association_model(features, labels)
    path = artifact.save(tmp_path / "model.json")

    graph = build_candidate_graph(
        [
            DetectionCandidate("demo", 0, 1, 1, 1, 1.0),
            DetectionCandidate("demo", 1, 1, 2, 1, 1.0),
        ],
        voxel_spacing_zyx=(1.0, 1.0, 1.0),
        config=CandidateGraphConfig(max_gap=1, divisions_enabled=False),
    )
    assert candidate_feature_matrix(graph).shape == (1, 14)
    LinearAssociationScorer.load(path).score(graph)
    assert np.isfinite(graph.edges[0].score)


def test_training_configuration_loads_without_torch() -> None:
    config = load_training_config("configs/training.yaml")
    assert config.detector.patch_shape_zyx == (32, 128, 128)
    assert config.detector.unet.depth == 3
    assert config.detector_seeds == (42,)
    assert config.association.candidate_graph.max_gap == 2
