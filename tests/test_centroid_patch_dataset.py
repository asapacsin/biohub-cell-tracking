from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracker.preprocessing import PreprocessingConfig
from biohub_tracker.training.data import (
    AugmentationConfig,
    CentroidPatchDataset,
    DatasetView,
    PatchMixConfig,
    apply_geometric_augmentation,
    choose_patch_kind,
    empty_crop_start,
    near_miss_crop_start,
)


def _write_sparse_training_pair(root: Path) -> None:
    """Larger XY so empty crops and near-miss offsets are feasible."""
    zarr = pytest.importorskip("zarr")
    train = root / "train"
    train.mkdir(parents=True)
    image_group = zarr.open_group(str(train / "demo.zarr"), mode="w")
    image = np.zeros((3, 8, 64, 64), dtype=np.uint16)
    image[0, 3, 16, 16] = 200
    image[1, 3, 20, 18] = 200
    image[2, 3, 24, 20] = 200
    image_group.create_array("0", data=image, chunks=(1, 8, 64, 64))
    image_group.attrs["multiscales"] = [
        {
            "axes": [{"name": axis} for axis in ("t", "z", "y", "x")],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                    ],
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
        "z": [3, 3, 3],
        "y": [16, 20, 24],
        "x": [16, 18, 20],
    }.items():
        prop = props.create_group(name)
        prop.create_array("values", data=np.asarray(values, dtype=np.int64))
    edges = lineage.create_group("edges")
    edges.create_array("ids", data=np.asarray([[10, 11], [11, 12]], dtype=np.uint64))


def _dataset(root: Path, **kwargs: object) -> CentroidPatchDataset:
    defaults: dict[str, object] = {
        "patch_shape_zyx": (8, 32, 32),
        "sigma_um": 2.0,
        "preprocessing": PreprocessingConfig(lower_percentile=0, upper_percentile=100),
        "jitter_voxels_zyx": (1, 4, 4),
        "seed": 42,
        "frame_cache_size": 2,
        "augmentation": AugmentationConfig(enabled=False),
    }
    defaults.update(kwargs)
    return CentroidPatchDataset(root, **defaults)  # type: ignore[arg-type]


def test_nodes_by_time_indexes_centroids(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    dataset = _dataset(tmp_path)
    assert ("demo", 0) in dataset.nodes_by_time
    assert dataset.nodes_by_time[("demo", 0)] == [(3.0, 16.0, 16.0)]
    assert len(dataset.nodes_by_time[("demo", 1)]) == 1


def test_frame_cache_avoids_repeated_reads(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    dataset = _dataset(tmp_path, augmentation=AugmentationConfig(enabled=False))
    dataset.eval()
    original = dataset.reader.read_frame
    calls: list[tuple[str, int]] = []

    def tracked(dataset_name: str, t: int) -> np.ndarray:
        calls.append((dataset_name, t))
        return original(dataset_name, t)

    dataset.reader.read_frame = tracked  # type: ignore[method-assign]
    _ = dataset[0]
    _ = dataset[0]
    assert calls == [("demo", 0)]


def test_set_epoch_changes_train_samples(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    dataset = _dataset(
        tmp_path,
        jitter_voxels_zyx=(2, 8, 8),
        patch_mix=PatchMixConfig(positive=1.0, near_miss=0.0, empty=0.0),
        augmentation=AugmentationConfig(enabled=False),
    )
    dataset.train()
    dataset.set_epoch(0)
    a = dataset[0][0].copy()
    dataset.set_epoch(1)
    b = dataset[0][0].copy()
    assert not np.array_equal(a, b)


def test_eval_mode_is_deterministic_across_epochs(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    dataset = _dataset(tmp_path, jitter_voxels_zyx=(2, 8, 8))
    view = DatasetView(dataset, train=False)
    dataset.set_epoch(0)
    a_img, a_tgt = view[0]
    dataset.set_epoch(7)
    b_img, b_tgt = view[0]
    np.testing.assert_array_equal(a_img, b_img)
    np.testing.assert_array_equal(a_tgt, b_tgt)
    assert float(a_tgt.max()) == 1.0


def test_choose_patch_kind_mix_frequencies() -> None:
    mix = PatchMixConfig(positive=0.80, near_miss=0.15, empty=0.05)
    rng = np.random.default_rng(0)
    counts = {"positive": 0, "near_miss": 0, "empty": 0}
    draws = 20_000
    for _ in range(draws):
        counts[choose_patch_kind(rng, mix)] += 1
    assert abs(counts["positive"] / draws - 0.80) < 0.02
    assert abs(counts["near_miss"] / draws - 0.15) < 0.02
    assert abs(counts["empty"] / draws - 0.05) < 0.02


def test_near_miss_keeps_center_clear_and_zero_target() -> None:
    spacing = (1.625, 0.40625, 0.40625)
    patch = (8, 32, 32)
    anchor = (3.0, 16.0, 16.0)
    centroids = [anchor]
    rng = np.random.default_rng(1)
    start = near_miss_crop_start(
        anchor_zyx=anchor,
        centroids_zyx=centroids,
        patch_shape_zyx=patch,
        voxel_spacing_zyx=spacing,
        positive_center_radius_um=6.0,
        rng=rng,
    )
    assert start is not None
    half = np.asarray(patch, dtype=np.float64) / 2.0
    patch_center_world = np.asarray(start, dtype=np.float64) + half
    for centroid in centroids:
        delta = (np.asarray(centroid) - patch_center_world) * np.asarray(spacing)
        assert float(np.linalg.norm(delta)) >= 6.0 - 1e-6
    # Source cell should land in the preferred annulus in XY (40-70% of half-extent).
    source_in_patch = np.asarray(anchor) - np.asarray(start, dtype=np.float64)
    offset_yx = source_in_patch[1:] - half[1:]
    half_yx = half[1:]
    radial = np.linalg.norm(offset_yx / half_yx)
    assert 0.40 - 0.05 <= radial <= 0.70 + 0.05


def test_empty_crop_excludes_centroids_with_margin() -> None:
    spacing = (1.625, 0.40625, 0.40625)
    patch = (8, 32, 32)
    frame_shape = (8, 64, 64)
    centroids = [(3.0, 16.0, 16.0)]
    rng = np.random.default_rng(2)
    start = empty_crop_start(
        frame_shape_zyx=frame_shape,
        centroids_zyx=centroids,
        patch_shape_zyx=patch,
        voxel_spacing_zyx=spacing,
        exclusion_margin_um=4.0,
        rng=rng,
    )
    assert start is not None
    margin_vox = tuple(4.0 / s for s in spacing)
    expanded = (
        start[0] - margin_vox[0],
        start[1] - margin_vox[1],
        start[2] - margin_vox[2],
        start[0] + patch[0] + margin_vox[0],
        start[1] + patch[1] + margin_vox[1],
        start[2] + patch[2] + margin_vox[2],
    )
    for centroid in centroids:
        z, y, x = centroid
        assert not (
            expanded[0] <= z < expanded[3]
            and expanded[1] <= y < expanded[4]
            and expanded[2] <= x < expanded[5]
        )


def test_train_negatives_have_zero_targets(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    near = _dataset(
        tmp_path,
        patch_mix=PatchMixConfig(positive=0.0, near_miss=1.0, empty=0.0),
        augmentation=AugmentationConfig(enabled=False),
    )
    near.train()
    near.set_epoch(0)
    _, near_tgt = near[0]
    assert float(near_tgt.max()) == 0.0

    empty = _dataset(
        tmp_path,
        patch_mix=PatchMixConfig(positive=0.0, near_miss=0.0, empty=1.0),
        augmentation=AugmentationConfig(enabled=False),
    )
    empty.train()
    empty.set_epoch(0)
    _, empty_tgt = empty[0]
    assert float(empty_tgt.max()) == 0.0


def test_geometric_augmentation_moves_target_with_image() -> None:
    image = np.zeros((1, 4, 8, 8), dtype=np.float32)
    target = np.zeros((1, 4, 8, 8), dtype=np.float32)
    image[0, 1, 2, 3] = 1.0
    target[0, 1, 2, 3] = 1.0

    class _Rng:
        def __init__(self) -> None:
            self._randoms = iter([0.0, 0.9, 0.0])  # flip Y, skip X, apply rot90

        def random(self) -> float:
            return next(self._randoms)

        def integers(self, low: int, high: int | None = None) -> int:
            del low, high
            return 1

    out_img, out_tgt = apply_geometric_augmentation(
        image,
        target,
        AugmentationConfig(enabled=True, flip_prob=0.5, rot90_prob=1.0),
        _Rng(),  # type: ignore[arg-type]
    )
    peak_img = tuple(int(v) for v in np.argwhere(out_img[0] == 1.0)[0])
    peak_tgt = tuple(int(v) for v in np.argwhere(out_tgt[0] == 1.0)[0])
    assert peak_img == peak_tgt
    assert peak_img != (1, 2, 3)


def test_training_config_loads_patch_pipeline_fields() -> None:
    from biohub_tracker.training.config import load_training_config

    config = load_training_config("configs/training.yaml")
    assert config.detector.frame_cache_size == 2
    assert config.detector.batch_size == 8
    assert config.detector.epochs == 3
    assert config.detector.frame_grouped_batches is True
    assert config.detector.use_amp is True
    assert config.detector.augmentation.noise_std == pytest.approx(0.0)
    assert config.detector.augmentation.blur_sigma_px == (0.0, 0.0)
    assert config.detector.patch_mix.positive == pytest.approx(0.80)
    assert config.detector.patch_mix.near_miss == pytest.approx(0.15)
    assert config.detector.patch_mix.empty == pytest.approx(0.05)
    assert config.detector.positive_center_radius_um == pytest.approx(6.0)
    assert config.detector.empty_exclusion_margin_um == pytest.approx(4.0)
    assert config.detector.augmentation.enabled is True


def test_empty_sample_prefers_anchor_frame(tmp_path: Path) -> None:
    _write_sparse_training_pair(tmp_path)
    dataset = _dataset(
        tmp_path,
        patch_mix=PatchMixConfig(positive=0.0, near_miss=0.0, empty=1.0),
        augmentation=AugmentationConfig(enabled=False),
    )
    original = dataset._cached_frame
    seen: list[tuple[str, int]] = []

    def tracked(dataset_name: str, t: int) -> np.ndarray:
        seen.append((dataset_name, t))
        return original(dataset_name, t)

    dataset._cached_frame = tracked  # type: ignore[method-assign]
    dataset.train()
    dataset.set_epoch(0)
    _ = dataset[0]
    assert seen[0] == ("demo", 0)


def test_frame_grouped_batch_sampler_keeps_single_frame(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    train = tmp_path / "train"
    train.mkdir(parents=True)
    image_group = zarr.open_group(str(train / "demo.zarr"), mode="w")
    image = np.zeros((2, 8, 64, 64), dtype=np.uint16)
    image_group.create_array("0", data=image, chunks=(1, 8, 64, 64))
    image_group.attrs["multiscales"] = [
        {
            "axes": [{"name": axis} for axis in ("t", "z", "y", "x")],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                    ],
                }
            ],
        }
    ]
    lineage = zarr.open_group(str(train / "demo.geff"), mode="w")
    lineage.attrs["geff"] = {"geff_version": "1.1", "directed": True}
    nodes = lineage.create_group("nodes")
    # Three cells on t=0, two on t=1.
    node_ids = [10, 11, 12, 13, 14]
    nodes.create_array("ids", data=np.asarray(node_ids, dtype=np.uint64))
    props = nodes.create_group("props")
    for name, values in {
        "t": [0, 0, 0, 1, 1],
        "z": [3, 3, 3, 3, 3],
        "y": [16, 20, 24, 16, 20],
        "x": [16, 18, 20, 16, 18],
    }.items():
        prop = props.create_group(name)
        prop.create_array("values", data=np.asarray(values, dtype=np.int64))
    edges = lineage.create_group("edges")
    edges.create_array("ids", data=np.asarray([[10, 13], [11, 14]], dtype=np.uint64))

    from biohub_tracker.training.samplers import FrameGroupedBatchSampler

    dataset = _dataset(tmp_path)
    subset_indices = list(range(len(dataset)))
    sampler = FrameGroupedBatchSampler(
        dataset, subset_indices, batch_size=2, seed=0
    )
    sampler.set_epoch(0)
    batches = list(sampler)
    assert len(batches) == 3  # ceil(3/2)+ceil(2/2)
    for batch in batches:
        frames = {
            int(dataset.nodes_by_id[dataset.samples[subset_indices[i]][0]][
                dataset.samples[subset_indices[i]][1]
            ].t)
            for i in batch
        }
        assert len(frames) == 1
