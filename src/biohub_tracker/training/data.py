from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.annotation_reader import discover_geff_stores, read_geff_graph
from biohub_tracker.models import LineageGraph
from biohub_tracker.preprocessing import PreprocessingConfig, normalize_frame
from biohub_tracker.training.targets import generate_centroid_heatmap
from biohub_tracker.zarr_reader import VolumeDatasetReader, discover_zarr_stores


@dataclass(frozen=True, slots=True)
class TrainingPair:
    dataset: str
    image_store: Path
    lineage_store: Path


def discover_training_pairs(competition_root: str | Path) -> list[TrainingPair]:
    """Pair training image and GEFF stores by their dataset stem."""
    root = Path(competition_root)
    images = {path.stem: path for path in discover_zarr_stores(root, "train")}
    lineages = {path.stem: path for path in discover_geff_stores(root)}
    duplicates = len(images) != len(discover_zarr_stores(root, "train")) or len(lineages) != len(
        discover_geff_stores(root)
    )
    if duplicates:
        raise ValueError("Training Zarr and GEFF stems must each be unique")
    missing_images = sorted(set(lineages) - set(images))
    missing_lineages = sorted(set(images) - set(lineages))
    if missing_images or missing_lineages:
        raise ValueError(
            "Training image/lineage pairing failed: "
            f"missing_images={missing_images[:10]}, missing_lineages={missing_lineages[:10]}"
        )
    if not images:
        raise FileNotFoundError(
            f"No paired training .zarr/.geff stores found below {root / 'train'}"
        )
    return [
        TrainingPair(dataset=name, image_store=images[name], lineage_store=lineages[name])
        for name in sorted(images)
    ]


def _extract_patch(
    frame: NDArray[np.float32],
    start_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
) -> NDArray[np.float32]:
    patch = np.zeros(patch_shape_zyx, dtype=np.float32)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for start, length, frame_length in zip(start_zyx, patch_shape_zyx, frame.shape, strict=True):
        source_start = max(0, start)
        source_end = min(frame_length, start + length)
        target_start = source_start - start
        target_end = target_start + max(0, source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    patch[tuple(target_slices)] = frame[tuple(source_slices)]
    return patch


class CentroidPatchDataset:
    """Lazy positive-centered Zarr patches with on-demand Gaussian heatmap targets."""

    def __init__(
        self,
        competition_root: str | Path,
        *,
        patch_shape_zyx: tuple[int, int, int],
        sigma_um: float,
        preprocessing: PreprocessingConfig,
        jitter_voxels_zyx: tuple[int, int, int] = (2, 8, 8),
        seed: int = 42,
    ) -> None:
        if any(size < 1 for size in patch_shape_zyx):
            raise ValueError("patch_shape_zyx must contain positive sizes")
        if any(value < 0 for value in jitter_voxels_zyx):
            raise ValueError("jitter_voxels_zyx must be non-negative")
        self.reader = VolumeDatasetReader(competition_root, split="train")
        self.pairs = discover_training_pairs(competition_root)
        self.graphs: dict[str, LineageGraph] = {
            pair.dataset: read_geff_graph(pair.lineage_store) for pair in self.pairs
        }
        self.nodes_by_id = {
            pair.dataset: {node.node_id: node for node in self.graphs[pair.dataset].nodes}
            for pair in self.pairs
        }
        self.samples = [
            (pair.dataset, node.node_id)
            for pair in self.pairs
            for node in self.graphs[pair.dataset].nodes
        ]
        self.patch_shape_zyx = patch_shape_zyx
        self.sigma_um = sigma_um
        self.preprocessing = preprocessing
        self.jitter_voxels_zyx = jitter_voxels_zyx
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        dataset, node_id = self.samples[index]
        graph = self.graphs[dataset]
        anchor = self.nodes_by_id[dataset][node_id]
        metadata = self.reader.metadata(dataset)
        frame = normalize_frame(self.reader.read_frame(dataset, anchor.t), self.preprocessing)
        rng = np.random.default_rng(self.seed + index)
        jitter = tuple(
            int(rng.integers(-limit, limit + 1)) if limit else 0 for limit in self.jitter_voxels_zyx
        )
        center = np.asarray(anchor.position_zyx, dtype=np.float64)
        half = np.asarray(self.patch_shape_zyx, dtype=np.int64) // 2
        start_array = np.floor(center).astype(np.int64) - half + np.asarray(jitter)
        start = (int(start_array[0]), int(start_array[1]), int(start_array[2]))
        patch = _extract_patch(frame, start, self.patch_shape_zyx)
        centroids = [
            tuple(
                float(value - offset)
                for value, offset in zip(node.position_zyx, start, strict=True)
            )
            for node in graph.nodes
            if node.t == anchor.t
        ]
        target = generate_centroid_heatmap(
            self.patch_shape_zyx,
            centroids,
            voxel_spacing_zyx=metadata.voxel_spacing_zyx,
            sigma_um=self.sigma_um,
        )
        return patch[None], target[None]
