from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

from biohub_tracker.annotation_reader import discover_geff_stores, read_geff_graph
from biohub_tracker.models import LineageGraph, LineageNode
from biohub_tracker.preprocessing import PreprocessingConfig, normalize_frame
from biohub_tracker.training.targets import generate_centroid_heatmap
from biohub_tracker.zarr_reader import VolumeDatasetReader, discover_zarr_stores

PatchKind = Literal["positive", "near_miss", "empty"]


@dataclass(frozen=True, slots=True)
class TrainingPair:
    dataset: str
    image_store: Path
    lineage_store: Path


@dataclass(frozen=True, slots=True)
class PatchMixConfig:
    positive: float = 0.80
    near_miss: float = 0.15
    empty: float = 0.05

    def __post_init__(self) -> None:
        total = self.positive + self.near_miss + self.empty
        if min(self.positive, self.near_miss, self.empty) < 0:
            raise ValueError("patch mix probabilities must be non-negative")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("patch mix probabilities must sum to 1")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    enabled: bool = True
    flip_prob: float = 0.5
    rot90_prob: float = 0.5
    intensity_scale: tuple[float, float] = (0.9, 1.1)
    intensity_shift: tuple[float, float] = (-0.05, 0.05)
    noise_std: float = 0.02
    blur_sigma_px: tuple[float, float] = (0.0, 0.8)

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_prob <= 1.0 or not 0.0 <= self.rot90_prob <= 1.0:
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.blur_sigma_px[0] < 0 or self.blur_sigma_px[1] < self.blur_sigma_px[0]:
            raise ValueError("blur_sigma_px must be a non-negative low<=high range")


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


def choose_patch_kind(rng: np.random.Generator, mix: PatchMixConfig) -> PatchKind:
    draw = float(rng.random())
    if draw < mix.positive:
        return "positive"
    if draw < mix.positive + mix.near_miss:
        return "near_miss"
    return "empty"


def _physical_distance(
    a_zyx: Sequence[float],
    b_zyx: Sequence[float],
    voxel_spacing_zyx: tuple[float, float, float],
) -> float:
    delta = (
        np.asarray(a_zyx, dtype=np.float64) - np.asarray(b_zyx, dtype=np.float64)
    ) * np.asarray(voxel_spacing_zyx, dtype=np.float64)
    return float(np.linalg.norm(delta))


def _patch_center_from_start(
    start_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
) -> tuple[float, float, float]:
    return (
        float(start_zyx[0] + patch_shape_zyx[0] / 2.0),
        float(start_zyx[1] + patch_shape_zyx[1] / 2.0),
        float(start_zyx[2] + patch_shape_zyx[2] / 2.0),
    )


def _center_is_clear(
    patch_center_zyx: Sequence[float],
    centroids_zyx: Sequence[Sequence[float]],
    *,
    voxel_spacing_zyx: tuple[float, float, float],
    positive_center_radius_um: float,
) -> bool:
    return all(
        _physical_distance(patch_center_zyx, centroid, voxel_spacing_zyx)
        >= positive_center_radius_um
        for centroid in centroids_zyx
    )


def near_miss_crop_start(
    *,
    anchor_zyx: Sequence[float],
    centroids_zyx: Sequence[Sequence[float]],
    patch_shape_zyx: tuple[int, int, int],
    voxel_spacing_zyx: tuple[float, float, float],
    positive_center_radius_um: float,
    rng: np.random.Generator,
    attempts: int = 48,
) -> tuple[int, int, int] | None:
    """Offset crop so the patch center is clear of centroids; prefer source in XY annulus."""
    half = np.asarray(patch_shape_zyx, dtype=np.float64) / 2.0
    anchor = np.asarray(anchor_zyx, dtype=np.float64)
    for _ in range(attempts):
        # Prefer 40-70% of half-extent in XY; milder offset in Z for thin patches.
        radial = float(rng.uniform(0.40, 0.70))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        offset_y = radial * half[1] * np.cos(angle)
        offset_x = radial * half[2] * np.sin(angle)
        z_frac = float(rng.uniform(0.20, 0.55)) if half[0] > 1 else 0.0
        sign_z = 1.0 if rng.random() < 0.5 else -1.0
        offset_z = sign_z * z_frac * half[0]
        # Place source cell at center + offset => start = floor(anchor - half - offset)
        offset = np.asarray([offset_z, offset_y, offset_x], dtype=np.float64)
        start_array = np.floor(anchor - half - offset).astype(np.int64)
        start = (int(start_array[0]), int(start_array[1]), int(start_array[2]))
        patch_center = _patch_center_from_start(start, patch_shape_zyx)
        if _center_is_clear(
            patch_center,
            centroids_zyx,
            voxel_spacing_zyx=voxel_spacing_zyx,
            positive_center_radius_um=positive_center_radius_um,
        ):
            return start
    return None


def _expanded_box_contains(
    centroid_zyx: Sequence[float],
    start_zyx: tuple[int, int, int],
    patch_shape_zyx: tuple[int, int, int],
    margin_voxels: Sequence[float],
) -> bool:
    z, y, x = (float(v) for v in centroid_zyx)
    return (
        start_zyx[0] - margin_voxels[0] <= z < start_zyx[0] + patch_shape_zyx[0] + margin_voxels[0]
        and start_zyx[1] - margin_voxels[1]
        <= y
        < start_zyx[1] + patch_shape_zyx[1] + margin_voxels[1]
        and start_zyx[2] - margin_voxels[2]
        <= x
        < start_zyx[2] + patch_shape_zyx[2] + margin_voxels[2]
    )


def empty_crop_start(
    *,
    frame_shape_zyx: tuple[int, int, int],
    centroids_zyx: Sequence[Sequence[float]],
    patch_shape_zyx: tuple[int, int, int],
    voxel_spacing_zyx: tuple[float, float, float],
    exclusion_margin_um: float,
    rng: np.random.Generator,
    attempts: int = 64,
) -> tuple[int, int, int] | None:
    """Random crop with no annotated centroid inside the crop or exclusion margin."""
    margin_voxels = tuple(exclusion_margin_um / spacing for spacing in voxel_spacing_zyx)
    max_starts = [max(1, frame_shape_zyx[axis] - patch_shape_zyx[axis] + 1) for axis in range(3)]
    # Allow starts that hang off the border (extract pads with zeros), but prefer in-bounds.
    for _ in range(attempts):
        start = (
            int(rng.integers(-patch_shape_zyx[0] // 4, max_starts[0])),
            int(rng.integers(0, max_starts[1])),
            int(rng.integers(0, max_starts[2])),
        )
        if any(
            _expanded_box_contains(centroid, start, patch_shape_zyx, margin_voxels)
            for centroid in centroids_zyx
        ):
            continue
        return start
    return None


def apply_geometric_augmentation(
    image: NDArray[np.float32],
    target: NDArray[np.float32],
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply matching XY flips / 90° rotations to image and target batches (C,Z,Y,X)."""
    if not config.enabled:
        return image, target
    out_image = image
    out_target = target
    if float(rng.random()) < config.flip_prob:
        out_image = np.flip(out_image, axis=2).copy()
        out_target = np.flip(out_target, axis=2).copy()
    if float(rng.random()) < config.flip_prob:
        out_image = np.flip(out_image, axis=3).copy()
        out_target = np.flip(out_target, axis=3).copy()
    if float(rng.random()) < config.rot90_prob:
        k = int(rng.integers(0, 4))
        if k:
            out_image = np.rot90(out_image, k=k, axes=(2, 3)).copy()
            out_target = np.rot90(out_target, k=k, axes=(2, 3)).copy()
    return out_image.astype(np.float32, copy=False), out_target.astype(np.float32, copy=False)


def apply_photometric_augmentation(
    image: NDArray[np.float32],
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    if not config.enabled:
        return image
    scale = float(rng.uniform(*config.intensity_scale))
    shift = float(rng.uniform(*config.intensity_shift))
    out = image * scale + shift
    if config.noise_std > 0:
        out = out + rng.normal(0.0, config.noise_std, size=out.shape).astype(np.float32)
    blur_sigma = float(rng.uniform(*config.blur_sigma_px))
    if blur_sigma > 0:
        # Blur spatial dims only; channel axis untouched.
        blurred = np.empty_like(out)
        for channel in range(out.shape[0]):
            blurred[channel] = gaussian_filter(out[channel], sigma=(0.0, blur_sigma, blur_sigma))
        out = blurred
    return out.astype(np.float32, copy=False)


def _rng_for(seed: int, epoch: int, index: int, stream: int = 0) -> np.random.Generator:
    # Stable, worker-friendly seeding without shared Generator mutation races.
    material = np.random.SeedSequence([seed & 0xFFFFFFFF, epoch, index, stream])
    return np.random.default_rng(material)


class DatasetView:
    """Fixed train/eval view so validation does not mutate the shared dataset mode."""

    def __init__(self, dataset: CentroidPatchDataset, *, train: bool) -> None:
        self.dataset = dataset
        self.train = train

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self.dataset.get_item(index, train=self.train)


class CentroidPatchDataset:
    """Lazy Zarr patches with frame cache, mixed sampling, and train-time augmentation."""

    def __init__(
        self,
        competition_root: str | Path,
        *,
        patch_shape_zyx: tuple[int, int, int],
        sigma_um: float,
        preprocessing: PreprocessingConfig,
        jitter_voxels_zyx: tuple[int, int, int] = (2, 8, 8),
        seed: int = 42,
        frame_cache_size: int = 4,
        patch_mix: PatchMixConfig | None = None,
        positive_center_radius_um: float = 6.0,
        empty_exclusion_margin_um: float = 4.0,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        if any(size < 1 for size in patch_shape_zyx):
            raise ValueError("patch_shape_zyx must contain positive sizes")
        if any(value < 0 for value in jitter_voxels_zyx):
            raise ValueError("jitter_voxels_zyx must be non-negative")
        if frame_cache_size < 1:
            raise ValueError("frame_cache_size must be positive")
        if positive_center_radius_um <= 0 or empty_exclusion_margin_um < 0:
            raise ValueError("positive_center_radius_um must be > 0 and margin >= 0")
        self.reader = VolumeDatasetReader(competition_root, split="train")
        self.pairs = discover_training_pairs(competition_root)
        self.graphs: dict[str, LineageGraph] = {
            pair.dataset: read_geff_graph(pair.lineage_store) for pair in self.pairs
        }
        self.nodes_by_id: dict[str, dict[int, LineageNode]] = {
            pair.dataset: {node.node_id: node for node in self.graphs[pair.dataset].nodes}
            for pair in self.pairs
        }
        self.nodes_by_time: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
        for pair in self.pairs:
            buckets: dict[int, list[tuple[float, float, float]]] = {}
            for node in self.graphs[pair.dataset].nodes:
                buckets.setdefault(node.t, []).append(
                    (
                        float(node.position_zyx[0]),
                        float(node.position_zyx[1]),
                        float(node.position_zyx[2]),
                    )
                )
            for t, centroids in buckets.items():
                self.nodes_by_time[(pair.dataset, t)] = centroids
        self.frame_keys = sorted(self.nodes_by_time)
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
        self.frame_cache_size = frame_cache_size
        self.patch_mix = patch_mix or PatchMixConfig()
        self.positive_center_radius_um = positive_center_radius_um
        self.empty_exclusion_margin_um = empty_exclusion_margin_um
        self.augmentation = augmentation or AugmentationConfig()
        self._epoch = 0
        self._train = True
        self._frame_cache: OrderedDict[tuple[str, int], NDArray[np.float32]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def __getitem__(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self.get_item(index, train=self._train)

    def get_item(
        self, index: int, *, train: bool | None = None
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        use_train = self._train if train is None else train
        dataset_name, node_id = self.samples[index]
        anchor = self.nodes_by_id[dataset_name][node_id]
        metadata = self.reader.metadata(dataset_name)
        spacing = metadata.voxel_spacing_zyx
        if use_train:
            rng = _rng_for(self.seed, self._epoch, index, stream=0)
            kind = choose_patch_kind(rng, self.patch_mix)
        else:
            rng = _rng_for(self.seed, 0, index, stream=0)
            kind = "positive"

        if kind == "positive":
            image, target = self._positive_sample(
                dataset_name, anchor, spacing, rng=rng, jitter=use_train
            )
        elif kind == "near_miss":
            image, target = self._near_miss_sample(dataset_name, anchor, spacing, rng=rng)
        else:
            image, target = self._empty_sample(rng=rng)

        image = image[None]
        target = target[None]
        if use_train and self.augmentation.enabled:
            image, target = apply_geometric_augmentation(image, target, self.augmentation, rng)
            image = apply_photometric_augmentation(image, self.augmentation, rng)
        return image.astype(np.float32, copy=False), target.astype(np.float32, copy=False)

    def _cached_frame(self, dataset_name: str, t: int) -> NDArray[np.float32]:
        key = (dataset_name, t)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        frame = normalize_frame(self.reader.read_frame(dataset_name, t), self.preprocessing)
        self._frame_cache[key] = frame
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def _centroids_in_patch(
        self,
        dataset_name: str,
        t: int,
        start: tuple[int, int, int],
    ) -> list[tuple[float, float, float]]:
        return [
            (
                float(centroid[0] - start[0]),
                float(centroid[1] - start[1]),
                float(centroid[2] - start[2]),
            )
            for centroid in self.nodes_by_time.get((dataset_name, t), ())
        ]

    def _positive_sample(
        self,
        dataset_name: str,
        anchor: LineageNode,
        spacing: tuple[float, float, float],
        *,
        rng: np.random.Generator,
        jitter: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        frame = self._cached_frame(dataset_name, anchor.t)
        if jitter:
            jitter_xyz = tuple(
                int(rng.integers(-limit, limit + 1)) if limit else 0
                for limit in self.jitter_voxels_zyx
            )
        else:
            jitter_xyz = (0, 0, 0)
        center = np.asarray(anchor.position_zyx, dtype=np.float64)
        half = np.asarray(self.patch_shape_zyx, dtype=np.int64) // 2
        start_array = np.floor(center).astype(np.int64) - half + np.asarray(jitter_xyz)
        start = (int(start_array[0]), int(start_array[1]), int(start_array[2]))
        patch = _extract_patch(frame, start, self.patch_shape_zyx)
        centroids = self._centroids_in_patch(dataset_name, anchor.t, start)
        target = generate_centroid_heatmap(
            self.patch_shape_zyx,
            centroids,
            voxel_spacing_zyx=spacing,
            sigma_um=self.sigma_um,
        )
        return patch, target

    def _near_miss_sample(
        self,
        dataset_name: str,
        anchor: LineageNode,
        spacing: tuple[float, float, float],
        *,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        centroids = self.nodes_by_time.get((dataset_name, anchor.t), [])
        start = near_miss_crop_start(
            anchor_zyx=anchor.position_zyx,
            centroids_zyx=centroids,
            patch_shape_zyx=self.patch_shape_zyx,
            voxel_spacing_zyx=spacing,
            positive_center_radius_um=self.positive_center_radius_um,
            rng=rng,
        )
        if start is None:
            return self._empty_sample(rng=rng)
        frame = self._cached_frame(dataset_name, anchor.t)
        patch = _extract_patch(frame, start, self.patch_shape_zyx)
        target = np.zeros(self.patch_shape_zyx, dtype=np.float32)
        return patch, target

    def _empty_sample(
        self, *, rng: np.random.Generator
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self.frame_keys:
            raise RuntimeError("No labeled frames available for empty sampling")
        for _ in range(16):
            dataset_name, t = self.frame_keys[int(rng.integers(0, len(self.frame_keys)))]
            metadata = self.reader.metadata(dataset_name)
            frame = self._cached_frame(dataset_name, t)
            start = empty_crop_start(
                frame_shape_zyx=(int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])),
                centroids_zyx=self.nodes_by_time.get((dataset_name, t), ()),
                patch_shape_zyx=self.patch_shape_zyx,
                voxel_spacing_zyx=metadata.voxel_spacing_zyx,
                exclusion_margin_um=self.empty_exclusion_margin_um,
                rng=rng,
            )
            if start is None:
                continue
            patch = _extract_patch(frame, start, self.patch_shape_zyx)
            target = np.zeros(self.patch_shape_zyx, dtype=np.float32)
            return patch, target
        # Last resort: zeros (still a valid negative) if the fixture is too dense.
        return (
            np.zeros(self.patch_shape_zyx, dtype=np.float32),
            np.zeros(self.patch_shape_zyx, dtype=np.float32),
        )
