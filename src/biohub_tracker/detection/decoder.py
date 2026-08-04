from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter  # type: ignore[import-untyped]

from biohub_tracker.models import DetectionCandidate


@dataclass(frozen=True, slots=True)
class DetectionDecoderConfig:
    threshold: float = 0.35
    adaptive_quantile: float = 0.995
    nms_radius_um: float = 3.0
    refinement_radius_voxels: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("decoder threshold must be in [0, 1]")
        if not 0 <= self.adaptive_quantile <= 1:
            raise ValueError("adaptive_quantile must be in [0, 1]")
        if self.nms_radius_um <= 0:
            raise ValueError("nms_radius_um must be positive")
        if self.refinement_radius_voxels < 0:
            raise ValueError("refinement_radius_voxels must be non-negative")


def _refine_peak(
    heatmap: NDArray[np.float32], peak: tuple[int, int, int], radius: int
) -> tuple[float, float, float]:
    if radius == 0:
        return tuple(float(value) for value in peak)  # type: ignore[return-value]
    slices = tuple(
        slice(max(0, value - radius), min(size, value + radius + 1))
        for value, size in zip(peak, heatmap.shape, strict=True)
    )
    patch = np.maximum(heatmap[slices], 0.0).astype(np.float64, copy=False)
    total = float(patch.sum())
    if total <= 0:
        return tuple(float(value) for value in peak)  # type: ignore[return-value]
    starts = np.asarray([item.start for item in slices], dtype=np.float64)
    coordinates = np.indices(patch.shape, dtype=np.float64)
    centroid = starts + np.asarray([(coordinates[axis] * patch).sum() / total for axis in range(3)])
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def decode_heatmap(
    heatmap_zyx: NDArray[np.generic],
    *,
    dataset: str,
    t: int,
    voxel_spacing_zyx: tuple[float, float, float],
    config: DetectionDecoderConfig,
) -> list[DetectionCandidate]:
    """Adaptive threshold -> anisotropic 3D NMS -> local subvoxel refinement."""
    heatmap = np.asarray(heatmap_zyx, dtype=np.float32)
    if heatmap.ndim != 3:
        raise ValueError(f"Expected a ZYX heatmap, got shape {heatmap.shape}")
    if not np.all(np.isfinite(heatmap)):
        raise ValueError("Heatmap must contain only finite values")
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("voxel_spacing_zyx must contain three positive values")

    adaptive = float(np.quantile(heatmap, config.adaptive_quantile))
    threshold = max(config.threshold, adaptive)
    radii = np.maximum(1, np.ceil(config.nms_radius_um / spacing).astype(np.int64))
    footprint_size = tuple(int(2 * radius + 1) for radius in radii)
    local_max = maximum_filter(heatmap, size=footprint_size, mode="nearest")
    peaks = np.argwhere((heatmap >= threshold) & (heatmap == local_max))
    ordered = sorted(
        ((int(peak[0]), int(peak[1]), int(peak[2])) for peak in peaks),
        key=lambda peak: (-float(heatmap[peak]), peak),
    )
    return [
        DetectionCandidate(
            dataset=dataset,
            t=t,
            z=position[0],
            y=position[1],
            x=position[2],
            score=float(heatmap[peak]),
        )
        for peak in ordered
        for position in [_refine_peak(heatmap, peak, config.refinement_radius_voxels)]
    ]
