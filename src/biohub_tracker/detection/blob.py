from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]
from skimage.feature import peak_local_max

from biohub_tracker.coordinates import round_and_clip_voxel_zyx
from biohub_tracker.models import DetectionCandidate


@dataclass(frozen=True, slots=True)
class BlobDetectionConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.8
    gaussian_sigma_um: float = 1.5
    threshold: float = 0.08
    minimum_separation_um: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError("detection percentiles must satisfy 0 <= lower < upper <= 100")
        if self.gaussian_sigma_um <= 0:
            raise ValueError("gaussian_sigma_um must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.minimum_separation_um <= 0:
            raise ValueError("minimum_separation_um must be positive")


def _normalize(frame: NDArray[np.generic], config: BlobDetectionConfig) -> NDArray[np.float32]:
    values = np.asarray(frame, dtype=np.float32)
    low_high = np.percentile(values, [config.lower_percentile, config.upper_percentile])
    low = float(low_high[0])
    high = float(low_high[1])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(frame.shape, dtype=np.float32)
    scaled = (values - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def detect_frame(
    frame_zyx: NDArray[np.generic],
    *,
    dataset: str,
    t: int,
    voxel_spacing_zyx: tuple[float, float, float],
    config: BlobDetectionConfig,
) -> list[DetectionCandidate]:
    if frame_zyx.ndim != 3:
        raise ValueError(f"Expected a (z, y, x) frame; got shape {frame_zyx.shape}")
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("voxel_spacing_zyx must be three positive floats")

    normalized = _normalize(frame_zyx, config)
    sigma_voxels = tuple(float(config.gaussian_sigma_um / axis) for axis in spacing)
    smoothed = ndimage.gaussian_filter(normalized, sigma=sigma_voxels)
    mask = smoothed >= config.threshold
    if not np.any(mask):
        return []

    min_distance = max(
        1,
        round(config.minimum_separation_um / float(np.min(spacing))),
    )
    peaks = peak_local_max(  # type: ignore[no-untyped-call]
        smoothed,
        min_distance=min_distance,
        threshold_abs=config.threshold,
        exclude_border=False,
        labels=mask.astype(np.uint8),
    )
    shape_zyx = tuple(int(size) for size in frame_zyx.shape)
    candidates: list[DetectionCandidate] = []
    for peak in peaks:
        z, y, x = round_and_clip_voxel_zyx(peak, shape_zyx)
        score = float(smoothed[z, y, x])
        intensity = float(frame_zyx[z, y, x])
        candidates.append(
            DetectionCandidate(
                dataset=dataset,
                t=t,
                z=float(z),
                y=float(y),
                x=float(x),
                score=score,
                volume=None,
                intensity=intensity,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.z, item.y, item.x))
    return candidates
