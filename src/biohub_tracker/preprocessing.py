from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.models import DatasetMetadata


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.8
    clip: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("preprocessing percentiles must satisfy 0 <= lower < upper <= 100")


def validate_volume_metadata(metadata: DatasetMetadata) -> None:
    """Validate the invariants required by all train and inference stages."""
    if metadata.time_points < 1:
        raise ValueError(f"Dataset {metadata.name!r} has no time points")
    if any(size < 1 for size in metadata.spatial_shape_zyx):
        raise ValueError(f"Dataset {metadata.name!r} has an empty spatial axis")
    spacing = np.asarray(metadata.voxel_spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Dataset {metadata.name!r} has invalid voxel spacing {spacing}")


def normalize_frame(
    frame_zyx: NDArray[np.generic], config: PreprocessingConfig
) -> NDArray[np.float32]:
    """Robustly normalize one ZYX frame while preserving its native anisotropic grid."""
    frame = np.asarray(frame_zyx, dtype=np.float32)
    if frame.ndim != 3:
        raise ValueError(f"Expected a ZYX frame, got shape {frame.shape}")
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        raise ValueError("Frame contains no finite intensities")
    low, high = np.percentile(finite, (config.lower_percentile, config.upper_percentile)).astype(
        np.float32
    )
    if high <= low:
        return np.zeros_like(frame, dtype=np.float32)
    normalized = (frame - low) / (high - low)
    if config.clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return np.asarray(normalized, dtype=np.float32)
