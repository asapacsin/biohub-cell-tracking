from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def generate_centroid_heatmap(
    shape_zyx: tuple[int, int, int],
    centroids_zyx: Sequence[Sequence[float]],
    *,
    voxel_spacing_zyx: tuple[float, float, float],
    sigma_um: float = 2.0,
    truncate: float = 3.0,
) -> NDArray[np.float32]:
    """Render max-composited anisotropic Gaussian centroid targets in physical units."""
    if any(size <= 0 for size in shape_zyx):
        raise ValueError("shape_zyx must contain positive sizes")
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("voxel_spacing_zyx must contain three positive values")
    if sigma_um <= 0 or truncate <= 0:
        raise ValueError("sigma_um and truncate must be positive")
    heatmap = np.zeros(shape_zyx, dtype=np.float32)
    radius = np.maximum(1, np.ceil(truncate * sigma_um / spacing).astype(np.int64))
    for centroid_values in centroids_zyx:
        centroid = np.asarray(centroid_values, dtype=np.float64)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError("each centroid must contain three finite ZYX values")
        lower = np.maximum(0, np.floor(centroid - radius).astype(np.int64))
        upper = np.minimum(shape_zyx, np.ceil(centroid + radius + 1).astype(np.int64))
        if np.any(lower >= upper):
            continue
        coordinates = np.meshgrid(
            *(np.arange(lower[axis], upper[axis]) for axis in range(3)), indexing="ij"
        )
        squared_physical_distance = sum(
            ((coordinates[axis] - centroid[axis]) * spacing[axis]) ** 2 for axis in range(3)
        )
        gaussian = np.exp(-0.5 * squared_physical_distance / sigma_um**2).astype(np.float32)
        slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
        heatmap[slices] = np.maximum(heatmap[slices], gaussian)
    return heatmap
