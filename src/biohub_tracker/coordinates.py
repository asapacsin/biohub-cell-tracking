from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def voxel_to_physical_zyx(
    position_zyx: Sequence[float],
    voxel_spacing_zyx: Sequence[float],
) -> NDArray[np.float64]:
    position = np.asarray(position_zyx, dtype=np.float64)
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    if position.shape != (3,) or spacing.shape != (3,):
        raise ValueError("position and spacing must each contain exactly (z, y, x)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(spacing)):
        raise ValueError("position and spacing must be finite")
    if np.any(spacing <= 0):
        raise ValueError("voxel spacing must be positive")
    return position * spacing


def physical_distance_zyx(
    first_zyx: Sequence[float],
    second_zyx: Sequence[float],
    voxel_spacing_zyx: Sequence[float],
) -> float:
    first = voxel_to_physical_zyx(first_zyx, voxel_spacing_zyx)
    second = voxel_to_physical_zyx(second_zyx, voxel_spacing_zyx)
    return float(np.linalg.norm(first - second))


def round_and_clip_voxel_zyx(
    position_zyx: Sequence[float],
    shape_zyx: Sequence[int],
) -> tuple[int, int, int]:
    position = np.asarray(position_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    if position.shape != (3,) or shape.shape != (3,):
        raise ValueError("position and shape must each contain exactly (z, y, x)")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must be finite")
    if np.any(shape <= 0):
        raise ValueError("spatial shape must be positive")
    rounded = np.rint(position).astype(np.int64)
    clipped = np.clip(rounded, 0, shape - 1)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def permutation_to_zyx(axes: Sequence[str]) -> tuple[int, int, int]:
    normalized = tuple(axis.lower() for axis in axes)
    missing = [axis for axis in ("z", "y", "x") if axis not in normalized]
    if missing:
        raise ValueError(f"Missing spatial axes: {missing}; found {normalized}")
    return tuple(normalized.index(axis) for axis in ("z", "y", "x"))  # type: ignore[return-value]
