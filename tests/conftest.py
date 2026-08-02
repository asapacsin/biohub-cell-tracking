from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracker.models import DatasetMetadata


@pytest.fixture
def tiny_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        name="tiny",
        store_path="test/tiny.zarr",
        array_path="0",
        shape=(3, 1, 4, 32, 32),
        axes=("t", "c", "z", "y", "x"),
        dtype="uint16",
        chunks=(1, 1, 2, 16, 16),
        time_points=3,
        channel_count=1,
        voxel_spacing_zyx=(2.0, 0.5, 0.5),
        multiscale_levels=("0",),
    )


def create_test_store(
    root: Path,
    *,
    axes: tuple[str, ...] = ("t", "c", "z", "y", "x"),
) -> np.ndarray:
    zarr = pytest.importorskip("zarr")
    test_root = root / "test"
    test_root.mkdir(parents=True)
    group = zarr.open_group(str(test_root / "tiny.zarr"), mode="w")
    canonical = np.arange(2 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 1, 3, 4, 5)
    canonical_axes = ("t", "c", "z", "y", "x")
    permutation = tuple(canonical_axes.index(axis) for axis in axes)
    data = np.transpose(canonical, permutation)
    chunks = tuple(
        1 if axis in {"t", "c"} else size for axis, size in zip(axes, data.shape, strict=True)
    )
    if hasattr(group, "create_array"):
        group.create_array("0", data=data, chunks=chunks)
    else:
        group.create_dataset("0", data=data, chunks=chunks)
    scale_by_axis = {"t": 1.0, "c": 1.0, "z": 2.0, "y": 0.5, "x": 0.25}
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": axis} for axis in axes],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [scale_by_axis[axis] for axis in axes]}
                    ],
                }
            ],
        }
    ]
    return canonical
