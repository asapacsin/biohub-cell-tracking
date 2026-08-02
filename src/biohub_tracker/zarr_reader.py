from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.models import ArrayMetadata, DatasetMetadata


def discover_zarr_stores(root: str | Path, split: str) -> list[Path]:
    split_root = Path(root) / split
    if not split_root.is_dir():
        return []
    return sorted(path for path in split_root.rglob("*.zarr") if path.is_dir())


def _plain_attributes(attrs: Any) -> dict[str, Any]:
    if hasattr(attrs, "asdict"):
        return dict(attrs.asdict())
    return dict(attrs)


def _walk_arrays(group: Any, prefix: str = "") -> list[tuple[str, Any]]:
    arrays: list[tuple[str, Any]] = []
    for name in sorted(group.keys()):
        item = group[name]
        path = f"{prefix}/{name}" if prefix else str(name)
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            arrays.append((path, item))
        elif hasattr(item, "keys"):
            arrays.extend(_walk_arrays(item, path))
    return arrays


def _axis_names(multiscale: dict[str, Any], array: Any) -> tuple[str, ...]:
    axes_spec = multiscale.get("axes")
    if axes_spec:
        names = tuple(
            str(axis["name"] if isinstance(axis, dict) else axis).lower() for axis in axes_spec
        )
    else:
        attrs = _plain_attributes(array.attrs)
        dimensions = attrs.get("_ARRAY_DIMENSIONS")
        if not dimensions:
            dimensions = getattr(array, "dimension_names", None)
        if not dimensions or any(value is None for value in dimensions):
            raise ValueError(
                "Axis metadata is absent. Expected OME-NGFF multiscales.axes, "
                "_ARRAY_DIMENSIONS, or Zarr dimension_names; refusing to guess."
            )
        names = tuple(str(axis).lower() for axis in dimensions)
    if len(names) != len(array.shape):
        raise ValueError(f"Axis count {len(names)} does not match array rank {len(array.shape)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Axis metadata contains duplicates: {names}")
    for required in ("z", "y", "x"):
        if required not in names:
            raise ValueError(f"Required spatial axis {required!r} not present in {names}")
    unknown = set(names) - {"t", "c", "z", "y", "x"}
    if unknown:
        raise ValueError(f"Unsupported axes {sorted(unknown)} in {names}")
    return names


def _dataset_transformations(multiscale: dict[str, Any], array_path: str) -> list[dict[str, Any]]:
    transformations: list[dict[str, Any]] = []
    global_transforms = multiscale.get("coordinateTransformations", [])
    if isinstance(global_transforms, list):
        transformations.extend(value for value in global_transforms if isinstance(value, dict))
    for dataset in multiscale.get("datasets", []):
        if str(dataset.get("path", "")) == array_path:
            local = dataset.get("coordinateTransformations", [])
            if isinstance(local, list):
                transformations.extend(value for value in local if isinstance(value, dict))
            break
    return transformations


def _voxel_spacing(
    axes: tuple[str, ...], transformations: list[dict[str, Any]]
) -> tuple[float, float, float]:
    effective_scale = np.ones(len(axes), dtype=np.float64)
    scale_found = False
    for transform in transformations:
        if transform.get("type") != "scale":
            continue
        scale = np.asarray(transform.get("scale", []), dtype=np.float64)
        if scale.shape != effective_scale.shape:
            raise ValueError(
                f"Coordinate scale has {len(scale)} values for {len(axes)} axes: {transform}"
            )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"Coordinate scale must contain positive finite values: {transform}")
        effective_scale *= scale
        scale_found = True
    if not scale_found:
        raise ValueError(
            "No coordinate scale transformation was found; refusing to assume voxel spacing."
        )
    return tuple(float(effective_scale[axes.index(axis)]) for axis in ("z", "y", "x"))


class VolumeDatasetReader:
    """Lazy, metadata-first reader exposing frames in strict ``(z, y, x)`` order."""

    def __init__(self, competition_root: str | Path, channel: int = 0) -> None:
        self.competition_root = Path(competition_root)
        self.channel = channel
        discovered = discover_zarr_stores(self.competition_root, "test")
        self._stores = {path.stem: path for path in discovered}
        if len(self._stores) != len(discover_zarr_stores(self.competition_root, "test")):
            raise ValueError("Test Zarr dataset names must be unique after removing .zarr")
        self._metadata_cache: dict[str, DatasetMetadata] = {}

    def dataset_names(self) -> list[str]:
        return sorted(self._stores)

    def _open(self, dataset: str) -> tuple[Any, Any, str, dict[str, Any]]:
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("Zarr support requires the 'zarr' package") from exc
        try:
            store = self._stores[dataset]
        except KeyError as exc:
            raise KeyError(
                f"Unknown test dataset {dataset!r}; found {self.dataset_names()}"
            ) from exc
        root = zarr.open_group(str(store), mode="r")
        root_attrs = _plain_attributes(root.attrs)
        multiscales = root_attrs.get("multiscales", [])
        multiscale = multiscales[0] if multiscales else {}
        if multiscale and not isinstance(multiscale, dict):
            raise ValueError(f"Invalid multiscales metadata in {store}")
        arrays = _walk_arrays(root)
        if not arrays:
            raise ValueError(f"No arrays found in Zarr store {store}")
        levels = [str(value.get("path")) for value in multiscale.get("datasets", [])]
        array_path = levels[0] if levels else arrays[0][0]
        array_map = dict(arrays)
        if array_path not in array_map:
            raise ValueError(f"Metadata references missing array {array_path!r} in {store}")
        return root, array_map[array_path], array_path, multiscale

    def metadata(self, dataset: str) -> DatasetMetadata:
        if dataset in self._metadata_cache:
            return self._metadata_cache[dataset]
        root, array, array_path, multiscale = self._open(dataset)
        store = self._stores[dataset]
        axes = _axis_names(multiscale, array)
        transformations = _dataset_transformations(multiscale, array_path)
        spacing = _voxel_spacing(axes, transformations)
        arrays = tuple(
            ArrayMetadata(
                path=path,
                shape=tuple(int(value) for value in item.shape),
                dtype=str(item.dtype),
                chunks=(
                    tuple(int(value) for value in item.chunks)
                    if getattr(item, "chunks", None) is not None
                    else None
                ),
            )
            for path, item in _walk_arrays(root)
        )
        shape = tuple(int(value) for value in array.shape)
        sizes = dict(zip(axes, shape, strict=True))
        channel_count = sizes.get("c", 1)
        if not 0 <= self.channel < channel_count:
            raise ValueError(
                f"Requested channel {self.channel} is outside [0, {channel_count}) for {dataset}"
            )
        result = DatasetMetadata(
            name=dataset,
            store_path=str(store),
            array_path=array_path,
            shape=shape,
            axes=axes,
            dtype=str(array.dtype),
            chunks=(
                tuple(int(value) for value in array.chunks)
                if getattr(array, "chunks", None) is not None
                else None
            ),
            time_points=sizes.get("t", 1),
            channel_count=channel_count,
            voxel_spacing_zyx=spacing,
            multiscale_levels=tuple(item.path for item in arrays if item.path in {
                str(value.get("path")) for value in multiscale.get("datasets", [])
            }),
            arrays=arrays,
            coordinate_transformations=tuple(transformations),
            attributes=json.loads(json.dumps(_plain_attributes(root.attrs), default=str)),
        )
        self._metadata_cache[dataset] = result
        return result

    def read_frame(self, dataset: str, t: int) -> NDArray[np.generic]:
        metadata = self.metadata(dataset)
        if not 0 <= t < metadata.time_points:
            raise IndexError(f"Time {t} outside [0, {metadata.time_points}) for {dataset}")
        _, array, _, _ = self._open(dataset)
        selection: list[int | slice] = []
        remaining_axes: list[str] = []
        for axis in metadata.axes:
            if axis == "t":
                selection.append(t)
            elif axis == "c":
                selection.append(self.channel)
            else:
                selection.append(slice(None))
                remaining_axes.append(axis)
        frame = np.asarray(array[tuple(selection)])
        permutation = tuple(remaining_axes.index(axis) for axis in ("z", "y", "x"))
        return np.transpose(frame, permutation)
