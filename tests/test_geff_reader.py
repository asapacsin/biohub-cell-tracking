from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biohub_tracker.annotation_reader import discover_annotation_files, inspect_geff


def _write_tiny_geff(path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["geff"] = {
        "geff_version": "1.1",
        "directed": True,
        "extra": {"estimated_number_of_nodes": 10},
    }
    nodes = root.create_group("nodes")
    nodes.create_array("ids", data=np.asarray([1, 2, 3], dtype=np.uint64))
    props = nodes.create_group("props")
    for name, values in {
        "t": [0, 1, 2],
        "z": [1, 2, 3],
        "y": [10, 11, 12],
        "x": [20, 21, 22],
    }.items():
        prop = props.create_group(name)
        prop.create_array("values", data=np.asarray(values, dtype=np.int64))
    edges = root.create_group("edges")
    edges.create_array("ids", data=np.asarray([[1, 2], [2, 3]], dtype=np.uint64))


def test_discover_and_inspect_geff(tmp_path: Path) -> None:
    geff = tmp_path / "train" / "demo.geff"
    _write_tiny_geff(geff)
    # Nested JSON inside the store must not be treated as annotations.
    (geff / "nodes" / "nested.json").write_text("{}", encoding="utf-8")
    found = discover_annotation_files(tmp_path)
    assert found == [geff]
    report = inspect_geff(geff)
    assert report.format == "geff"
    assert report.row_count == 5
    assert report.dtypes["estimated_number_of_nodes"] == "10"
    assert report.sample_rows[0]["node_id"] == 1
    assert report.sample_rows[-1]["target_id"] == 3
