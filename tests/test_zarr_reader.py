from __future__ import annotations

import pytest
from conftest import create_test_store

from biohub_tracker.zarr_reader import VolumeDatasetReader


@pytest.mark.parametrize(
    "axes",
    [
        ("t", "c", "z", "y", "x"),
        ("t", "z", "y", "x", "c"),
        ("c", "t", "x", "z", "y"),
    ],
)
def test_reader_uses_metadata_to_return_zyx(tmp_path, axes) -> None:
    canonical = create_test_store(tmp_path, axes=axes)
    reader = VolumeDatasetReader(tmp_path)
    metadata = reader.metadata("tiny")
    assert metadata.axes == axes
    assert metadata.voxel_spacing_zyx == (2.0, 0.5, 0.25)
    assert metadata.time_points == 2
    assert metadata.channel_count == 1
    assert reader.read_frame("tiny", 1).shape == (3, 4, 5)
    assert (reader.read_frame("tiny", 1) == canonical[1, 0]).all()


def test_reader_refuses_to_guess_missing_axes(tmp_path) -> None:
    zarr = pytest.importorskip("zarr")
    target = tmp_path / "test" / "bad.zarr"
    target.parent.mkdir(parents=True)
    group = zarr.open_group(str(target), mode="w")
    if hasattr(group, "create_array"):
        group.create_array("0", shape=(2, 3, 4, 5), dtype="uint8")
    else:
        group.create_dataset("0", shape=(2, 3, 4, 5), dtype="uint8")
    with pytest.raises(ValueError, match="refusing to guess"):
        VolumeDatasetReader(tmp_path).metadata("bad")


def test_reader_is_lazy_until_read_frame(tmp_path, monkeypatch) -> None:
    create_test_store(tmp_path)
    reader = VolumeDatasetReader(tmp_path)
    metadata = reader.metadata("tiny")
    assert metadata.shape == (2, 1, 3, 4, 5)
