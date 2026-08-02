import numpy as np
import pytest

from biohub_tracker.coordinates import (
    physical_distance_zyx,
    round_and_clip_voxel_zyx,
    voxel_to_physical_zyx,
)


def test_coordinates_remain_zyx() -> None:
    result = voxel_to_physical_zyx((2, 3, 5), (10, 2, 1))
    np.testing.assert_array_equal(result, np.array([20.0, 6.0, 5.0]))


def test_physical_distance_respects_anisotropy() -> None:
    assert physical_distance_zyx((0, 0, 0), (1, 2, 2), (4, 1, 1)) == pytest.approx(np.sqrt(24))


def test_round_and_clip_uses_zyx_shape() -> None:
    assert round_and_clip_voxel_zyx((9.8, -2.0, 100.0), (11, 20, 30)) == (10, 0, 29)


def test_coordinate_helpers_reject_bad_shapes() -> None:
    with pytest.raises(ValueError, match="exactly"):
        voxel_to_physical_zyx((1, 2), (1, 1, 1))
