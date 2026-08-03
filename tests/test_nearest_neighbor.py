from __future__ import annotations

from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.nearest_neighbor import link_consecutive_frames

SPACING = (1.0, 1.0, 1.0)


def _det(frame: int, det_id: int, z: float, y: float, x: float) -> Detection:
    return Detection("v", frame, det_id, z, y, x, confidence=1.0)


def _obs(frame: int, det_id: int, cell_id: int, z: float, y: float, x: float) -> TrackObservation:
    return TrackObservation(
        "v", frame, det_id, cell_id, z, y, x, parent_id=None, link_status="initial"
    )


def test_stable_movement_keeps_ids() -> None:
    prev = [_obs(0, 0, 0, 0, 0, 0), _obs(0, 1, 1, 0, 10, 10)]
    curr = [_det(1, 0, 0, 1, 0), _det(1, 1, 0, 11, 10)]
    linked, next_id, _ = link_consecutive_frames(
        prev, curr, next_cell_id=2, max_link_distance=5.0, voxel_spacing_zyx=SPACING
    )
    by_det = {obs.detection_id: obs.cell_id for obs in linked}
    assert by_det[0] == 0
    assert by_det[1] == 1
    assert next_id == 2


def test_new_detection_gets_new_id() -> None:
    prev = [_obs(0, 0, 0, 0, 0, 0)]
    curr = [_det(1, 0, 0, 1, 0), _det(1, 1, 0, 20, 20)]
    linked, next_id, _ = link_consecutive_frames(
        prev, curr, next_cell_id=1, max_link_distance=5.0, voxel_spacing_zyx=SPACING
    )
    by_det = {obs.detection_id: obs.cell_id for obs in linked}
    assert by_det[0] == 0
    assert by_det[1] == 1
    assert next_id == 2


def test_ended_track() -> None:
    prev = [_obs(0, 0, 0, 0, 0, 0), _obs(0, 1, 1, 0, 20, 20)]
    curr = [_det(1, 0, 0, 1, 0)]
    linked, _, stats = link_consecutive_frames(
        prev, curr, next_cell_id=2, max_link_distance=5.0, voxel_spacing_zyx=SPACING
    )
    assert len(linked) == 1
    assert linked[0].cell_id == 0
    assert stats["ended_tracks"] == 1


def test_matching_conflict_prefers_shorter() -> None:
    prev = [_obs(0, 0, 0, 0, 0, 0), _obs(0, 1, 1, 0, 3, 0)]
    curr = [_det(1, 0, 0, 1, 0)]
    linked, _, _ = link_consecutive_frames(
        prev, curr, next_cell_id=2, max_link_distance=10.0, voxel_spacing_zyx=SPACING
    )
    assert len(linked) == 1
    assert linked[0].cell_id == 0


def test_one_previous_two_current_no_division() -> None:
    prev = [_obs(0, 0, 0, 0, 0, 0)]
    curr = [_det(1, 0, 0, 1, 0), _det(1, 1, 0, 2, 0)]
    linked, next_id, _ = link_consecutive_frames(
        prev, curr, next_cell_id=1, max_link_distance=10.0, voxel_spacing_zyx=SPACING
    )
    by_det = {obs.detection_id: obs for obs in linked}
    assert by_det[0].cell_id == 0
    assert by_det[1].cell_id == 1
    assert by_det[1].parent_id is None
    assert next_id == 2
