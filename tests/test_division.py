from __future__ import annotations

from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.division import DivisionConfig
from biohub_tracker.tracking.nearest_neighbor import link_consecutive_frames

SPACING = (1.0, 1.0, 1.0)

DEFAULT_DIVISION = DivisionConfig(
    enabled=True,
    max_distance=5.0,
    max_daughter_separation=5.0,
    min_daughter_separation=0.5,
    max_midpoint_distance=2.0,
    midpoint_weight=1.0,
    separation_weight=0.5,
    volume_weight=0.0,
    max_candidates_per_parent=20,
    require_matched_daughter=True,
    max_divisions_per_frame=5,
)


def _det(frame: int, det_id: int, z: float, y: float, x: float) -> Detection:
    return Detection("v", frame, det_id, z, y, x, confidence=1.0)


def _obs(frame: int, det_id: int, cell_id: int, z: float, y: float, x: float) -> TrackObservation:
    return TrackObservation(
        "v", frame, det_id, cell_id, z, y, x, parent_id=None, link_status="initial"
    )


def test_obvious_division() -> None:
    prev = [_obs(0, 0, 5, 0.0, 0.0, 0.0)]
    curr = [_det(1, 0, 0.0, -1.0, 0.0), _det(1, 1, 0.0, 1.0, 0.0)]
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=6,
        max_link_distance=5.0,
        voxel_spacing_zyx=SPACING,
        division=DEFAULT_DIVISION,
    )
    assert stats["divisions_accepted"] == 1
    assert {obs.cell_id for obs in linked} != {5}
    assert all(obs.cell_id != 5 for obs in linked)
    assert len({obs.cell_id for obs in linked}) == 2
    assert all(obs.parent_id == 5 for obs in linked)
    assert all(obs.link_status == "division_child" for obs in linked)
    assert all(obs.division_score is not None for obs in linked)
    assert all(obs.link_distance is not None for obs in linked)


def test_continuation_plus_unrelated_new_cell() -> None:
    prev = [_obs(0, 0, 0, 0.0, 0.0, 0.0)]
    curr = [_det(1, 0, 0.0, 1.0, 0.0), _det(1, 1, 0.0, 20.0, 20.0)]
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=1,
        max_link_distance=5.0,
        voxel_spacing_zyx=SPACING,
        division=DEFAULT_DIVISION,
    )
    by_det = {obs.detection_id: obs for obs in linked}
    assert by_det[0].cell_id == 0
    assert by_det[0].link_status == "matched"
    assert by_det[0].parent_id is None
    assert by_det[1].link_status == "new"
    assert by_det[1].parent_id is None
    assert stats["divisions_accepted"] == 0


def test_matched_plus_nearby_unmatched_becomes_division() -> None:
    prev = [_obs(0, 0, 3, 0.0, 0.0, 0.0)]
    # D1 is closer so first-pass matches P -> D1; D2 stays unmatched but nearby.
    curr = [_det(1, 0, 0.0, 0.5, 0.0), _det(1, 1, 0.0, -1.0, 0.0)]
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=4,
        max_link_distance=5.0,
        voxel_spacing_zyx=SPACING,
        division=DEFAULT_DIVISION,
    )
    assert stats["divisions_accepted"] == 1
    assert stats["continuations_revoked"] == 1
    assert all(obs.link_status == "division_child" for obs in linked)
    assert all(obs.parent_id == 3 for obs in linked)
    assert all(obs.cell_id != 3 for obs in linked)
    assert linked[0].cell_id != linked[1].cell_id


def test_two_parents_cannot_share_one_daughter() -> None:
    # Symmetric daughters around P0; P1 is farther from the midpoint.
    prev = [
        _obs(0, 0, 10, 0.0, 0.0, 0.0),
        _obs(0, 1, 11, 0.0, 4.0, 0.0),
    ]
    curr = [
        _det(1, 0, 0.0, -1.0, 0.0),
        _det(1, 1, 0.0, 1.0, 0.0),
    ]
    loose = DivisionConfig(
        enabled=True,
        max_distance=5.0,
        max_daughter_separation=3.0,
        min_daughter_separation=0.5,
        max_midpoint_distance=5.0,
        midpoint_weight=1.0,
        separation_weight=0.5,
        volume_weight=0.0,
        max_candidates_per_parent=20,
        require_matched_daughter=False,
        max_divisions_per_frame=5,
    )
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=12,
        max_link_distance=0.25,  # no first-pass matches; both daughters free
        voxel_spacing_zyx=SPACING,
        division=loose,
    )
    assert stats["divisions_accepted"] == 1
    parents = {obs.parent_id for obs in linked if obs.link_status == "division_child"}
    assert parents == {10}  # closer parent has the lower score
    daughter_ids = [obs.cell_id for obs in linked if obs.link_status == "division_child"]
    assert len(daughter_ids) == 2
    assert len(set(daughter_ids)) == 2
    # Second parent must not also claim either daughter.
    assert all(obs.parent_id != 11 for obs in linked)


def test_daughters_too_far_apart_rejected() -> None:
    prev = [_obs(0, 0, 1, 0.0, 0.0, 0.0)]
    curr = [_det(1, 0, 0.0, -2.0, 0.0), _det(1, 1, 0.0, 2.0, 0.0)]
    tight = DivisionConfig(
        enabled=True,
        max_distance=5.0,
        max_daughter_separation=3.0,  # separation is 4.0
        min_daughter_separation=0.5,
        max_midpoint_distance=2.0,
        midpoint_weight=1.0,
        separation_weight=0.5,
        volume_weight=0.0,
        max_candidates_per_parent=20,
        require_matched_daughter=True,
        max_divisions_per_frame=5,
    )
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=2,
        max_link_distance=5.0,
        voxel_spacing_zyx=SPACING,
        division=tight,
    )
    assert stats["divisions_accepted"] == 0
    assert all(obs.parent_id is None for obs in linked)
    assert all(obs.link_status != "division_child" for obs in linked)


def test_unmatched_parent_two_unmatched_daughters() -> None:
    prev = [_obs(0, 0, 7, 0.0, 0.0, 0.0)]
    curr = [_det(1, 0, 0.0, -1.0, 0.0), _det(1, 1, 0.0, 1.0, 0.0)]
    # First-pass cannot match (threshold below daughter distance 1.0).
    division = DivisionConfig(
        enabled=True,
        max_distance=5.0,
        max_daughter_separation=5.0,
        min_daughter_separation=0.5,
        max_midpoint_distance=2.0,
        midpoint_weight=1.0,
        separation_weight=0.5,
        volume_weight=0.0,
        max_candidates_per_parent=20,
        require_matched_daughter=False,
        max_divisions_per_frame=5,
    )
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=8,
        max_link_distance=0.5,
        voxel_spacing_zyx=SPACING,
        division=division,
    )
    assert stats["divisions_accepted"] == 1
    assert all(obs.link_status == "division_child" for obs in linked)
    assert all(obs.parent_id == 7 for obs in linked)
    assert all(obs.cell_id != 7 for obs in linked)


def test_division_disabled_preserves_baseline() -> None:
    prev = [_obs(0, 0, 0, 0.0, 0.0, 0.0)]
    curr = [_det(1, 0, 0.0, -1.0, 0.0), _det(1, 1, 0.0, 1.0, 0.0)]
    linked, _, stats = link_consecutive_frames(
        prev,
        curr,
        next_cell_id=1,
        max_link_distance=10.0,
        voxel_spacing_zyx=SPACING,
        division=DivisionConfig(enabled=False),
    )
    statuses = {obs.link_status for obs in linked}
    assert statuses == {"matched", "new"}
    matched = next(obs for obs in linked if obs.link_status == "matched")
    newborn = next(obs for obs in linked if obs.link_status == "new")
    assert matched.cell_id == 0
    assert matched.parent_id is None
    assert newborn.parent_id is None
    assert newborn.cell_id == 1
    assert stats["divisions_accepted"] == 0
