from __future__ import annotations

from collections.abc import Sequence

from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.division import DivisionConfig
from biohub_tracker.tracking.nearest_neighbor import link_consecutive_frames


def track_video_detections(
    detections_by_frame: Sequence[Sequence[Detection]],
    *,
    max_link_distance: float,
    voxel_spacing_zyx: tuple[float, float, float],
    candidate_neighbors: int = 5,
    cell_id_start: int = 0,
    division: DivisionConfig | None = None,
) -> tuple[list[TrackObservation], list[dict[str, float | int]]]:
    """Track per-frame detections with greedy one-to-one linking and optional division."""
    observations: list[TrackObservation] = []
    frame_stats: list[dict[str, float | int]] = []
    previous: list[TrackObservation] = []
    next_cell_id = cell_id_start
    for frame_detections in detections_by_frame:
        linked, next_cell_id, stats = link_consecutive_frames(
            previous,
            frame_detections,
            next_cell_id=next_cell_id,
            max_link_distance=max_link_distance,
            voxel_spacing_zyx=voxel_spacing_zyx,
            candidate_neighbors=candidate_neighbors,
            division=division,
        )
        observations.extend(linked)
        frame_stats.append(stats)
        previous = linked
    return observations, frame_stats
