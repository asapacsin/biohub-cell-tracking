from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.division import DivisionConfig, apply_divisions


def _physical_points(
    centroids_zyx: Sequence[tuple[float, float, float]],
    voxel_spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    points = np.asarray(centroids_zyx, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return points * spacing


def link_consecutive_frames(
    previous_observations: Sequence[TrackObservation],
    current_detections: Sequence[Detection],
    *,
    next_cell_id: int,
    max_link_distance: float,
    voxel_spacing_zyx: tuple[float, float, float],
    candidate_neighbors: int = 5,
    division: DivisionConfig | None = None,
) -> tuple[list[TrackObservation], int, dict[str, float | int]]:
    """Greedy one-to-one nearest-neighbour linking, optional division post-pass."""
    if not current_detections:
        stats = {
            "matched_links": 0,
            "new_detections": 0,
            "ended_tracks": len(previous_observations),
            "mean_link_distance": 0.0,
            "p95_link_distance": 0.0,
            "max_link_distance_obs": 0.0,
            "divisions_accepted": 0,
            "division_candidates": 0,
            "continuations_revoked": 0,
        }
        return [], next_cell_id, stats

    if not previous_observations:
        initial = [
            TrackObservation(
                video_id=detection.video_id,
                frame_index=detection.frame_index,
                detection_id=detection.detection_id,
                cell_id=next_cell_id + index,
                centroid_z=detection.centroid_z,
                centroid_y=detection.centroid_y,
                centroid_x=detection.centroid_x,
                parent_id=None,
                link_distance=None,
                link_status="initial",
            )
            for index, detection in enumerate(current_detections)
        ]
        stats = {
            "matched_links": 0,
            "new_detections": len(initial),
            "ended_tracks": 0,
            "mean_link_distance": 0.0,
            "p95_link_distance": 0.0,
            "max_link_distance_obs": 0.0,
            "divisions_accepted": 0,
            "division_candidates": 0,
            "continuations_revoked": 0,
        }
        return initial, next_cell_id + len(initial), stats

    previous_points = _physical_points(
        [obs.centroid_zyx for obs in previous_observations], voxel_spacing_zyx
    )
    current_points = _physical_points(
        [det.centroid_zyx for det in current_detections], voxel_spacing_zyx
    )
    tree = cKDTree(current_points)
    k = min(max(1, candidate_neighbors), len(current_detections))
    distances_list, indices_list = tree.query(
        previous_points, k=k, distance_upper_bound=max_link_distance
    )
    if k == 1:
        distances_list = np.asarray(distances_list)[:, None]
        indices_list = np.asarray(indices_list)[:, None]

    candidates: list[tuple[float, int, int]] = []
    for prev_index, (dists, idxs) in enumerate(zip(distances_list, indices_list, strict=True)):
        for distance, curr_index in zip(dists, idxs, strict=True):
            if not np.isfinite(distance) or curr_index >= len(current_detections):
                continue
            candidates.append((float(distance), prev_index, int(curr_index)))
    candidates.sort(key=lambda item: item[0])

    used_previous: set[int] = set()
    used_current: set[int] = set()
    current_cell_ids: dict[int, int] = {}
    link_distances: dict[int, float] = {}
    accepted_distances: list[float] = []

    for distance, prev_index, curr_index in candidates:
        if prev_index in used_previous or curr_index in used_current:
            continue
        cell_id = previous_observations[prev_index].cell_id
        current_cell_ids[curr_index] = cell_id
        link_distances[curr_index] = distance
        accepted_distances.append(distance)
        used_previous.add(prev_index)
        used_current.add(curr_index)

    linked: list[TrackObservation] = []
    for curr_index, detection in enumerate(current_detections):
        if curr_index in current_cell_ids:
            linked.append(
                TrackObservation(
                    video_id=detection.video_id,
                    frame_index=detection.frame_index,
                    detection_id=detection.detection_id,
                    cell_id=current_cell_ids[curr_index],
                    centroid_z=detection.centroid_z,
                    centroid_y=detection.centroid_y,
                    centroid_x=detection.centroid_x,
                    parent_id=None,
                    link_distance=link_distances[curr_index],
                    link_status="matched",
                )
            )
        else:
            linked.append(
                TrackObservation(
                    video_id=detection.video_id,
                    frame_index=detection.frame_index,
                    detection_id=detection.detection_id,
                    cell_id=next_cell_id,
                    centroid_z=detection.centroid_z,
                    centroid_y=detection.centroid_y,
                    centroid_x=detection.centroid_x,
                    parent_id=None,
                    link_distance=None,
                    link_status="new",
                )
            )
            next_cell_id += 1

    if accepted_distances:
        arr = np.asarray(accepted_distances, dtype=np.float64)
        mean_d = float(arr.mean())
        p95_d = float(np.percentile(arr, 95))
        max_d = float(arr.max())
    else:
        mean_d = p95_d = max_d = 0.0

    stats: dict[str, float | int] = {
        "matched_links": len(accepted_distances),
        "new_detections": len(current_detections) - len(accepted_distances),
        "ended_tracks": len(previous_observations) - len(accepted_distances),
        "mean_link_distance": mean_d,
        "p95_link_distance": p95_d,
        "max_link_distance_obs": max_d,
        "divisions_accepted": 0,
        "division_candidates": 0,
        "continuations_revoked": 0,
    }

    division_config = division if division is not None else DivisionConfig()
    if division_config.enabled:
        linked, next_cell_id, div_stats = apply_divisions(
            previous_observations,
            current_detections,
            linked,
            next_cell_id=next_cell_id,
            config=division_config,
            voxel_spacing_zyx=voxel_spacing_zyx,
        )
        divisions = int(div_stats["divisions_accepted"])
        revoked = int(div_stats["continuations_revoked"])
        stats["divisions_accepted"] = divisions
        stats["division_candidates"] = int(div_stats["division_candidates"])
        stats["continuations_revoked"] = revoked
        # Revoked continuations end the parent; unmatched parents were already ended.
        stats["matched_links"] = int(stats["matched_links"]) - revoked
        stats["ended_tracks"] = int(stats["ended_tracks"]) + revoked
        stats["new_detections"] = sum(1 for obs in linked if obs.link_status == "new")

    return linked, next_cell_id, stats
