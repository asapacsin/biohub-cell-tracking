from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from biohub_tracker.tracking.data_types import Detection, TrackObservation


def _physical_points(
    centroids_zyx: Sequence[tuple[float, float, float]],
    voxel_spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    spacing = np.asarray(voxel_spacing_zyx, dtype=np.float64)
    points = np.asarray(centroids_zyx, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return points * spacing


@dataclass(frozen=True, slots=True)
class DivisionConfig:
    """Post-pass cell-division heuristics. Disabled by default to preserve baseline.

    Distance knobs are in physical µm. Defaults follow train-GEFF p95 (+slack).
    """

    enabled: bool = False
    max_distance: float = 10.5
    max_daughter_separation: float = 15.5
    min_daughter_separation: float = 6.0
    max_midpoint_distance: float = 5.5
    midpoint_weight: float = 2.5
    separation_weight: float = 0.25
    volume_weight: float = 0.0
    max_candidates_per_parent: int = 5
    # Prefer converting a 1-1 continuation + nearby unmatched daughter (usual mitosis).
    require_matched_daughter: bool = True
    # Train GEFF has at most 1 labeled division per frame-transition.
    max_divisions_per_frame: int = 1


def _detection_volume(detection: Detection) -> float | None:
    if detection.mask is None:
        return None
    return float(np.asarray(detection.mask).sum())


def apply_divisions(
    previous_observations: Sequence[TrackObservation],
    current_detections: Sequence[Detection],
    linked: Sequence[TrackObservation],
    *,
    next_cell_id: int,
    config: DivisionConfig,
    voxel_spacing_zyx: tuple[float, float, float],
    parent_volumes: Sequence[float | None] | None = None,
) -> tuple[list[TrackObservation], int, dict[str, float | int]]:
    """Greedily accept non-conflicting divisions after one-to-one linking."""
    result = list(linked)
    empty_stats: dict[str, float | int] = {
        "divisions_accepted": 0,
        "division_candidates": 0,
        "continuations_revoked": 0,
    }
    if (
        not config.enabled
        or not previous_observations
        or not current_detections
        or len(result) != len(current_detections)
    ):
        return result, next_cell_id, empty_stats

    previous_points = _physical_points(
        [obs.centroid_zyx for obs in previous_observations], voxel_spacing_zyx
    )
    current_points = _physical_points(
        [det.centroid_zyx for det in current_detections], voxel_spacing_zyx
    )

    matched_curr_for_prev: dict[int, int] = {}
    matched_prev_for_curr: dict[int, int] = {}
    for curr_index, obs in enumerate(result):
        if obs.link_status != "matched":
            continue
        for prev_index, prev in enumerate(previous_observations):
            if prev.cell_id == obs.cell_id:
                matched_curr_for_prev[prev_index] = curr_index
                matched_prev_for_curr[curr_index] = prev_index
                break

    volumes = [_detection_volume(detection) for detection in current_detections]
    if parent_volumes is None:
        parent_vols: list[float | None] = [None] * len(previous_observations)
    else:
        parent_vols = list(parent_volumes)
        if len(parent_vols) != len(previous_observations):
            raise ValueError("parent_volumes length must match previous_observations")

    per_parent_candidates: list[tuple[float, int, int, int, float, float]] = []
    for prev_index, _prev in enumerate(previous_observations):
        matched_curr = matched_curr_for_prev.get(prev_index)
        if config.require_matched_daughter and matched_curr is None:
            continue

        daughter_indices: list[int] = []
        if matched_curr is not None:
            daughter_indices.append(matched_curr)
        for curr_index, obs in enumerate(result):
            owner = matched_prev_for_curr.get(curr_index)
            if owner is not None:
                # Do not steal a detection matched to a different parent.
                continue
            if obs.link_status == "matched":
                continue
            distance = float(
                np.linalg.norm(previous_points[prev_index] - current_points[curr_index])
            )
            if distance <= config.max_distance:
                daughter_indices.append(curr_index)

        seen: set[int] = set()
        unique_daughters: list[int] = []
        for curr_index in daughter_indices:
            if curr_index in seen:
                continue
            seen.add(curr_index)
            unique_daughters.append(curr_index)

        parent_pairs: list[tuple[float, int, int, int, float, float]] = []
        for d1, d2 in combinations(unique_daughters, 2):
            if config.require_matched_daughter and matched_curr not in (d1, d2):
                continue
            dist_p_d1 = float(np.linalg.norm(previous_points[prev_index] - current_points[d1]))
            dist_p_d2 = float(np.linalg.norm(previous_points[prev_index] - current_points[d2]))
            if dist_p_d1 > config.max_distance or dist_p_d2 > config.max_distance:
                continue
            separation = float(np.linalg.norm(current_points[d1] - current_points[d2]))
            if separation > config.max_daughter_separation:
                continue
            if separation < config.min_daughter_separation:
                continue
            midpoint = 0.5 * (current_points[d1] + current_points[d2])
            mid_dist = float(np.linalg.norm(previous_points[prev_index] - midpoint))
            if mid_dist > config.max_midpoint_distance:
                continue
            # Parent should sit near the daughter midpoint (symmetric split).
            if separation > 0 and mid_dist > 0.55 * separation:
                continue

            score = (
                dist_p_d1
                + dist_p_d2
                + config.midpoint_weight * mid_dist
                + config.separation_weight * separation
            )
            if config.volume_weight != 0.0:
                v1, v2, vp = volumes[d1], volumes[d2], parent_vols[prev_index]
                if v1 is not None and v2 is not None and vp is not None and vp > 0:
                    score += config.volume_weight * abs((v1 + v2) / vp - 1.0)

            parent_pairs.append((score, prev_index, d1, d2, dist_p_d1, dist_p_d2))

        parent_pairs.sort(key=lambda item: item[0])
        per_parent_candidates.extend(parent_pairs[: max(0, config.max_candidates_per_parent)])

    candidates = sorted(per_parent_candidates, key=lambda item: item[0])
    used_parents: set[int] = set()
    used_daughters: set[int] = set()
    accepted: list[tuple[float, int, int, int, float, float]] = []
    for candidate in candidates:
        if len(accepted) >= max(0, config.max_divisions_per_frame):
            break
        score, prev_index, d1, d2, dist_p_d1, dist_p_d2 = candidate
        if prev_index in used_parents:
            continue
        if d1 in used_daughters or d2 in used_daughters:
            continue
        for daughter in (d1, d2):
            owner = matched_prev_for_curr.get(daughter)
            if owner is not None and owner != prev_index:
                break
        else:
            used_parents.add(prev_index)
            used_daughters.add(d1)
            used_daughters.add(d2)
            accepted.append(candidate)

    continuations_revoked = 0
    for score, prev_index, d1, d2, dist_p_d1, dist_p_d2 in accepted:
        if prev_index in matched_curr_for_prev:
            continuations_revoked += 1
        parent_cell_id = previous_observations[prev_index].cell_id
        for curr_index, distance in ((d1, dist_p_d1), (d2, dist_p_d2)):
            detection = current_detections[curr_index]
            result[curr_index] = TrackObservation(
                video_id=detection.video_id,
                frame_index=detection.frame_index,
                detection_id=detection.detection_id,
                cell_id=next_cell_id,
                centroid_z=detection.centroid_z,
                centroid_y=detection.centroid_y,
                centroid_x=detection.centroid_x,
                parent_id=parent_cell_id,
                link_distance=distance,
                link_status="division_child",
                node_id=result[curr_index].node_id,
                division_score=score,
            )
            next_cell_id += 1

    stats: dict[str, float | int] = {
        "divisions_accepted": len(accepted),
        "division_candidates": len(candidates),
        "continuations_revoked": continuations_revoked,
    }
    return result, next_cell_id, stats
