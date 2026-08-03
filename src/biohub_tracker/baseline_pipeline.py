from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from biohub_tracker.detection.blob import BlobDetectionConfig
from biohub_tracker.detection.detector import detect_frame_as_detections
from biohub_tracker.models import NodeIdAllocator, PredictedEdge, PredictedNode, PredictionGraph
from biohub_tracker.submission import build_submission, validate_submission, write_submission
from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.division import DivisionConfig
from biohub_tracker.tracking.tracker import track_video_detections
from biohub_tracker.visualization.overlay import save_tracking_overlay
from biohub_tracker.zarr_reader import VolumeDatasetReader

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    detection: BlobDetectionConfig
    max_link_distance: float
    candidate_neighbors: int
    division: DivisionConfig
    voxel_size_zyx: tuple[float, float, float]
    save_masks: bool
    save_visualizations: bool
    save_detections: bool
    save_tracks: bool
    node_id_start: int
    sort_rows: bool
    strict_validation: bool


def load_baseline_config(path: str | Path) -> BaselineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    runtime = raw.get("runtime", {})
    detection_raw = raw.get("detection", {})
    tracking_raw = raw.get("tracking", {})
    submission_raw = raw.get("submission", {})
    detection = BlobDetectionConfig(
        lower_percentile=float(detection_raw.get("lower_percentile", 1.0)),
        upper_percentile=float(detection_raw.get("upper_percentile", 99.8)),
        gaussian_sigma_um=float(detection_raw.get("gaussian_sigma_um", 1.5)),
        threshold=float(detection_raw.get("threshold", 0.08)),
        minimum_separation_um=float(detection_raw.get("minimum_separation_um", 4.0)),
    )
    division = DivisionConfig(
        enabled=bool(tracking_raw.get("division_enabled", False)),
        max_distance=float(tracking_raw.get("division_max_distance", 10.5)),
        max_daughter_separation=float(tracking_raw.get("max_daughter_separation", 15.5)),
        min_daughter_separation=float(tracking_raw.get("min_daughter_separation", 6.0)),
        max_midpoint_distance=float(tracking_raw.get("max_midpoint_distance", 5.5)),
        midpoint_weight=float(tracking_raw.get("midpoint_weight", 2.5)),
        separation_weight=float(tracking_raw.get("separation_weight", 0.25)),
        volume_weight=float(tracking_raw.get("volume_weight", 0.0)),
        max_candidates_per_parent=int(
            tracking_raw.get("max_division_candidates_per_parent", 5)
        ),
        require_matched_daughter=bool(tracking_raw.get("require_matched_daughter", True)),
        max_divisions_per_frame=int(tracking_raw.get("max_divisions_per_frame", 1)),
    )
    return BaselineConfig(
        detection=detection,
        max_link_distance=float(tracking_raw.get("max_link_distance", 15.0)),
        candidate_neighbors=int(tracking_raw.get("candidate_neighbors", 5)),
        division=division,
        voxel_size_zyx=(
            float(raw.get("voxel_size_z", 1.625)),
            float(raw.get("voxel_size_y", 0.40625)),
            float(raw.get("voxel_size_x", 0.40625)),
        ),
        save_masks=bool(runtime.get("save_masks", False)),
        save_visualizations=bool(runtime.get("save_visualizations", True)),
        save_detections=bool(runtime.get("save_detections", True)),
        save_tracks=bool(runtime.get("save_tracks", True)),
        node_id_start=int(submission_raw.get("node_id_start", 1)),
        sort_rows=bool(submission_raw.get("sort_rows", True)),
        strict_validation=bool(submission_raw.get("strict_validation", True)),
    )


def _detections_to_table(detections: list[Detection]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": d.video_id,
                "frame_index": d.frame_index,
                "detection_id": d.detection_id,
                "centroid_z": d.centroid_z,
                "centroid_y": d.centroid_y,
                "centroid_x": d.centroid_x,
                "confidence": d.confidence,
            }
            for d in detections
        ]
    )


def _tracks_to_table(observations: list[TrackObservation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": o.video_id,
                "frame_index": o.frame_index,
                "detection_id": o.detection_id,
                "cell_id": o.cell_id,
                "centroid_z": o.centroid_z,
                "centroid_y": o.centroid_y,
                "centroid_x": o.centroid_x,
                "parent_id": -1 if o.parent_id is None else o.parent_id,
                "link_distance": o.link_distance,
                "link_status": o.link_status,
                "division_score": o.division_score,
                "node_id": o.node_id,
            }
            for o in observations
        ]
    )


def observations_to_graph(
    observations: list[TrackObservation], dataset: str
) -> tuple[PredictionGraph, list[TrackObservation]]:
    """Convert tracks to competition graph.

    Continuations use same ``cell_id`` edges. Divisions emit two edges from the
    parent's last node (looked up via daughter ``parent_id`` = parent ``cell_id``).
    """
    allocator = NodeIdAllocator(next_id=1)
    nodes: list[PredictedNode] = []
    edges: list[PredictedEdge] = []
    last_node_for_cell: dict[int, int] = {}
    stamped: list[TrackObservation] = []
    for obs in sorted(observations, key=lambda item: (item.frame_index, item.detection_id)):
        node_id = allocator.allocate()
        nodes.append(
            PredictedNode(
                dataset=dataset,
                node_id=node_id,
                t=obs.frame_index,
                z=round(obs.centroid_z),
                y=round(obs.centroid_y),
                x=round(obs.centroid_x),
            )
        )
        if obs.link_status == "matched" and obs.cell_id in last_node_for_cell:
            edges.append(
                PredictedEdge(
                    dataset=dataset,
                    source_id=last_node_for_cell[obs.cell_id],
                    target_id=node_id,
                )
            )
        elif (
            obs.link_status == "division_child"
            and obs.parent_id is not None
            and obs.parent_id in last_node_for_cell
        ):
            edges.append(
                PredictedEdge(
                    dataset=dataset,
                    source_id=last_node_for_cell[obs.parent_id],
                    target_id=node_id,
                )
            )
        last_node_for_cell[obs.cell_id] = node_id
        stamped.append(
            TrackObservation(
                video_id=obs.video_id,
                frame_index=obs.frame_index,
                detection_id=obs.detection_id,
                cell_id=obs.cell_id,
                centroid_z=obs.centroid_z,
                centroid_y=obs.centroid_y,
                centroid_x=obs.centroid_x,
                parent_id=obs.parent_id,
                link_distance=obs.link_distance,
                link_status=obs.link_status,
                node_id=node_id,
                division_score=obs.division_score,
            )
        )
    return PredictionGraph(nodes=nodes, edges=edges), stamped


def summarize_video(
    video_id: str,
    n_frames: int,
    detections: list[Detection],
    observations: list[TrackObservation],
    frame_stats: list[dict[str, float | int]],
) -> dict[str, Any]:
    per_frame = Counter(d.frame_index for d in detections)
    counts = [per_frame.get(t, 0) for t in range(n_frames)]
    matched = sum(int(stats["matched_links"]) for stats in frame_stats)
    new_dets = sum(int(stats["new_detections"]) for stats in frame_stats)
    ended = sum(int(stats["ended_tracks"]) for stats in frame_stats)
    divisions = sum(int(stats.get("divisions_accepted", 0)) for stats in frame_stats)
    link_distances = [
        float(obs.link_distance)
        for obs in observations
        if obs.link_distance is not None
    ]
    if link_distances:
        arr = np.asarray(link_distances, dtype=np.float64)
        mean_d, p95_d, max_d = float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())
    else:
        mean_d = p95_d = max_d = 0.0
    track_ids = {obs.cell_id for obs in observations}
    summary = {
        "video_id": video_id,
        "frames": n_frames,
        "total_detections": len(detections),
        "detections_per_frame_min": int(min(counts) if counts else 0),
        "detections_per_frame_mean": float(np.mean(counts) if counts else 0.0),
        "detections_per_frame_max": int(max(counts) if counts else 0),
        "n_tracks": len(track_ids),
        "matched_links": matched,
        "new_unmatched_detections": new_dets,
        "ended_tracks": ended,
        "divisions_accepted": divisions,
        "division_children": sum(
            1 for obs in observations if obs.link_status == "division_child"
        ),
        "mean_link_distance": mean_d,
        "p95_link_distance": p95_d,
        "max_link_distance": max_d,
        "zero_detection_frames": [t for t, c in enumerate(counts) if c == 0],
    }
    if summary["zero_detection_frames"]:
        LOGGER.warning(
            "%s: frames with zero detections: %s",
            video_id,
            summary["zero_detection_frames"],
        )
    if n_frames and new_dets / max(len(detections), 1) > 0.5:
        LOGGER.warning("%s: >50%% detections are new IDs (possible under-linking)", video_id)
    if link_distances and p95_d > 0.9 * max(mean_d, 1e-6) * 3:
        pass
    return summary


def process_video(
    reader: VolumeDatasetReader,
    video_id: str,
    config: BaselineConfig,
    output_dir: Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    save_visualizations: bool | None = None,
) -> tuple[PredictionGraph, dict[str, Any]]:
    metadata = reader.metadata(video_id)
    spacing = metadata.voxel_spacing_zyx
    LOGGER.info(
        "%s: raw shape=%s axes=%s frames=%s channels=%s voxel_spacing_zyx=%s",
        video_id,
        metadata.shape,
        metadata.axes,
        metadata.time_points,
        metadata.channel_count,
        spacing,
    )
    if spacing != config.voxel_size_zyx:
        LOGGER.info(
            "%s: using store metadata spacing %s (config default was %s)",
            video_id,
            spacing,
            config.voxel_size_zyx,
        )

    t0 = 0 if start_frame is None else max(0, start_frame)
    t1 = metadata.time_points if end_frame is None else min(metadata.time_points, end_frame)
    frames = list(range(t0, t1))

    detections_by_frame: list[list[Detection]] = []
    all_detections: list[Detection] = []
    for t in frames:
        frame = reader.read_frame(video_id, t)
        dets = detect_frame_as_detections(
            frame,
            video_id=video_id,
            frame_index=t,
            voxel_spacing_zyx=spacing,
            config=config.detection,
        )
        detections_by_frame.append(dets)
        all_detections.extend(dets)

    observations, frame_stats = track_video_detections(
        detections_by_frame,
        max_link_distance=config.max_link_distance,
        voxel_spacing_zyx=spacing,
        candidate_neighbors=config.candidate_neighbors,
        cell_id_start=0,
        division=config.division,
    )
    graph, stamped = observations_to_graph(observations, video_id)
    summary = summarize_video(video_id, len(frames), all_detections, stamped, frame_stats)
    LOGGER.info("summary %s", json.dumps(summary, sort_keys=True))

    video_out = output_dir
    if config.save_detections:
        path = video_out / "detections" / f"{video_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        _detections_to_table(all_detections).to_csv(path, index=False)
    if config.save_tracks:
        path = video_out / "tracks" / f"{video_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        _tracks_to_table(stamped).to_csv(path, index=False)

    do_viz = config.save_visualizations if save_visualizations is None else save_visualizations
    if do_viz and frames:
        viz_dir = video_out / "visualizations" / video_id
        picks = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
        by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
        for obs in stamped:
            by_frame[obs.frame_index].append(obs)
        for t in picks:
            prev_map = {
                obs.cell_id: obs
                for obs in by_frame.get(t - 1, [])
            }
            save_tracking_overlay(
                reader.read_frame(video_id, t),
                by_frame.get(t, []),
                prev_map,
                viz_dir / f"frame_{t:04d}.png",
            )
            LOGGER.info("%s: wrote visualization frame %s (method=MIP)", video_id, t)

    (video_out / "diagnostics").mkdir(parents=True, exist_ok=True)
    (video_out / "diagnostics" / f"{video_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return graph, summary


def run_public_test_baseline(
    competition_root: str | Path,
    output_dir: str | Path,
    config: BaselineConfig,
    *,
    video_ids: list[str] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    save_visualizations: bool | None = None,
    overwrite: bool = True,
) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True")
    output.mkdir(parents=True, exist_ok=True)

    reader = VolumeDatasetReader(competition_root, split="test")
    names = video_ids or reader.dataset_names()
    graphs: list[PredictionGraph] = []
    summaries: list[dict[str, Any]] = []
    for video_id in names:
        graph, summary = process_video(
            reader,
            video_id,
            config,
            output,
            start_frame=start_frame,
            end_frame=end_frame,
            save_visualizations=save_visualizations,
        )
        graphs.append(graph)
        summaries.append(summary)

    submission = build_submission(graphs, sort_rows=config.sort_rows)
    submission_path = write_submission(submission, output / "submission" / "submission.csv")
    if config.strict_validation:
        validate_submission(submission, competition_root)

    report = {
        "parameters": {
            "detection_threshold": config.detection.threshold,
            "gaussian_sigma_um": config.detection.gaussian_sigma_um,
            "minimum_separation_um": config.detection.minimum_separation_um,
            "max_link_distance_um": config.max_link_distance,
            "candidate_neighbors": config.candidate_neighbors,
            "voxel_size_zyx_default": config.voxel_size_zyx,
            "division_enabled": config.division.enabled,
            "division_max_distance_um": config.division.max_distance,
            "max_daughter_separation_um": config.division.max_daughter_separation,
            "max_midpoint_distance_um": config.division.max_midpoint_distance,
        },
        "videos": summaries,
        "submission_path": str(submission_path),
        "known_limitations": [
            "Division is a greedy scored heuristic (no mitosis classifier).",
            "No gap closing across missing frames.",
            "Greedy nearest-neighbour matching, not Hungarian.",
            "Blob peak detector; no mask-based size filtering yet.",
            "Public test volumes are Zarr images; GEFF graphs exist only for train.",
        ],
    }
    (output / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return submission_path
