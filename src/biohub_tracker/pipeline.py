from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from biohub_tracker.association import (
    CandidateGraphScorer,
    HandcraftedAssociationScorer,
    LinearAssociationScorer,
    build_candidate_graph,
    optimize_candidate_graph,
)
from biohub_tracker.config import ProjectConfig
from biohub_tracker.coordinates import round_and_clip_voxel_zyx
from biohub_tracker.detection import (
    HeatmapPredictor,
    SlidingWindowHeatmapPredictor,
    TorchScriptHeatmapPredictor,
    decode_heatmap,
    detect_frame,
    predict_ensemble_heatmap,
)
from biohub_tracker.models import (
    DetectionCandidate,
    NodeIdAllocator,
    PredictedEdge,
    PredictedNode,
    PredictionGraph,
)
from biohub_tracker.postprocessing import postprocess_prediction_graph
from biohub_tracker.preprocessing import normalize_frame, validate_volume_metadata
from biohub_tracker.zarr_reader import VolumeDatasetReader


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_predictors(config: ProjectConfig) -> list[HeatmapPredictor]:
    if not config.learned_detection.model_paths:
        raise ValueError(
            "detection.method=learned requires at least one path in detection.model_paths"
        )
    return [
        SlidingWindowHeatmapPredictor(
            TorchScriptHeatmapPredictor(path, device=config.learned_detection.device),
            window_shape_zyx=config.learned_detection.window_shape_zyx,
            overlap=config.learned_detection.window_overlap,
        )
        for path in config.learned_detection.model_paths
    ]


def _load_scorer(config: ProjectConfig) -> CandidateGraphScorer:
    if config.association_scorer_kind == "handcrafted":
        return HandcraftedAssociationScorer(config.scoring)
    if config.association_model_path is None:
        raise ValueError(
            "association.scorer.kind=learned_linear requires association.scorer.model_path"
        )
    return LinearAssociationScorer.load(config.association_model_path)


def _detect_video(
    reader: VolumeDatasetReader,
    dataset: str,
    config: ProjectConfig,
    predictors: Sequence[HeatmapPredictor] | None,
) -> list[DetectionCandidate]:
    metadata = reader.metadata(dataset)
    validate_volume_metadata(metadata)
    learned_predictors = predictors
    if config.detection_method == "learned" and learned_predictors is None:
        learned_predictors = _load_predictors(config)

    detections: list[DetectionCandidate] = []
    for t in range(metadata.time_points):
        if t == 0 or (t + 1) % 10 == 0 or t + 1 == metadata.time_points:
            _log(f"  detect {dataset} t={t + 1}/{metadata.time_points}")
        frame = reader.read_frame(dataset, t)
        if config.detection_method == "blob":
            detections.extend(
                detect_frame(
                    frame,
                    dataset=dataset,
                    t=t,
                    voxel_spacing_zyx=metadata.voxel_spacing_zyx,
                    config=config.detection,
                )
            )
            continue
        normalized = normalize_frame(frame, config.preprocessing)
        heatmap = predict_ensemble_heatmap(
            normalized,
            learned_predictors or [],
            tta_flips=config.learned_detection.tta_flips,
        )
        detections.extend(
            decode_heatmap(
                heatmap,
                dataset=dataset,
                t=t,
                voxel_spacing_zyx=metadata.voxel_spacing_zyx,
                config=config.decoder,
            )
        )
    return detections


def run_prediction_pipeline(
    competition_root: str | Path,
    config: ProjectConfig,
    *,
    predictors: Sequence[HeatmapPredictor] | None = None,
    scorer: CandidateGraphScorer | None = None,
) -> list[PredictionGraph]:
    """Run detection -> sparse candidates -> scoring -> global optimization -> cleanup."""
    reader = VolumeDatasetReader(competition_root, split="test")
    active_scorer = scorer or _load_scorer(config)
    graphs: list[PredictionGraph] = []
    datasets = list(reader.dataset_names())
    for dataset_index, dataset in enumerate(datasets, start=1):
        _log(f"[{dataset_index}/{len(datasets)}] dataset={dataset}")
        metadata = reader.metadata(dataset)
        detections = _detect_video(reader, dataset, config, predictors)
        _log(f"  detections={len(detections)}")
        candidates = build_candidate_graph(
            detections,
            voxel_spacing_zyx=metadata.voxel_spacing_zyx,
            config=config.candidate_graph,
        )
        _log(
            f"  candidates nodes={len(candidates.nodes)} "
            f"events={len(candidates.edges)}"
        )
        active_scorer.score(candidates)
        event_count = len(candidates.edges) + len(candidates.divisions)
        if (
            config.optimizer.method == "ilp"
            and event_count > config.optimizer.ilp_event_limit
        ):
            _log(
                f"  optimize via greedy fallback "
                f"(events={event_count} > ilp_event_limit={config.optimizer.ilp_event_limit})"
            )
        selected = optimize_candidate_graph(candidates, config.optimizer)
        _log(f"  selected_edges={len(selected)}")

        allocator = NodeIdAllocator(next_id=config.submission.node_id_start)
        nodes: list[PredictedNode] = []
        node_id_by_index: dict[int, int] = {}
        for index, detection in enumerate(candidates.nodes):
            node_id = allocator.allocate()
            node_id_by_index[index] = node_id
            z, y, x = round_and_clip_voxel_zyx(
                (detection.z, detection.y, detection.x), metadata.spatial_shape_zyx
            )
            nodes.append(
                PredictedNode(dataset=dataset, node_id=node_id, t=detection.t, z=z, y=y, x=x)
            )
        edges = [
            PredictedEdge(
                dataset=dataset,
                source_id=node_id_by_index[source],
                target_id=node_id_by_index[target],
            )
            for source, target in selected
        ]
        graph = postprocess_prediction_graph(
            PredictionGraph(nodes=nodes, edges=edges), config.postprocessing
        )
        _log(f"  postprocess nodes={len(graph.nodes)} edges={len(graph.edges)}")
        graphs.append(graph)
    return graphs
