from __future__ import annotations

import numpy as np

from biohub_tracker.association import (
    AssociationScoringConfig,
    CandidateGraphConfig,
    OptimizerConfig,
    build_candidate_graph,
    optimize_candidate_graph,
    score_candidate_graph,
)
from biohub_tracker.detection import (
    DetectionDecoderConfig,
    SlidingWindowHeatmapPredictor,
    decode_heatmap,
)
from biohub_tracker.models import DetectionCandidate
from biohub_tracker.training import generate_centroid_heatmap, label_candidate_graph


def test_physical_gaussian_target_is_anisotropic_in_voxel_space() -> None:
    target = generate_centroid_heatmap(
        (9, 17, 17),
        [(4.0, 8.0, 8.0)],
        voxel_spacing_zyx=(2.0, 0.5, 0.5),
        sigma_um=2.0,
    )
    assert target.dtype == np.float32
    assert target[4, 8, 8] == 1.0
    # Four Y voxels and one Z voxel are both two micrometres from the centroid.
    assert np.isclose(target[5, 8, 8], target[4, 12, 8])


def test_decoder_applies_nms_and_subvoxel_refinement() -> None:
    heatmap = np.zeros((5, 9, 9), dtype=np.float32)
    heatmap[2, 4, 4] = 1.0
    heatmap[2, 4, 5] = 0.5
    heatmap[2, 7, 7] = 0.8
    decoded = decode_heatmap(
        heatmap,
        dataset="sample",
        t=3,
        voxel_spacing_zyx=(2.0, 1.0, 1.0),
        config=DetectionDecoderConfig(
            threshold=0.4,
            adaptive_quantile=0.0,
            nms_radius_um=1.0,
            refinement_radius_voxels=1,
        ),
    )
    assert len(decoded) == 2
    assert decoded[0].score == 1.0
    assert 4.0 < decoded[0].x < 5.0


def test_candidate_labels_and_optimizer_keep_atomic_division() -> None:
    detections = [
        DetectionCandidate("sample", 0, 2.0, 5.0, 5.0, 1.0),
        DetectionCandidate("sample", 1, 2.0, 4.0, 5.0, 1.0),
        DetectionCandidate("sample", 1, 2.0, 6.0, 5.0, 1.0),
    ]
    graph = build_candidate_graph(
        detections,
        voxel_spacing_zyx=(1.0, 1.0, 1.0),
        config=CandidateGraphConfig(
            max_neighbors=3,
            max_gap=1,
            max_speed_um_per_frame=5.0,
            max_daughter_separation_um=5.0,
        ),
    )
    labels = label_candidate_graph(
        graph,
        detection_to_geff_node={0: 10, 1: 11, 2: 12},
        geff_edges={(10, 11), (10, 12)},
    )
    assert labels.edge_labels.tolist() == [1, 1]
    assert labels.division_labels.tolist() == [1]

    score_candidate_graph(
        graph,
        AssociationScoringConfig(link_bias=0.0, division_bias=20.0),
    )
    selected = optimize_candidate_graph(graph, OptimizerConfig(method="ilp"))
    assert set(selected) == {(0, 1), (0, 2)}


def test_sliding_window_predictor_reassembles_full_frame() -> None:
    class IdentityPredictor:
        def predict(self, frame_zyx: np.ndarray) -> np.ndarray:
            return frame_zyx

    frame = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    predictor = SlidingWindowHeatmapPredictor(
        IdentityPredictor(), window_shape_zyx=(4, 4, 4), overlap=0.5
    )
    assert np.array_equal(predictor.predict(frame), frame)
