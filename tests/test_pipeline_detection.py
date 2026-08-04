from __future__ import annotations

from pathlib import Path

import numpy as np

from biohub_tracker.config import load_config
from biohub_tracker.fixtures import generate_tiny_competition
from biohub_tracker.pipeline import run_prediction_pipeline


def test_pipeline_detects_nodes_on_tiny_fixture(tmp_path: Path) -> None:
    root = generate_tiny_competition(tmp_path / "competition")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
detection:
  method: blob
  lower_percentile: 0.0
  upper_percentile: 100.0
  gaussian_sigma_um: 0.5
  threshold: 0.05
  minimum_separation_um: 1.0
submission:
  node_id_start: 1
  sort_rows: true
  strict_validation: false
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    graphs = run_prediction_pipeline(root, config)
    assert len(graphs) == 1
    assert len(graphs[0].nodes) >= 1
    assert all(node.dataset == "tiny" for node in graphs[0].nodes)
    assert {node.t for node in graphs[0].nodes} <= {0, 1, 2}
    # Tiny fixture may or may not produce links depending on detections; edges must be valid ids.
    node_ids = {node.node_id for node in graphs[0].nodes}
    for edge in graphs[0].edges:
        assert edge.source_id in node_ids
        assert edge.target_id in node_ids


def test_pipeline_accepts_injected_learned_heatmap_predictor(tmp_path: Path) -> None:
    class IdentityHeatmapPredictor:
        def predict(self, frame_zyx: np.ndarray) -> np.ndarray:
            return frame_zyx.astype(np.float32, copy=False)

    root = generate_tiny_competition(tmp_path / "competition")
    config_path = tmp_path / "learned.yaml"
    config_path.write_text(
        """
preprocessing:
  lower_percentile: 0
  upper_percentile: 100
detection:
  method: learned
  tta_flips: false
  decoder:
    threshold: 0.5
    adaptive_quantile: 0.9
    nms_radius_um: 1.0
    refinement_radius_voxels: 0
association:
  candidate_graph:
    max_gap: 1
    max_speed_um_per_frame: 10
    divisions_enabled: true
  optimizer:
    method: ilp
submission:
  strict_validation: false
""",
        encoding="utf-8",
    )
    graphs = run_prediction_pipeline(
        root, load_config(config_path), predictors=[IdentityHeatmapPredictor()]
    )
    assert len(graphs) == 1
    assert len(graphs[0].nodes) == 7
    assert all(edge.source_id != edge.target_id for edge in graphs[0].edges)
