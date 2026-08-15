"""Offline z-coordinate shift of frozen detections (no re-inference).

Missed GT cells on 6bba_07e24132 / 6bba_fc83837d sit 7–10 µm from a free peak,
almost entirely as a negative z offset. Shifting predicted node z tests whether
a one-voxel convention fix recovers them without a detector retrain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import evaluate_tables


def shift_node_z(predictions: pd.DataFrame, delta_voxels: float) -> pd.DataFrame:
    out = predictions.copy()
    nodes = out["row_type"] == "node"
    out.loc[nodes, "z"] = out.loc[nodes, "z"].astype(float) + float(delta_voxels)
    return out


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    data_dir: Path,
    datasets: list[str],
    *,
    estimated_total_nodes: dict[str, float],
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_distance_um: float = 7.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    found = sorted(predictions["dataset"].unique().tolist())
    if found != sorted(datasets):
        raise RuntimeError(f"prediction datasets {found} != expected {sorted(datasets)}")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        part = predictions[predictions["dataset"] == dataset]
        pred_nodes = part[part["row_type"] == "node"].loc[:, ["node_id", "t", "z", "y", "x"]]
        pred_edges = part[part["row_type"] == "edge"].loc[:, ["source_id", "target_id"]]
        gt = read_geff_tables(data_dir / f"{dataset}.geff")
        rows.append(
            evaluate_tables(
                dataset,
                pred_nodes,
                pred_edges,
                gt.nodes.loc[:, ["node_id", "t", "z", "y", "x"]],
                gt.edges.loc[:, ["source_id", "target_id"]],
                float(estimated_total_nodes[dataset]),
                scale=scale,
                max_distance_um=max_distance_um,
            )
        )
    return rows, {**official_spec_summarise(rows), "datasets": list(datasets)}
