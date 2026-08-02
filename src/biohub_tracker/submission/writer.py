from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biohub_tracker.models import PredictionGraph

SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
NUMERIC_COLUMNS = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_submission(
    graphs: list[PredictionGraph],
    *,
    sort_rows: bool = True,
) -> pd.DataFrame:
    node_rows = [
        {
            "dataset": node.dataset,
            "row_type": "node",
            "node_id": node.node_id,
            "t": node.t,
            "z": node.z,
            "y": node.y,
            "x": node.x,
            "source_id": -1,
            "target_id": -1,
        }
        for graph in graphs
        for node in graph.nodes
    ]
    edge_rows = [
        {
            "dataset": edge.dataset,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
        }
        for graph in graphs
        for edge in graph.edges
    ]
    if sort_rows:
        node_rows.sort(key=lambda row: (row["dataset"], row["t"], row["node_id"]))
        edge_rows.sort(key=lambda row: (row["dataset"], row["source_id"], row["target_id"]))
    table = pd.DataFrame(
        node_rows + edge_rows,
        columns=[
            "dataset",
            "row_type",
            "node_id",
            "t",
            "z",
            "y",
            "x",
            "source_id",
            "target_id",
        ],
    )
    for column in NUMERIC_COLUMNS[1:]:
        table[column] = table[column].astype(np.int64)
    table["dataset"] = table["dataset"].astype(str)
    table["row_type"] = table["row_type"].astype(str)
    table.insert(0, "id", np.arange(len(table), dtype=np.int64))
    return table[SUBMISSION_COLUMNS]


def write_submission(table: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination

