"""Pure submission conversion and integrity checks for clean V106."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from biohub_pipeline.config import PipelineConfig

CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def validate_graph(
    dataset: str, nodes: dict[int, dict[str, Any]], edges: list[dict[str, Any]]
) -> None:
    if not dataset:
        raise ValueError("dataset/video ID must not be empty")
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    for node_id, node in nodes.items():
        if int(node["node_id"]) != int(node_id):
            raise ValueError(f"node dictionary key does not match node_id: {node_id}")
        if int(node["t"]) < 0:
            raise ValueError(f"negative frame index for node {node_id}")
        for axis in ("z", "y", "x"):
            if float(node[axis]) < 0:
                raise ValueError(f"negative {axis} coordinate for node {node_id}")
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if source not in nodes or target not in nodes:
            raise ValueError(f"dangling edge {source}->{target}")
        if int(nodes[target]["t"]) != int(nodes[source]["t"]) + 1:
            raise ValueError(f"edge {source}->{target} is not between consecutive frames")
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
    if any(count > 1 for count in incoming.values()):
        raise ValueError("a node has more than one parent")
    if any(count > 2 for count in outgoing.values()):
        raise ValueError("a node has more than two children")


def graph_rows(
    dataset: str,
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    start_id: int = 0,
) -> list[dict[str, Any]]:
    # Match upstream V106: clamp after round before integrity checks / CSV write.
    clamped: dict[int, dict[str, Any]] = {}
    for node_id, node in nodes.items():
        clamped[node_id] = {
            **node,
            "z": max(0, int(round(float(node["z"])))),
            "y": max(0, int(round(float(node["y"])))),
            "x": max(0, int(round(float(node["x"])))),
        }
    validate_graph(dataset, clamped, edges)
    rows: list[dict[str, Any]] = []
    row_id = start_id
    for node_id in sorted(clamped):
        node = clamped[node_id]
        rows.append(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node_id),
                "t": int(node["t"]),
                "z": int(node["z"]),
                "y": int(node["y"]),
                "x": int(node["x"]),
                "source_id": -1,
                "target_id": -1,
            }
        )
        row_id += 1
    for edge in edges:
        rows.append(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(edge["source_id"]),
                "target_id": int(edge["target_id"]),
            }
        )
        row_id += 1
    return rows


def write_rows(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(materialized)
    return destination


def validate_submission_file(path: str | Path) -> dict[str, int]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"submission columns must be {CSV_COLUMNS}; got {reader.fieldnames}")
        rows = list(reader)
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("submission row IDs must be contiguous from zero")
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        part = [row for row in rows if row["dataset"] == dataset]
        node_rows = [row for row in part if row["row_type"] == "node"]
        edge_rows = [row for row in part if row["row_type"] == "edge"]
        nodes = {
            int(row["node_id"]): {
                "node_id": int(row["node_id"]),
                "t": int(row["t"]),
                "z": int(row["z"]),
                "y": int(row["y"]),
                "x": int(row["x"]),
            }
            for row in node_rows
        }
        edges = [
            {"source_id": int(row["source_id"]), "target_id": int(row["target_id"])}
            for row in edge_rows
        ]
        validate_graph(dataset, nodes, edges)
    return {
        "rows": len(rows),
        "datasets": len(datasets),
        "nodes": sum(r["row_type"] == "node" for r in rows),
        "edges": sum(r["row_type"] == "edge" for r in rows),
    }


def write_submission_from_geff(
    geff_paths: list[Path],
    config: PipelineConfig,
    test_dir: Path,
    output: Path,
) -> dict[str, int]:
    import tracksdata as td

    from biohub_pipeline import postprocessing

    postprocessing.configure(config.postprocessing, test_dir)
    all_rows: list[dict[str, Any]] = []
    for geff_path in sorted(geff_paths):
        dataset = geff_path.stem
        loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = loaded[0] if isinstance(loaded, tuple) else loaded
        nodes: dict[int, dict[str, Any]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            nodes[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
        edges: list[dict[str, Any]] = []
        for row in graph.edge_attrs().iter_rows(named=True):
            probability = row.get("edge_prob") if hasattr(row, "get") else None
            edges.append(
                {
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_prob": None if probability is None else float(probability),
                }
            )
        nodes, edges, _ = postprocessing.filter_output_graph(nodes, edges, dataset=dataset)
        all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))
    write_rows(all_rows, output)
    return validate_submission_file(output)
