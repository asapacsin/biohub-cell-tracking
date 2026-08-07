"""Pure submission conversion and integrity checks for clean V106."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from biohub_pipeline.config import PipelineConfig

CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _edge_key(edge: dict[str, Any]) -> tuple[int, int]:
    return int(edge["source_id"]), int(edge["target_id"])


def _infer_edge_source(edge: dict[str, Any], *, dt: int) -> str:
    """Best-effort label for how an edge likely entered the graph (diagnostic only)."""
    if int(edge.get("safe_division", 0) or 0) == 1:
        return "safe_division"
    if int(edge.get("gap_closed", 0) or 0) == 1 or int(edge.get("gap_synthetic", 0) or 0) == 1:
        return "gap_recovery"
    if int(edge.get("motion_relink", 0) or 0) == 1:
        return "motion_relink"
    if dt == 1:
        return "ordinary_or_division_association"
    if dt > 1:
        return "nonconsecutive_gap_like"
    return "unknown"


def collect_outdegree_violations(
    dataset: str,
    nodes: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Return structured diagnostics for parents with more than two children."""
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pair_counts: Counter[tuple[int, int]] = Counter()
    for edge in edges:
        key = _edge_key(edge)
        pair_counts[key] += 1
        by_source[key[0]].append(edge)

    violations: list[dict[str, Any]] = []
    for parent_id, parent_edges in sorted(by_source.items()):
        if len(parent_edges) <= 2:
            continue
        parent = nodes.get(parent_id, {})
        children: list[dict[str, Any]] = []
        child_ids = [int(edge["target_id"]) for edge in parent_edges]
        for edge in parent_edges:
            child_id = int(edge["target_id"])
            child = nodes.get(child_id, {})
            parent_t = int(parent.get("t", -1)) if parent else -1
            child_t = int(child.get("t", -1)) if child else -1
            dt = child_t - parent_t if parent_t >= 0 and child_t >= 0 else -999
            children.append(
                {
                    "child_id": child_id,
                    "child_t": child_t,
                    "source_id": int(edge["source_id"]),
                    "target_id": child_id,
                    "edge_prob": edge.get("edge_prob"),
                    "distance_um": edge.get("distance_um"),
                    "score": edge.get("score"),
                    "cost": edge.get("cost"),
                    "edge_type": edge.get("edge_type"),
                    "duplicate_edge_count": int(pair_counts[_edge_key(edge)]),
                    "inferred_source": _infer_edge_source(edge, dt=dt),
                    "flags": {
                        key: edge.get(key)
                        for key in (
                            "gap_closed",
                            "gap_synthetic",
                            "safe_division",
                            "motion_relink",
                        )
                        if key in edge
                    },
                }
            )
        violations.append(
            {
                "dataset": dataset,
                "stage": stage,
                "parent_id": parent_id,
                "parent_t": int(parent.get("t", -1)) if parent else -1,
                "n_children": len(parent_edges),
                "child_ids": child_ids,
                "duplicate_child_ids": sorted(
                    {cid for cid, count in Counter(child_ids).items() if count > 1}
                ),
                "children": children,
            }
        )
    return violations


def format_outdegree_violation_report(violations: list[dict[str, Any]]) -> str:
    if not violations:
        return "No parents with more than two children."
    lines = [
        f"OUTDEGREE_VIOLATIONS total_parents={len(violations)} "
        f"max_children={max(int(v['n_children']) for v in violations)}"
    ]
    per_dataset = Counter(str(v["dataset"]) for v in violations)
    lines.append(
        "per_dataset: "
        + ", ".join(f"{dataset}={count}" for dataset, count in sorted(per_dataset.items()))
    )
    for violation in violations:
        lines.append(
            f"dataset={violation['dataset']} stage={violation['stage']} "
            f"parent={violation['parent_id']} t={violation['parent_t']} "
            f"n_children={violation['n_children']} "
            f"duplicate_child_ids={violation['duplicate_child_ids']}"
        )
        for child in violation["children"]:
            lines.append(
                "  child="
                f"{child['child_id']} t={child['child_t']} "
                f"edge={child['source_id']}->{child['target_id']} "
                f"prob={child['edge_prob']} dist_um={child['distance_um']} "
                f"score={child['score']} cost={child['cost']} "
                f"type={child['edge_type']} dup={child['duplicate_edge_count']} "
                f"source={child['inferred_source']} flags={child['flags']}"
            )
    return "\n".join(lines)


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
        report = format_outdegree_violation_report(
            collect_outdegree_violations(dataset, nodes, edges, stage="validate_graph")
        )
        raise ValueError("a node has more than two children\n" + report)


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
    pre_violations: list[dict[str, Any]] = []
    post_violations: list[dict[str, Any]] = []
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
        pre_violations.extend(
            collect_outdegree_violations(dataset, nodes, edges, stage="raw_geff_before_postprocess")
        )
        nodes, edges, _ = postprocessing.filter_output_graph(nodes, edges, dataset=dataset)
        post_here = collect_outdegree_violations(
            dataset, nodes, edges, stage="after_postprocess"
        )
        post_violations.extend(post_here)
        if post_here:
            print(format_outdegree_violation_report(post_here))
        # Persist diagnostics before validate_graph can abort conversion.
        if pre_violations or post_violations:
            diagnostics = {
                "raw_before_postprocess_parent_count": len(pre_violations),
                "after_postprocess_parent_count": len(post_violations),
                "raw_before_postprocess": pre_violations,
                "after_postprocess": post_violations,
                "appeared_only_after_postprocess": (
                    len(pre_violations) == 0 and len(post_violations) > 0
                ),
            }
            print(
                "OUTDEGREE_STAGE_SUMMARY "
                + json.dumps(
                    {
                        "raw_before_postprocess_parent_count": diagnostics[
                            "raw_before_postprocess_parent_count"
                        ],
                        "after_postprocess_parent_count": diagnostics[
                            "after_postprocess_parent_count"
                        ],
                        "appeared_only_after_postprocess": diagnostics[
                            "appeared_only_after_postprocess"
                        ],
                    },
                    sort_keys=True,
                )
            )
            diagnostics_path = Path(output).parent / "outdegree_violations.json"
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote outdegree diagnostics: {diagnostics_path}")
        all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))
    write_rows(all_rows, output)
    return validate_submission_file(output)
