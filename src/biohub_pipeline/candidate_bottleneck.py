"""Causal analysis of captured pre-gate edge scores through final postprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biohub_pipeline.association_density import (
    GraphTables,
    add_bins,
    read_geff_tables,
)
from biohub_pipeline.evaluation import _node_match

CAUSE_ORDER = (
    "correct",
    "detection_miss",
    "scorer_ranking",
    "candidate_threshold",
    "ilp_global",
    "postprocessing_removed",
    "postprocessing_rematch",
)


def _load_capture_nodes(path: Path) -> pd.DataFrame:
    with np.load(path / "nodes.npz") as data:
        return pd.DataFrame(
            {
                "node_id": data["node_id"].astype(np.int64),
                "t": data["t"].astype(np.int64),
                "z": data["z"].astype(float),
                "y": data["y"].astype(float),
                "x": data["x"].astype(float),
            }
        )


def _capture_metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _read_relevant_candidates(
    path: Path,
    relevant_pairs: set[tuple[int, int]],
    relevant_targets: set[int],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    pair_records: dict[tuple[int, int], dict[str, Any]] = {}
    best_by_target: dict[int, dict[str, Any]] = {}
    fields = (
        "source_id",
        "target_id",
        "blended_logit",
        "blended_prob",
        "seed1_logit",
        "seed2_logit",
        "seed1_prob",
        "seed2_prob",
        "source_rank",
        "target_rank",
        "above_threshold",
    )
    for file in sorted(path.glob("t*_to_t*.npz")):
        with np.load(file) as data:
            sources = data["source_id"].astype(np.int64)
            targets = data["target_id"].astype(np.int64)
            wanted = np.fromiter(
                (
                    (int(source), int(target)) in relevant_pairs
                    or (int(target) in relevant_targets and int(rank) == 1)
                    for source, target, rank in zip(
                        sources, targets, data["target_rank"], strict=True
                    )
                ),
                dtype=bool,
                count=len(sources),
            )
            for index in np.flatnonzero(wanted):
                record = {field: data[field][index].item() for field in fields}
                key = (int(record["source_id"]), int(record["target_id"]))
                if key in relevant_pairs:
                    pair_records[key] = record
                if int(record["target_rank"]) == 1:
                    best_by_target[int(record["target_id"])] = record
    return pair_records, best_by_target


def _selected_edge_set(graph: GraphTables) -> set[tuple[int, int]]:
    selected = graph.edges[graph.edges["selected"]]
    return {(int(row.source_id), int(row.target_id)) for row in selected.itertuples(index=False)}


def _final_tables(predictions: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = predictions[predictions["dataset"] == dataset]
    nodes = part[part["row_type"] == "node"].loc[:, ["node_id", "t", "z", "y", "x"]]
    edges = part[part["row_type"] == "edge"].loc[:, ["source_id", "target_id"]]
    nodes = nodes.astype({"node_id": "int64", "t": "int64"})
    edges = edges.astype({"source_id": "int64", "target_id": "int64"})
    return nodes, edges


def _mapped_edge_set(edges: pd.DataFrame, p2g: dict[int, int]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for row in edges.itertuples(index=False):
        source = p2g.get(int(row.source_id))
        target = p2g.get(int(row.target_id))
        if source is not None and target is not None:
            result.add((source, target))
    return result


def _density(
    nodes: pd.DataFrame,
    gt_target: pd.Series,
    scale: np.ndarray,
    radius_um: float,
) -> int:
    next_frame = nodes[nodes["t"] == int(gt_target["t"])]
    if next_frame.empty:
        return 0
    distances = np.linalg.norm(
        (next_frame[["z", "y", "x"]].to_numpy(float) - gt_target[["z", "y", "x"]].to_numpy(float))
        * scale,
        axis=1,
    )
    return int((distances <= radius_um).sum())


def analyze_dataset(
    dataset: str,
    capture_dir: Path,
    ilp_geff: Path,
    ground_truth: GraphTables,
    final_predictions: pd.DataFrame,
    *,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_match_um: float = 7.0,
    density_radius_um: float = 15.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    detected = _load_capture_nodes(capture_dir)
    metadata = _capture_metadata(capture_dir)
    top_k = int(metadata["top_k"])
    gt_nodes = ground_truth.nodes.loc[:, ["node_id", "t", "z", "y", "x"]]
    gt_edges = ground_truth.edges.loc[:, ["source_id", "target_id"]].drop_duplicates()
    _, detected_g2p, _ = _node_match(detected, gt_nodes, scale=scale, max_distance_um=max_match_um)
    ilp_edges = _selected_edge_set(read_geff_tables(ilp_geff))
    final_nodes, final_edges = _final_tables(final_predictions, dataset)
    final_p2g, final_g2p, _ = _node_match(
        final_nodes, gt_nodes, scale=scale, max_distance_um=max_match_um
    )
    final_gt_edges = _mapped_edge_set(final_edges, final_p2g)
    final_edge_ids = {
        (int(row.source_id), int(row.target_id)) for row in final_edges.itertuples(index=False)
    }

    outdegree = gt_edges.groupby("source_id").size()
    ordinary = gt_edges[gt_edges["source_id"].map(outdegree).eq(1)].copy()
    matched_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for row in ordinary.itertuples(index=False):
        gs, gt = int(row.source_id), int(row.target_id)
        if gs in detected_g2p and gt in detected_g2p:
            matched_pairs[(gs, gt)] = (int(detected_g2p[gs]), int(detected_g2p[gt]))
    relevant_pairs = set(matched_pairs.values())
    relevant_targets = {target for _, target in relevant_pairs}
    candidate_records, best_by_target = _read_relevant_candidates(
        capture_dir, relevant_pairs, relevant_targets
    )

    scale_array = np.asarray(scale, dtype=float)
    gt_by_id = gt_nodes.set_index("node_id")
    rows: list[dict[str, Any]] = []
    for edge in ordinary.itertuples(index=False):
        gs, gt = int(edge.source_id), int(edge.target_id)
        source_detected = gs in detected_g2p
        target_detected = gt in detected_g2p
        ps = int(detected_g2p[gs]) if source_detected else None
        pt = int(detected_g2p[gt]) if target_detected else None
        record = candidate_records.get((ps, pt)) if ps is not None and pt is not None else None
        best = best_by_target.get(pt) if pt is not None else None
        final_correct = (gs, gt) in final_gt_edges
        candidate_present = bool(record and record["above_threshold"])
        ilp_selected = bool(ps is not None and pt is not None and (ps, pt) in ilp_edges)
        final_pair_present = bool(ps is not None and pt is not None and (ps, pt) in final_edge_ids)
        target_rank = int(record["target_rank"]) if record else top_k + 1
        source_rank = int(record["source_rank"]) if record else top_k + 1

        if not source_detected or not target_detected:
            cause = "detection_miss"
        elif final_correct:
            cause = "correct"
        elif target_rank > 1:
            cause = "scorer_ranking"
        elif not candidate_present:
            cause = "candidate_threshold"
        elif not ilp_selected:
            cause = "ilp_global"
        elif not final_pair_present:
            cause = "postprocessing_removed"
        else:
            cause = "postprocessing_rematch"

        gt_prob = float(record["blended_prob"]) if record else np.nan
        best_prob = float(best["blended_prob"]) if best else np.nan
        rows.append(
            {
                "dataset": dataset,
                "dataset_family": dataset.split("_")[0],
                "t": int(gt_by_id.loc[gs, "t"]),
                "gt_source_id": gs,
                "gt_target_id": gt,
                "pred_source_id": ps,
                "pred_target_id": pt,
                "source_detected": source_detected,
                "target_detected": target_detected,
                "candidate_recorded": record is not None,
                "candidate_present": candidate_present,
                "ilp_selected": ilp_selected,
                "final_pair_present": final_pair_present,
                "final_correct": final_correct,
                "cause": cause,
                "blended_logit": float(record["blended_logit"]) if record else np.nan,
                "blended_prob": gt_prob,
                "seed1_logit": float(record["seed1_logit"]) if record else np.nan,
                "seed2_logit": float(record["seed2_logit"]) if record else np.nan,
                "seed1_prob": float(record["seed1_prob"]) if record else np.nan,
                "seed2_prob": float(record["seed2_prob"]) if record else np.nan,
                "source_rank": source_rank,
                "target_rank": target_rank,
                "rank_censored": record is None,
                "best_competing_source_id": int(best["source_id"]) if best else None,
                "best_competing_prob": best_prob,
                "competitor_margin": best_prob - gt_prob
                if np.isfinite(best_prob) and np.isfinite(gt_prob)
                else np.nan,
                "local_density_pred": _density(
                    detected, gt_by_id.loc[gt], scale_array, density_radius_um
                ),
                "density_radius_um": density_radius_um,
                "source_in_final": gs in final_g2p,
                "target_in_final": gt in final_g2p,
            }
        )

    frame = pd.DataFrame(rows)
    dataset_metadata = {
        "dataset": dataset,
        "ordinary_associations": len(frame),
        "detected_nodes": len(detected),
        "gt_nodes": len(gt_nodes),
        "detected_node_recall": len(detected_g2p) / len(gt_nodes) if len(gt_nodes) else 1.0,
        "final_node_recall": len(final_g2p) / len(gt_nodes) if len(gt_nodes) else 1.0,
        "top_k": top_k,
        "edge_threshold": float(metadata["edge_threshold"]),
    }
    return frame, dataset_metadata


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _summary_row(group_type: str, group: str, frame: pd.DataFrame) -> dict[str, Any]:
    errors = frame[frame["cause"] != "correct"]
    endpoint_matched = frame[frame["source_detected"] & frame["target_detected"]]
    causal_errors = errors[errors["source_detected"] & errors["target_detected"]]
    counts = {cause: int((frame["cause"] == cause).sum()) for cause in CAUSE_ORDER}
    return {
        "group_type": group_type,
        "group": group,
        "ordinary_associations": len(frame),
        "correct": counts["correct"],
        "errors": len(errors),
        "error_rate": _safe_rate(len(errors), len(frame)),
        "endpoint_matched": len(endpoint_matched),
        "candidate_recall": float(endpoint_matched["candidate_present"].mean())
        if len(endpoint_matched)
        else np.nan,
        "target_top1_rate": float((endpoint_matched["target_rank"] == 1).mean())
        if len(endpoint_matched)
        else np.nan,
        "ilp_recall": float(endpoint_matched["ilp_selected"].mean())
        if len(endpoint_matched)
        else np.nan,
        "final_edge_recall": float(endpoint_matched["final_correct"].mean())
        if len(endpoint_matched)
        else np.nan,
        "causal_errors": len(causal_errors),
        **{cause: counts[cause] for cause in CAUSE_ORDER if cause != "correct"},
        **{
            f"{cause}_share": _safe_rate(counts[cause], len(causal_errors))
            for cause in CAUSE_ORDER
            if cause not in {"correct", "detection_miss"}
        },
        "mean_density": float(frame["local_density_pred"].mean()) if len(frame) else np.nan,
        "median_competitor_margin": float(frame["competitor_margin"].median())
        if frame["competitor_margin"].notna().any()
        else np.nan,
    }


def summarize(
    frame: pd.DataFrame, dataset_metadata: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned, _ = add_bins(frame.rename(columns={"competitor_margin": "candidate_margin"}))
    binned = binned.rename(columns={"candidate_margin": "competitor_margin"})
    rows = [_summary_row("overall", "all", binned)]
    for column, kind in (
        ("dataset_family", "dataset_family"),
        ("density_bin", "density_bin"),
        ("dataset", "dataset"),
    ):
        for name, part in binned.groupby(column, observed=True, sort=False):
            rows.append(_summary_row(kind, str(name), part))
    metadata = pd.DataFrame(dataset_metadata)
    by_dataset = pd.DataFrame([row for row in rows if row["group_type"] == "dataset"])
    by_dataset = by_dataset.merge(metadata, left_on="group", right_on="dataset", how="left")
    return pd.DataFrame(rows), by_dataset


def classify(summary: pd.DataFrame) -> dict[str, Any]:
    overall = summary[(summary["group_type"] == "overall") & (summary["group"] == "all")].iloc[0]
    mechanisms = {
        cause: float(overall[f"{cause}_share"])
        for cause in (
            "scorer_ranking",
            "candidate_threshold",
            "ilp_global",
            "postprocessing_removed",
            "postprocessing_rematch",
        )
    }
    ranking_share = mechanisms["scorer_ranking"]
    runner_up = max(value for key, value in mechanisms.items() if key != "scorer_ranking")
    causal_errors = int(overall["causal_errors"])
    supported = causal_errors >= 20 and ranking_share >= 0.60 and ranking_share - runner_up >= 0.20
    rejected = (
        ranking_share < 0.40
        or max(
            mechanisms["candidate_threshold"],
            mechanisms["ilp_global"],
            mechanisms["postprocessing_removed"],
        )
        >= 0.40
    )
    classification = (
        "C. BOTTLENECK VERIFIED"
        if supported
        else "D. HYPOTHESIS REJECTED"
        if rejected
        else "B. PROMISING - NEEDS INDEPENDENT VALIDATION"
    )
    if supported:
        next_action = (
            "Retrain the edge scorer with dense-scene hard-negative mining and pairwise "
            "ranking supervision, then evaluate that one change on fixed-8 plus an "
            "independent holdout before changing the production recipe."
        )
    elif rejected:
        dominant = max(mechanisms, key=mechanisms.get)
        next_actions = {
            "candidate_threshold": (
                "Calibrate the pre-ILP edge gate from the captured logits, then test one "
                "selected threshold on fixed-8 plus an independent holdout."
            ),
            "ilp_global": (
                "Hold detections and edge scores fixed and audit one ILP objective change "
                "on fixed-8 plus an independent holdout."
            ),
            "postprocessing_removed": (
                "Hold raw GEFF predictions fixed and test one targeted postprocessing "
                "retention change on fixed-8 plus an independent holdout."
            ),
            "postprocessing_rematch": (
                "Audit node rematching after postprocessing and test one identity-preserving "
                "repair on fixed-8 plus an independent holdout."
            ),
            "scorer_ranking": (
                "Repeat the ranking diagnostic on an independent holdout before changing "
                "the production recipe."
            ),
        }
        next_action = next_actions[dominant]
    else:
        next_action = (
            "Repeat the same score-capture diagnostic on an independent holdout before "
            "choosing a mechanism-specific intervention."
        )
    return {
        "classification": classification,
        "hypothesis_supported": supported,
        "hypothesis_rejected": rejected,
        "causal_errors": causal_errors,
        "ranking_share": ranking_share,
        "runner_up_share": runner_up,
        "mechanism_shares": mechanisms,
        "recommended_next_action": next_action,
    }
