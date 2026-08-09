"""Association-error diagnostics stratified by local detection density.

The diagnostic intentionally operates on saved GEFF artifacts.  It can consume a
candidate-rich GEFF (selected and rejected edges) when one is available, while
remaining explicit when an exported graph contains selected edges only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from biohub_pipeline.evaluation import _node_match

FIXED8_DATASETS = (
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_969618f6",
    "6bba_fc83837d",
)

DENSITY_LABELS = ("low", "medium_low", "medium_high", "high")
MARGIN_LABELS = ("<0.01", "0.01-0.05", "0.05-0.15", ">=0.15", "unavailable")


@dataclass(frozen=True)
class GraphTables:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    candidate_export_complete: bool


def _optional_array(group: zarr.Group, key: str, length: int, default: object) -> np.ndarray:
    try:
        return np.asarray(group[key])
    except KeyError:
        return np.full(length, default)


def read_geff_tables(path: str | Path) -> GraphTables:
    """Read the small GEFF subset required by the diagnostic directly from Zarr."""
    group = zarr.open_group(Path(path), mode="r")
    node_ids = np.asarray(group["nodes/ids"], dtype=np.int64)
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "t": np.asarray(group["nodes/props/t/values"], dtype=np.int64),
            "z": np.asarray(group["nodes/props/z/values"], dtype=float),
            "y": np.asarray(group["nodes/props/y/values"], dtype=float),
            "x": np.asarray(group["nodes/props/x/values"], dtype=float),
            "selected": _optional_array(
                group, "nodes/props/solution/values", len(node_ids), True
            ).astype(bool),
        }
    )
    edge_ids = np.asarray(group["edges/ids"], dtype=np.int64)
    if edge_ids.size == 0:
        edge_ids = edge_ids.reshape(0, 2)
    edge_selected = _optional_array(
        group, "edges/props/solution/values", len(edge_ids), True
    ).astype(bool)
    edge_scores = _optional_array(
        group, "edges/props/edge_prob/values", len(edge_ids), np.nan
    ).astype(float)
    edges = pd.DataFrame(
        {
            "source_id": edge_ids[:, 0],
            "target_id": edge_ids[:, 1],
            "edge_score": edge_scores,
            "selected": edge_selected,
        }
    )
    # A selected-only graph cannot identify the runner-up, even if a source has
    # two selected division branches.  At least one rejected edge is the minimum
    # evidence that candidate alternatives were retained.
    complete = bool(len(edges) and (~edge_selected).any())
    return GraphTables(nodes=nodes, edges=edges, candidate_export_complete=complete)


def assign_density_bins(
    values: pd.Series | Iterable[float], thresholds: tuple[float, float, float] | None = None
) -> tuple[pd.Categorical, tuple[float, float, float]]:
    """Assign deterministic global quartile-threshold density bins."""
    series = pd.Series(values, dtype=float)
    finite = series[np.isfinite(series)]
    if thresholds is None:
        thresholds = (
            tuple(float(v) for v in np.quantile(finite, [0.25, 0.5, 0.75]))
            if len(finite)
            else (float("nan"),) * 3
        )
    q1, q2, q3 = thresholds
    labels = np.full(len(series), "unavailable", dtype=object)
    valid = np.isfinite(series.to_numpy(float))
    data = series.to_numpy(float)
    labels[valid & (data <= q1)] = "low"
    labels[valid & (data > q1) & (data <= q2)] = "medium_low"
    labels[valid & (data > q2) & (data <= q3)] = "medium_high"
    labels[valid & (data > q3)] = "high"
    categories = [*DENSITY_LABELS, "unavailable"]
    return pd.Categorical(labels, categories=categories, ordered=True), thresholds


def assign_margin_bins(values: pd.Series | Iterable[float]) -> pd.Categorical:
    """Bin top-one minus top-two edge-score margins with fixed boundaries."""
    series = pd.Series(values, dtype=float)
    data = series.to_numpy(float)
    labels = np.full(len(series), "unavailable", dtype=object)
    valid = np.isfinite(data)
    labels[valid & (data < 0.01)] = "<0.01"
    labels[valid & (data >= 0.01) & (data < 0.05)] = "0.01-0.05"
    labels[valid & (data >= 0.05) & (data < 0.15)] = "0.05-0.15"
    labels[valid & (data >= 0.15)] = ">=0.15"
    return pd.Categorical(labels, categories=MARGIN_LABELS, ordered=True)


def _distance_um(a: pd.Series, b: pd.Series, scale: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            (a[["z", "y", "x"]].to_numpy(float) - b[["z", "y", "x"]].to_numpy(float)) * scale
        )
    )


def _candidate_details(
    source_id: int | None,
    candidates_by_source: dict[int, pd.DataFrame],
    complete: bool,
) -> dict[str, object]:
    unavailable = {
        "candidate_count": np.nan,
        "best_candidate_target_id": np.nan,
        "best_candidate_score": np.nan,
        "second_candidate_score": np.nan,
        "candidate_margin": np.nan,
    }
    if source_id is None or not complete:
        return unavailable
    candidates = candidates_by_source.get(source_id)
    if candidates is None or candidates.empty:
        return {**unavailable, "candidate_count": 0}
    ranked = candidates.sort_values(
        ["edge_score", "target_id"], ascending=[False, True], na_position="last"
    )
    scores = ranked["edge_score"].to_numpy(float)
    best = float(scores[0]) if len(scores) and np.isfinite(scores[0]) else np.nan
    second = float(scores[1]) if len(scores) > 1 and np.isfinite(scores[1]) else np.nan
    return {
        "candidate_count": len(ranked),
        "best_candidate_target_id": int(ranked.iloc[0]["target_id"]) if len(ranked) else np.nan,
        "best_candidate_score": best,
        "second_candidate_score": second,
        "candidate_margin": best - second if np.isfinite(best) and np.isfinite(second) else np.nan,
    }


def diagnose_dataset(
    dataset: str,
    prediction: GraphTables,
    ground_truth: GraphTables,
    *,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_match_um: float = 7.0,
    density_radius_um: float = 15.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one row per non-division GT parent edge and node-match metadata."""
    scale_arr = np.asarray(scale, dtype=float)
    pred_nodes = prediction.nodes[prediction.nodes["selected"]].copy()
    selected_ids = set(pred_nodes["node_id"].astype(int))
    pred_edges = prediction.edges[
        prediction.edges["selected"]
        & prediction.edges["source_id"].isin(selected_ids)
        & prediction.edges["target_id"].isin(selected_ids)
    ].copy()
    gt_nodes = ground_truth.nodes.copy()
    gt_edges = ground_truth.edges.loc[:, ["source_id", "target_id"]].drop_duplicates()

    p2g, g2p, match_distances = _node_match(
        pred_nodes.loc[:, ["node_id", "t", "z", "y", "x"]],
        gt_nodes.loc[:, ["node_id", "t", "z", "y", "x"]],
        scale=scale,
        max_distance_um=max_match_um,
    )
    pred_by_id = pred_nodes.set_index("node_id")
    gt_by_id = gt_nodes.set_index("node_id")
    selected_by_source = {
        int(source): part.copy() for source, part in pred_edges.groupby("source_id", sort=False)
    }
    candidates_by_source = {
        int(source): part.copy()
        for source, part in prediction.edges.groupby("source_id", sort=False)
    }
    gt_outdegree = gt_edges.groupby("source_id").size()
    ordinary = gt_edges[gt_edges["source_id"].map(gt_outdegree).eq(1)].sort_values(
        ["source_id", "target_id"]
    )

    rows: list[dict[str, object]] = []
    for gt_edge in ordinary.itertuples(index=False):
        gs, gt = int(gt_edge.source_id), int(gt_edge.target_id)
        gt_source = gt_by_id.loc[gs]
        gt_target = gt_by_id.loc[gt]
        ps = int(g2p[gs]) if gs in g2p else None
        matched_target_pred = int(g2p[gt]) if gt in g2p else None
        selected = selected_by_source.get(ps, pd.DataFrame()) if ps is not None else pd.DataFrame()
        if not selected.empty:
            selected = selected.assign(
                _score=selected["edge_score"].fillna(float("-inf"))
            ).sort_values(["_score", "target_id"], ascending=[False, True])
            top_edge = selected.iloc[0]
            predicted_target = int(top_edge["target_id"])
            selected_score = float(top_edge["edge_score"])
            if not np.isfinite(selected_score):
                selected_score = np.nan
            predicted_target_gt = p2g.get(predicted_target)
            edge_distance = _distance_um(
                pred_by_id.loc[ps], pred_by_id.loc[predicted_target], scale_arr
            )
            correct_edges = [
                int(r.target_id)
                for r in selected.itertuples(index=False)
                if p2g.get(int(r.target_id)) == gt
            ]
            fp = sum(1 for r in selected.itertuples(index=False) if p2g.get(int(r.target_id)) != gt)
        else:
            predicted_target = None
            predicted_target_gt = None
            selected_score = np.nan
            edge_distance = np.nan
            correct_edges = []
            fp = 0
        correct = bool(correct_edges)
        if ps is None:
            outcome = "source_node_missed"
        elif matched_target_pred is None:
            outcome = "target_node_missed"
        elif correct and fp:
            outcome = "correct_with_extra_edge"
        elif correct:
            outcome = "correct"
        elif len(selected):
            outcome = "wrong_association"
        else:
            outcome = "missing_edge"

        next_frame = pred_nodes[pred_nodes["t"] == int(gt_target["t"])]
        if len(next_frame):
            distances = np.linalg.norm(
                (
                    next_frame[["z", "y", "x"]].to_numpy(float)
                    - gt_target[["z", "y", "x"]].to_numpy(float)
                )
                * scale_arr,
                axis=1,
            )
            density = int((distances <= density_radius_um).sum())
        else:
            density = 0
        persisted_outgoing = candidates_by_source.get(ps) if ps is not None else None
        candidate = _candidate_details(
            ps, candidates_by_source, prediction.candidate_export_complete
        )
        rows.append(
            {
                "dataset": dataset,
                "dataset_family": dataset.split("_")[0],
                "t": int(gt_source["t"]),
                "gt_source_id": gs,
                "gt_target_id": gt,
                "pred_source_id": ps,
                "predicted_target_id": predicted_target,
                "predicted_target_gt_id": predicted_target_gt,
                "matched_gt_target_pred_id": matched_target_pred,
                "source_node_matched": ps is not None,
                "target_node_matched": matched_target_pred is not None,
                "source_match_distance_um": match_distances.get(ps, np.nan)
                if ps is not None
                else np.nan,
                "target_match_distance_um": match_distances.get(matched_target_pred, np.nan)
                if matched_target_pred is not None
                else np.nan,
                "correct": correct,
                "outcome": outcome,
                "edge_tp": int(correct),
                "edge_fp": int(fp),
                "edge_fn": int(not correct),
                "selected_edge_score": selected_score,
                "selected_edge_distance_um": edge_distance,
                "gt_displacement_um": _distance_um(gt_source, gt_target, scale_arr),
                "selected_outgoing_edge_count": len(selected),
                "persisted_outgoing_edge_count": 0
                if persisted_outgoing is None
                else len(persisted_outgoing),
                "candidate_export_complete": prediction.candidate_export_complete,
                "local_density_pred": density,
                "density_radius_um": density_radius_um,
                **candidate,
            }
        )

    frame = pd.DataFrame(rows)
    metadata = {
        "dataset": dataset,
        "num_pred_nodes": len(pred_nodes),
        "num_gt_nodes": len(gt_nodes),
        "matched_gt_nodes": len(g2p),
        "node_recall": float(len(g2p) / len(gt_nodes)) if len(gt_nodes) else 1.0,
        "candidate_export_complete": prediction.candidate_export_complete,
    }
    return frame, metadata


def add_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    result = frame.copy()
    density, thresholds = assign_density_bins(result["local_density_pred"])
    result["density_bin"] = density
    result["margin_bin"] = assign_margin_bins(result["candidate_margin"])
    return result, thresholds


def _metric_row(group_type: str, group: str, part: pd.DataFrame) -> dict[str, object]:
    tp, fp, fn = (int(part[c].sum()) for c in ("edge_tp", "edge_fp", "edge_fn"))
    denominator = tp + fp + fn
    errors = int((~part["correct"]).sum())
    return {
        "group_type": group_type,
        "group": group,
        "associations": len(part),
        "correct": int(part["correct"].sum()),
        "errors": errors,
        "error_rate": float(errors / len(part)) if len(part) else np.nan,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "edge_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "edge_jaccard": float(tp / denominator) if denominator else 1.0,
        "source_node_missed": int((part["outcome"] == "source_node_missed").sum()),
        "target_node_missed": int((part["outcome"] == "target_node_missed").sum()),
        "wrong_association": int((part["outcome"] == "wrong_association").sum()),
        "missing_edge": int((part["outcome"] == "missing_edge").sum()),
        "mean_density": float(part["local_density_pred"].mean()) if len(part) else np.nan,
        "mean_candidate_count": float(part["candidate_count"].mean())
        if part["candidate_count"].notna().any()
        else np.nan,
        "median_candidate_count": float(part["candidate_count"].median())
        if part["candidate_count"].notna().any()
        else np.nan,
        "mean_candidate_margin": float(part["candidate_margin"].mean())
        if part["candidate_margin"].notna().any()
        else np.nan,
        "median_candidate_margin": float(part["candidate_margin"].median())
        if part["candidate_margin"].notna().any()
        else np.nan,
        "margin_available": int(part["candidate_margin"].notna().sum()),
    }


def build_summaries(
    frame: pd.DataFrame, node_metadata: list[dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [_metric_row("overall", "all", frame)]
    for column, kind in (
        ("dataset_family", "dataset_family"),
        ("density_bin", "density_bin"),
        ("margin_bin", "margin_bin"),
        ("outcome", "outcome"),
    ):
        for name, part in frame.groupby(column, observed=True, sort=False):
            rows.append(_metric_row(kind, str(name), part))
    summary = pd.DataFrame(rows)
    node_by_dataset = {str(row["dataset"]): row for row in node_metadata}
    dataset_rows = []
    for dataset, part in frame.groupby("dataset", sort=True):
        row = _metric_row("dataset", str(dataset), part)
        row.update(node_by_dataset[str(dataset)])
        dataset_rows.append(row)
    return summary, pd.DataFrame(dataset_rows)


def ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0:
        return np.nan
    if denominator == 0:
        return float("inf") if numerator > 0 else np.nan
    return float(numerator / denominator)


def classify_hypothesis(frame: pd.DataFrame) -> dict[str, object]:
    """Classify predeclared density/margin evidence as verified/partial/rejected."""
    density_rates = frame.groupby("density_bin", observed=True)["correct"].apply(
        lambda s: float((~s).mean())
    )
    low_density = float(density_rates.get("low", np.nan))
    high_density = float(density_rates.get("high", np.nan))
    density_ratio = ratio(high_density, low_density)
    margin_available = bool(frame["candidate_margin"].notna().any())
    small = frame[frame["candidate_margin"] < 0.05]
    large = frame[frame["candidate_margin"] >= 0.15]
    small_rate = float((~small["correct"]).mean()) if len(small) else np.nan
    large_rate = float((~large["correct"]).mean()) if len(large) else np.nan
    margin_ratio = ratio(small_rate, large_rate)
    errors = frame[~frame["correct"]]
    joint = errors[(errors["density_bin"] == "high") & (errors["candidate_margin"] < 0.05)]
    joint_share = float(len(joint) / len(errors)) if margin_available and len(errors) else np.nan
    association_errors = int(frame["outcome"].isin(["wrong_association", "missing_edge"]).sum())
    detection_errors = int(
        frame["outcome"].isin(["source_node_missed", "target_node_missed"]).sum()
    )

    density_signal = bool(not np.isnan(density_ratio) and density_ratio >= 1.5)
    margin_signal = bool(not np.isnan(margin_ratio) and margin_ratio >= 1.5)
    burden_signal = bool(np.isfinite(joint_share) and joint_share >= 0.40)
    association_dominates = association_errors > detection_errors
    # VERIFIED requires every predeclared signal.  Missing candidate margins can
    # support at most PARTIALLY VERIFIED; it is never silently treated as failure.
    if density_signal and margin_signal and burden_signal and association_dominates:
        conclusion = "A VERIFIED"
    elif density_signal and association_dominates:
        conclusion = "B PARTIALLY VERIFIED"
    else:
        conclusion = "C REJECTED"
    return {
        "conclusion": conclusion,
        "density_error_ratio_high_vs_low": density_ratio,
        "low_density_error_rate": low_density,
        "high_density_error_rate": high_density,
        "margin_available": margin_available,
        "small_margin_error_rate": small_rate,
        "large_margin_error_rate": large_rate,
        "margin_error_ratio_small_vs_large": margin_ratio,
        "high_density_small_margin_error_share": joint_share,
        "association_errors": association_errors,
        "detection_errors": detection_errors,
        "density_signal": density_signal,
        "margin_signal": margin_signal,
        "burden_signal": burden_signal,
        "association_dominates": association_dominates,
    }
