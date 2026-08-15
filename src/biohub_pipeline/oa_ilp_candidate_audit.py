"""Audit remaining ordinary-association failures with frozen detections.

Splits bottleneck ``scorer_ranking`` rows into GT-below-gate vs GT-in-ILP-graph,
and inspects ILP source/target contention on the gated candidate set. Does not
change the production recipe or retrain the edge scorer.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biohub_pipeline.association_density import read_geff_tables

CAUSE_BUCKETS = (
    "detection_miss",
    "candidate_absent",
    "candidate_threshold",
    "ranking_below_gate",
    "ranking_gated",
    "ilp_global",
    "postprocessing_rematch",
    "other",
)


def bucket_error(row: pd.Series | dict[str, Any]) -> str:
    """Map one diagnostic error row onto an ILP / candidate-set bucket."""
    cause = str(row["cause"])
    if cause == "detection_miss":
        return "detection_miss"
    recorded = bool(row.get("candidate_recorded", False))
    present = bool(row.get("candidate_present", False))
    rank = int(row["target_rank"]) if pd.notna(row.get("target_rank")) else 99
    if not recorded:
        return "candidate_absent"
    if rank > 1 and not present:
        return "ranking_below_gate"
    if rank > 1 and present:
        return "ranking_gated"
    if cause == "candidate_threshold" or not present:
        return "candidate_threshold"
    if cause == "ilp_global" or not bool(row.get("ilp_selected", False)):
        return "ilp_global"
    if cause == "postprocessing_rematch":
        return "postprocessing_rematch"
    return "other"


def load_capture_frame(path: Path) -> pd.DataFrame:
    with np.load(path) as data:
        return pd.DataFrame(
            {
                "source_id": data["source_id"].astype(np.int64),
                "target_id": data["target_id"].astype(np.int64),
                "blended_prob": data["blended_prob"].astype(np.float64),
                "source_rank": data["source_rank"].astype(np.int64),
                "target_rank": data["target_rank"].astype(np.int64),
                "above_threshold": data["above_threshold"].astype(bool),
            }
        )


def load_capture_frames(capture_dir: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for path in sorted(capture_dir.glob("t*_to_t*.npz")):
        t_src = int(path.name.split("_")[0][1:])
        frames[t_src] = load_capture_frame(path)
    return frames


def load_selected_pairs(geff_path: Path | None, submission: pd.DataFrame | None, dataset: str) -> set[tuple[int, int]]:
    if geff_path is not None and Path(geff_path).exists():
        graph = read_geff_tables(geff_path)
        selected = graph.edges[graph.edges["selected"]]
        return {
            (int(row.source_id), int(row.target_id))
            for row in selected.itertuples(index=False)
        }
    if submission is None:
        return set()
    part = submission[(submission["dataset"] == dataset) & (submission["row_type"] == "edge")]
    return {(int(row.source_id), int(row.target_id)) for row in part.itertuples(index=False)}


def _best_row(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any] | None:
    part = frame.loc[mask]
    if part.empty:
        return None
    best = part.loc[part["blended_prob"].idxmax()]
    return {
        "source_id": int(best.source_id),
        "target_id": int(best.target_id),
        "blended_prob": float(best.blended_prob),
        "source_rank": int(best.source_rank),
        "target_rank": int(best.target_rank),
        "above_threshold": bool(best.above_threshold),
    }


def inspect_assignment(
    frame: pd.DataFrame | None,
    selected: set[tuple[int, int]],
    source_id: int | None,
    target_id: int | None,
) -> dict[str, Any]:
    """Describe gated competitors and selected occupancy for one predicted GT pair."""
    empty = {
        "gated_out_from_source": 0,
        "gated_in_to_target": 0,
        "source_top1_targets_gated": 0,
        "best_gated_outgoing": None,
        "best_gated_incoming": None,
        "selected_out": [],
        "selected_in": [],
        "conflict": "unknown",
        "gt_is_source_top1": False,
        "source_has_wrong_gated_child": False,
        "source_has_no_gated_child": False,
    }
    if source_id is None or target_id is None or pd.isna(source_id) or pd.isna(target_id):
        return empty
    source_id = int(source_id)
    target_id = int(target_id)
    selected_out = sorted(t for s, t in selected if s == source_id)
    selected_in = sorted(s for s, t in selected if t == target_id)
    result = dict(empty)
    result["selected_out"] = selected_out
    result["selected_in"] = selected_in
    if (source_id, target_id) in selected:
        result["conflict"] = "gt_selected"
    elif selected_out and selected_in:
        result["conflict"] = "both_taken"
    elif selected_out:
        result["conflict"] = "source_taken"
    elif selected_in:
        result["conflict"] = "target_taken"
    else:
        result["conflict"] = "unmatched"

    if frame is None or frame.empty:
        return result
    gated = frame[frame["above_threshold"]]
    src_gated = gated[gated["source_id"] == source_id]
    tgt_gated = gated[gated["target_id"] == target_id]
    result["gated_out_from_source"] = int(len(src_gated))
    result["gated_in_to_target"] = int(len(tgt_gated))
    result["source_top1_targets_gated"] = int(
        ((gated["source_id"] == source_id) & (gated["source_rank"] == 1)).sum()
    )
    best_out = _best_row(frame, frame["source_id"] == source_id)
    best_gated_out = _best_row(gated, gated["source_id"] == source_id)
    best_gated_in = _best_row(gated, gated["target_id"] == target_id)
    result["best_gated_outgoing"] = best_gated_out
    result["best_gated_incoming"] = best_gated_in
    result["gt_is_source_top1"] = bool(
        best_out is not None and int(best_out["target_id"]) == target_id and int(best_out["source_rank"]) == 1
    )
    result["source_has_no_gated_child"] = best_gated_out is None
    result["source_has_wrong_gated_child"] = bool(
        best_gated_out is not None and int(best_gated_out["target_id"]) != target_id
    )
    return result


def _recoverability(bucket: str, row: pd.Series, assignment: dict[str, Any]) -> str:
    if bucket == "detection_miss":
        return "needs_detection"
    if bucket == "postprocessing_rematch":
        return "needs_identity_repair"
    if bucket == "ilp_global":
        return "ilp_contention_repair"
    if bucket == "ranking_gated":
        return "ilp_override_rank2"
    if bucket in {"ranking_below_gate", "candidate_threshold", "candidate_absent"}:
        source_rank = int(row["source_rank"]) if pd.notna(row.get("source_rank")) else 99
        if source_rank == 1 or assignment.get("gt_is_source_top1"):
            return "admit_source_top1"
        return "needs_ranking_or_lower_gate"
    return "other"


def census_source_top1_below_gate(
    frames: dict[int, pd.DataFrame],
    gt_err_pairs: set[tuple[int, int]],
    gt_ok_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    """Count source-top-1 pairs excluded by the 0.40 gate (FP-risk of admitting them)."""
    n_below = 0
    n_gt_err = 0
    n_gt_ok = 0
    probs: list[float] = []
    for frame in frames.values():
        if frame.empty:
            continue
        top1 = frame[frame["source_rank"] == 1]
        below = top1[~top1["above_threshold"]]
        n_below += int(len(below))
        for row in below.itertuples(index=False):
            pair = (int(row.source_id), int(row.target_id))
            probs.append(float(row.blended_prob))
            if pair in gt_err_pairs:
                n_gt_err += 1
            if pair in gt_ok_pairs:
                n_gt_ok += 1
    extra_to_err = float(n_below / n_gt_err) if n_gt_err else float("inf") if n_below else 0.0
    return {
        "source_top1_below_gate": n_below,
        "source_top1_below_gate_gt_err": n_gt_err,
        "source_top1_below_gate_gt_ok": n_gt_ok,
        "source_top1_below_gate_median_prob": float(np.median(probs)) if probs else float("nan"),
        "extra_candidates_per_gt_err": extra_to_err,
        "admit_source_top1_unsafe": bool(n_below >= 50 and extra_to_err >= 20.0),
    }


def audit_errors(
    diagnostic: pd.DataFrame,
    capture_dir: Path,
    *,
    dataset: str,
    selected: set[tuple[int, int]] | None = None,
    geff_path: Path | None = None,
    submission: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect OA errors for one dataset under frozen detections."""
    frame_index = load_capture_frames(capture_dir)
    if selected is None:
        selected = load_selected_pairs(geff_path, submission, dataset)
    part = diagnostic[diagnostic["dataset"] == dataset].copy()
    errors = part[part["cause"] != "correct"].copy()
    rows: list[dict[str, Any]] = []
    for row in errors.itertuples(index=False):
        record = row._asdict()
        bucket = bucket_error(record)
        t = int(record["t"])
        assignment = inspect_assignment(
            frame_index.get(t),
            selected,
            record.get("pred_source_id"),
            record.get("pred_target_id"),
        )
        recover = _recoverability(bucket, pd.Series(record), assignment)
        best_out = assignment.get("best_gated_outgoing") or {}
        rows.append(
            {
                **{k: record[k] for k in record},
                "bucket": bucket,
                "recoverability": recover,
                "conflict": assignment["conflict"],
                "gated_out_from_source": assignment["gated_out_from_source"],
                "gated_in_to_target": assignment["gated_in_to_target"],
                "gt_is_source_top1": assignment["gt_is_source_top1"],
                "source_has_wrong_gated_child": assignment["source_has_wrong_gated_child"],
                "source_has_no_gated_child": assignment["source_has_no_gated_child"],
                "selected_out": ",".join(str(v) for v in assignment["selected_out"]),
                "selected_in": ",".join(str(v) for v in assignment["selected_in"]),
                "best_gated_outgoing_target": best_out.get("target_id"),
                "best_gated_outgoing_prob": best_out.get("blended_prob"),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["bucket", "recoverability", "conflict"])

    # Rematch cascade: pred node shared with an ILP/ranking failure.
    conflict_nodes: set[int] = set()
    for row in table.itertuples(index=False):
        if row.bucket in {"ilp_global", "ranking_gated", "ranking_below_gate"}:
            if pd.notna(row.pred_source_id):
                conflict_nodes.add(int(row.pred_source_id))
            if pd.notna(row.pred_target_id):
                conflict_nodes.add(int(row.pred_target_id))
    cascade = []
    for row in table.itertuples(index=False):
        if row.bucket != "postprocessing_rematch":
            cascade.append(False)
            continue
        nodes = set()
        if pd.notna(row.pred_source_id):
            nodes.add(int(row.pred_source_id))
        if pd.notna(row.pred_target_id):
            nodes.add(int(row.pred_target_id))
        cascade.append(bool(nodes & conflict_nodes))
    table["rematch_shares_conflict_node"] = cascade

    causal = table[table["bucket"] != "detection_miss"]
    bucket_counts = {name: int((table["bucket"] == name).sum()) for name in CAUSE_BUCKETS}
    recover_counts: dict[str, int] = defaultdict(int)
    for value in table["recoverability"]:
        recover_counts[str(value)] += 1
    conflict_counts: dict[str, int] = defaultdict(int)
    for value in table.loc[table["bucket"] == "ilp_global", "conflict"]:
        conflict_counts[str(value)] += 1

    n_causal = int(len(causal))
    n_errors = int(len(table))
    below_gate = bucket_counts["ranking_below_gate"] + bucket_counts["candidate_threshold"] + bucket_counts["candidate_absent"]
    in_ilp_graph = bucket_counts["ranking_gated"] + bucket_counts["ilp_global"]
    gt_err_pairs = {
        (int(r.pred_source_id), int(r.pred_target_id))
        for r in errors.itertuples(index=False)
        if pd.notna(r.pred_source_id) and pd.notna(r.pred_target_id)
    }
    gt_ok_pairs = {
        (int(r.pred_source_id), int(r.pred_target_id))
        for r in part[part["cause"] == "correct"].itertuples(index=False)
        if pd.notna(r.pred_source_id) and pd.notna(r.pred_target_id)
    }
    census = census_source_top1_below_gate(frame_index, gt_err_pairs, gt_ok_pairs)
    ilp_stolen_not_gt = 0
    for row in table.itertuples(index=False):
        if row.bucket != "ilp_global":
            continue
        stolen_raw = str(row.selected_out).strip()
        if not stolen_raw:
            continue
        stolen = int(stolen_raw.split(",")[0])
        src = int(row.pred_source_id)
        if (src, stolen) not in gt_ok_pairs and (src, stolen) not in gt_err_pairs:
            ilp_stolen_not_gt += 1
    summary = {
        "dataset": dataset,
        "ordinary_associations": int(len(part)),
        "errors": n_errors,
        "causal_errors": n_causal,
        "bucket_counts": bucket_counts,
        "recoverability_counts": dict(recover_counts),
        "ilp_conflict_counts": dict(conflict_counts),
        "causal_share_below_gate": float(below_gate / n_causal) if n_causal else float("nan"),
        "causal_share_in_ilp_graph": float(in_ilp_graph / n_causal) if n_causal else float("nan"),
        "causal_share_ilp_contention": float(bucket_counts["ilp_global"] / n_causal) if n_causal else float("nan"),
        "causal_share_rematch": float(bucket_counts["postprocessing_rematch"] / n_causal) if n_causal else float("nan"),
        "ranking_below_gate_source_top1": int(
            (
                (table["bucket"] == "ranking_below_gate")
                & (table["source_rank"] == 1)
            ).sum()
        ),
        "ranking_below_gate_wrong_gated_child": int(
            (
                (table["bucket"] == "ranking_below_gate")
                & table["source_has_wrong_gated_child"]
            ).sum()
        ),
        "ranking_below_gate_no_gated_child": int(
            (
                (table["bucket"] == "ranking_below_gate")
                & table["source_has_no_gated_child"]
            ).sum()
        ),
        "rematch_cascade_from_conflict": int(table["rematch_shares_conflict_node"].sum()),
        "selected_edge_source": "geff" if geff_path is not None and Path(geff_path).exists() else "submission",
        "ilp_stolen_child_not_gt": ilp_stolen_not_gt,
        **census,
        "recommended_next": _recommend(bucket_counts, n_causal, table, census, ilp_stolen_not_gt),
    }
    return table, summary


def _recommend(
    bucket_counts: dict[str, int],
    n_causal: int,
    table: pd.DataFrame,
    census: dict[str, Any],
    ilp_stolen_not_gt: int,
) -> dict[str, Any]:
    closed = [
        "edge-scorer retrain",
        "global edge_threshold < 0.40",
        "short-track OFF",
        "motion-relink ON",
        "admit every source-top1 below 0.40",
    ]
    below = bucket_counts["ranking_below_gate"] + bucket_counts["candidate_threshold"]
    source_top1 = int(
        ((table["bucket"] == "ranking_below_gate") & (table["source_rank"] == 1)).sum()
    )
    admit_unsafe = bool(census.get("admit_source_top1_unsafe"))
    if n_causal and below / n_causal >= 0.40 and source_top1 >= 5 and not admit_unsafe:
        return {
            "experiment": "admit_source_top1_below_gate",
            "rationale": (
                "Most remaining OA failures never enter the ILP graph because GT "
                "softmax is below 0.40, even though the pair is the source's top-1 "
                "outgoing. Admit each source's top-1 continuation regardless of the "
                "global gate; keep edge_threshold=0.40 for all other pairs."
            ),
            "do_not": closed,
        }
    if (
        bucket_counts["ilp_global"] >= 4
        and ilp_stolen_not_gt == bucket_counts["ilp_global"]
        and admit_unsafe
    ):
        return {
            "experiment": "close_ilp_candidate_set_branch",
            "rationale": (
                "The candidate-set hole is real (most causal GTs never enter ILP) but "
                "admitting source-top-1 below 0.40 would add too many extra candidates "
                "for the current disappearance cost to ignore. The ILP occupancy errors "
                "are the same proximal extra-detection theft already rejected as a "
                "ranking term: the stolen child is not a GT association. Do not change "
                "the frozen recipe. Do not start another edge-scorer retrain."
            ),
            "do_not": closed,
        }
    if bucket_counts["ilp_global"] >= max(4, 0.30 * n_causal if n_causal else 0):
        return {
            "experiment": "ilp_source_contention_repair",
            "rationale": (
                "Rank-1 gated GT parents lose because the source's preferred child "
                "occupies the unique outgoing slot. Hold detections and scores fixed "
                "and test one ILP occupancy rule."
            ),
            "do_not": closed,
        }
    return {
        "experiment": "close_ilp_candidate_set_branch",
        "rationale": "No single ILP/candidate-set lever covers remaining errors without a false-positive bomb or a scorer retrain.",
        "do_not": closed,
    }
