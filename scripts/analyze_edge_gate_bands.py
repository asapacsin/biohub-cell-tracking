#!/usr/bin/env python3
"""Probability-band diagnostic for the candidate-edge gate (motion-relink OFF base)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.evaluation import _node_match

BANDS = [(0.45, 0.50), (0.40, 0.45), (0.35, 0.40), (0.30, 0.35), (0.25, 0.30)]
SCALE = (1.625, 0.40625, 0.40625)
MAX_UM = 7.0
DATA = Path("data/competition/train")

FIXED8 = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_969618f6",
    "6bba_fc83837d",
]
HOLDOUT8 = [
    "44b6_0c582fdc",
    "44b6_0db75fae",
    "44b6_12dfb391",
    "44b6_144b256d",
    "6bba_062c8d37",
    "6bba_07477033",
    "6bba_07e24132",
    "6bba_085bf656",
]


def band_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{hi:.2f}"


def load_geff_edges(path: Path) -> set[tuple[int, int]]:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    rows = list(graph.edge_attrs().iter_rows(named=True))
    if not rows:
        return set()
    if "selected" in rows[0]:
        return {
            (int(r["source_id"]), int(r["target_id"]))
            for r in rows
            if bool(r.get("selected", True))
        }
    return {(int(r["source_id"]), int(r["target_id"])) for r in rows}


def count_geff_edges(path: Path) -> int:
    return len(load_geff_edges(path))


def final_edge_set(csv_path: Path, dataset: str) -> set[tuple[int, int]]:
    df = pd.read_csv(csv_path)
    part = df[(df["dataset"] == dataset) & (df["row_type"] == "edge")]
    return {(int(r.source_id), int(r.target_id)) for r in part.itertuples(index=False)}


def iter_capture_arrays(cap_dir: Path):
    for file in sorted(cap_dir.glob("t*_to_t*.npz")):
        with np.load(file) as data:
            yield (
                data["source_id"].astype(np.int64),
                data["target_id"].astype(np.int64),
                data["blended_prob"].astype(np.float64),
                data["target_rank"].astype(np.int64),
                data["above_threshold"].astype(bool),
            )


def analyze_split(
    name: str,
    datasets: list[str],
    cap_root: Path,
    raw_map: dict[str, Path],
    final_map: dict[str, Path],
    diag_path: Path,
) -> dict:
    print(f"[{name}] loading diagnostic {diag_path}", flush=True)
    diag = pd.read_csv(diag_path)
    band_cand = {band_label(*b): 0 for b in BANDS}
    band_true_any = {band_label(*b): 0 for b in BANDS}
    band_true_top1 = {band_label(*b): 0 for b in BANDS}
    band_ilp = {k: {band_label(*b): 0 for b in BANDS} for k in raw_map}
    band_final = {k: {band_label(*b): 0 for b in BANDS} for k in final_map}
    gt_threshold_misses_by_band = {band_label(*b): 0 for b in BANDS}
    gt_threshold_misses_total = 0
    band_tp = {band_label(*b): 0 for b in BANDS}
    band_n = {band_label(*b): 0 for b in BANDS}

    # Map ordinary GT associations to predicted node ids once per dataset.
    true_pairs_by_ds: dict[str, set[tuple[int, int]]] = {}
    for dataset in datasets:
        print(f"[{name}] mapping GT ordinary pairs {dataset}", flush=True)
        gt = read_geff_tables(DATA / f"{dataset}.geff")
        gt_edges = gt.edges.loc[:, ["source_id", "target_id"]].drop_duplicates()
        outdegree = gt_edges.groupby("source_id").size()
        ordinary = gt_edges[gt_edges["source_id"].map(outdegree).eq(1)]
        with np.load(cap_root / dataset / "nodes.npz") as data:
            detected = pd.DataFrame(
                {
                    "node_id": data["node_id"].astype(np.int64),
                    "t": data["t"].astype(np.int64),
                    "z": data["z"].astype(float),
                    "y": data["y"].astype(float),
                    "x": data["x"].astype(float),
                }
            )
        gt_nodes = gt.nodes.loc[:, ["node_id", "t", "z", "y", "x"]]
        _, g2p, _ = _node_match(detected, gt_nodes, scale=SCALE, max_distance_um=MAX_UM)
        mapped: set[tuple[int, int]] = set()
        for row in ordinary.itertuples(index=False):
            gs, gt_id = int(row.source_id), int(row.target_id)
            if gs in g2p and gt_id in g2p:
                mapped.add((int(g2p[gs]), int(g2p[gt_id])))
        true_pairs_by_ds[dataset] = mapped

    # Stream capture once: band volumes + precision proxy.
    for dataset in datasets:
        print(f"[{name}] streaming capture {dataset}", flush=True)
        truth = true_pairs_by_ds[dataset]
        for sources, targets, probs, _ranks, _above in iter_capture_arrays(cap_root / dataset):
            for lo, hi in BANDS:
                lab = band_label(lo, hi)
                mask = (probs >= lo) & (probs < hi)
                if not mask.any():
                    continue
                band_cand[lab] += int(mask.sum())
                band_n[lab] += int(mask.sum())
                src = sources[mask]
                tgt = targets[mask]
                # Hash pairs for set membership without Python row loops when possible.
                # Use python set for correctness on moderate band sizes (~5-8k per dataset).
                tp = 0
                for s, t in zip(src.tolist(), tgt.tolist(), strict=True):
                    if (s, t) in truth:
                        tp += 1
                band_tp[lab] += tp

    admitted: dict[str, int] = {}
    for label, raw_dir in raw_map.items():
        total = 0
        for dataset in datasets:
            path = raw_dir / f"{dataset}.geff"
            if path.exists():
                total += count_geff_edges(path)
        admitted[label] = total
        print(f"[{name}] admitted {label}={total}", flush=True)

    for row in diag.itertuples(index=False):
        if not (bool(row.source_detected) and bool(row.target_detected)):
            continue
        prob = float(row.blended_prob) if pd.notna(row.blended_prob) else float("nan")
        if not np.isfinite(prob):
            continue
        rank = int(row.target_rank) if pd.notna(row.target_rank) else 999
        cause = str(row.cause)
        for lo, hi in BANDS:
            if lo <= prob < hi:
                lab = band_label(lo, hi)
                band_true_any[lab] += 1
                if rank == 1:
                    band_true_top1[lab] += 1
                if cause == "candidate_threshold":
                    gt_threshold_misses_by_band[lab] += 1
                    gt_threshold_misses_total += 1
                break

    for dataset, part in diag.groupby("dataset"):
        if dataset not in datasets:
            continue
        print(f"[{name}] ILP/final survival {dataset}", flush=True)
        ilp_sets = {
            label: load_geff_edges(raw_dir / f"{dataset}.geff")
            if (raw_dir / f"{dataset}.geff").exists()
            else set()
            for label, raw_dir in raw_map.items()
        }
        fin_sets = {
            label: final_edge_set(csv_path, dataset) if csv_path.exists() else set()
            for label, csv_path in final_map.items()
        }
        for row in part.itertuples(index=False):
            if pd.isna(row.pred_source_id) or pd.isna(row.pred_target_id):
                continue
            if not (bool(row.source_detected) and bool(row.target_detected)):
                continue
            prob = float(row.blended_prob) if pd.notna(row.blended_prob) else float("nan")
            if not np.isfinite(prob):
                continue
            edge = (int(row.pred_source_id), int(row.pred_target_id))
            for lo, hi in BANDS:
                if lo <= prob < hi:
                    lab = band_label(lo, hi)
                    for label, eset in ilp_sets.items():
                        if edge in eset:
                            band_ilp[label][lab] += 1
                    for label, eset in fin_sets.items():
                        if edge in eset:
                            band_final[label][lab] += 1
                    break

    band_precision = {
        lab: {
            "captured_pairs": band_n[lab],
            "true_ordinary_matches": band_tp[lab],
            "precision_proxy": (band_tp[lab] / band_n[lab]) if band_n[lab] else None,
        }
        for lab in band_n
    }

    return {
        "split": name,
        "note": (
            "Capture is top-k censored; precision_proxy uses captured pairs only. "
            "GT threshold-error bands use motion-off diagnostic causes when available."
        ),
        "control_edge_threshold": 0.5,
        "captured_pairs_by_band": band_cand,
        "gt_ordinary_with_prob_in_band": band_true_any,
        "gt_ordinary_top1_in_band": band_true_top1,
        "gt_candidate_threshold_errors_by_band": gt_threshold_misses_by_band,
        "gt_candidate_threshold_errors_total": gt_threshold_misses_total,
        "ilp_selected_gt_ordinary_by_band": band_ilp,
        "final_present_gt_ordinary_by_band": band_final,
        "raw_geff_admitted_edges": admitted,
        "band_precision_proxy": band_precision,
        "additional_admitted_vs_0.5": {
            key: admitted[key] - admitted["edge_0.50"]
            for key in admitted
            if key != "edge_0.50" and "edge_0.50" in admitted
        },
    }


def main() -> None:
    out = Path(os.environ["OUT_NFS"]) / "band_diagnostic"
    out.mkdir(parents=True, exist_ok=True)

    fixed_diag = Path(".tmp_edge_gate_diags/fixed8_motion_off_diagnostic.csv")
    if not fixed_diag.exists():
        fixed_diag = Path(".tmp_edge_gate_diags/fixed8_diagnostic.csv")
    hold_diag = Path(".tmp_edge_gate_diags/holdout8_motion_off_diagnostic.csv")
    if not hold_diag.exists():
        hold_diag = Path(".tmp_edge_gate_diags/holdout8_diagnostic.csv")

    fixed = analyze_split(
        "fixed8",
        FIXED8,
        Path(os.environ["FIXED_CAP"]),
        {
            "edge_0.50": Path(os.environ["RAW50_F"]),
            "edge_0.45": Path(os.environ["RAW45_F"]),
            "edge_0.40": Path(os.environ["RAW40_F"]),
        },
        {
            "edge_0.50_motion_off": Path(os.environ["FINAL50_F"]),
            "edge_0.45_motion_off": Path(os.environ["FINAL45_F"]),
            "edge_0.40_motion_off": Path(os.environ["FINAL40_F"]),
        },
        fixed_diag,
    )
    hold = analyze_split(
        "holdout8",
        HOLDOUT8,
        Path(os.environ["HOLD_CAP"]),
        {
            "edge_0.50": Path(os.environ["RAW50_H"]),
            "edge_0.45": Path(os.environ["RAW45_H"]),
            "edge_0.40": Path(os.environ["RAW40_H"]),
        },
        {
            "edge_0.50_motion_off": Path(os.environ["FINAL50_H"]),
            "edge_0.45_motion_off": Path(os.environ["FINAL45_H"]),
            "edge_0.40_motion_off": Path(os.environ["FINAL40_H"]),
        },
        hold_diag,
    )
    combined = {"fixed8": fixed, "holdout8": hold}
    (out / "fixed8_band_summary.json").write_text(
        json.dumps(fixed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "holdout8_band_summary.json").write_text(
        json.dumps(hold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "combined_band_summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
