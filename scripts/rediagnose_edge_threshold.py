#!/usr/bin/env python3
"""Cause taxonomy for edge_threshold=0.40 motion-off vs edge=0.5 motion-off control."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from biohub_pipeline.association_density import read_geff_tables
from biohub_pipeline.candidate_bottleneck import analyze_dataset, classify, summarize
from biohub_pipeline.config import load_config

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

DATA_DIR = Path("data/competition/train")
CFG = Path("configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml")
OUT = Path(os.environ["OUT_NFS"])


def run_split(
    name: str,
    datasets: list[str],
    raw_dir: Path,
    cap_dir: Path,
    pred_csv: Path,
    *,
    edge_threshold: float,
) -> dict:
    config = load_config(CFG)
    scale = tuple(float(v) for v in config.postprocessing["voxel_scale_um"])
    max_um = float(config.local_cv["max_match_um"])
    preds = pd.read_csv(pred_csv)
    frames, metadata = [], []
    for dataset in datasets:
        frame, meta = analyze_dataset(
            dataset,
            Path(cap_dir) / dataset,
            Path(raw_dir) / f"{dataset}.geff",
            read_geff_tables(DATA_DIR / f"{dataset}.geff"),
            preds,
            scale=scale,
            max_match_um=max_um,
            edge_threshold=edge_threshold,
        )
        frames.append(frame)
        metadata.append(meta)
        print(name, dataset, "errors", int((frame.cause != "correct").sum()), flush=True)
    diagnostic = pd.concat(frames, ignore_index=True)
    summary, by_dataset = summarize(diagnostic, metadata)
    decision = classify(summary)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(out / "candidate_edge_diagnostic.csv", index=False)
    summary.to_csv(out / "cause_summary.csv", index=False)
    by_dataset.to_csv(out / "cause_by_dataset.csv", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    overall = summary[(summary.group_type == "overall") & (summary.group == "all")].iloc[0]
    # Extra short-track / endpoint view among postprocessing_removed
    pp = diagnostic[diagnostic.cause == "postprocessing_removed"]
    endpoint_stats = {
        "postprocessing_removed": int(len(pp)),
        "both_endpoints_in_final": int((pp.source_in_final & pp.target_in_final).sum()) if len(pp) else 0,
        "missing_either_endpoint": int((~pp.source_in_final | ~pp.target_in_final).sum()) if len(pp) else 0,
    }
    compact = {
        "split": name,
        "classification": decision["classification"],
        "mechanism_shares": decision["mechanism_shares"],
        "causal_errors": int(decision["causal_errors"]),
        "ranking_share": float(decision["ranking_share"]),
        "counts": {
            c: int(overall[c])
            for c in (
                "correct",
                "detection_miss",
                "scorer_ranking",
                "candidate_threshold",
                "ilp_global",
                "postprocessing_removed",
                "postprocessing_rematch",
            )
        },
        "postprocessing_removed_endpoints": endpoint_stats,
        "recommended_next_action": decision["recommended_next_action"],
        # User-facing A–F taxonomy mapping
        "taxonomy_AF": {
            "A_detection_miss_candidate_generation": int(overall["detection_miss"]),
            "B_scorer_ranking": int(overall["scorer_ranking"]),
            "C_threshold_gating": int(overall["candidate_threshold"]),
            "D_ilp_global": int(overall["ilp_global"]),
            "E_postprocessing_removed": int(overall["postprocessing_removed"]),
            "F_postprocessing_rematch": int(overall["postprocessing_rematch"]),
        },
    }
    (out / "compact.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    return compact


def main() -> None:
    fixed40 = run_split(
        "fixed8_edge_0_40",
        FIXED8,
        Path(os.environ["FIXED_RAW"]),
        Path(os.environ["FIXED_CAP"]),
        Path(os.environ["FIXED_PRED"]),
        edge_threshold=0.40,
    )
    hold40 = run_split(
        "holdout8_edge_0_40",
        HOLDOUT8,
        Path(os.environ["HOLD_RAW"]),
        Path(os.environ["HOLD_CAP"]),
        Path(os.environ["HOLD_PRED"]),
        edge_threshold=0.40,
    )
    fixed50 = run_split(
        "fixed8_edge_0_50_motion_off",
        FIXED8,
        Path(os.environ["FIXED_CTRL_RAW"]),
        Path(os.environ["FIXED_CAP"]),
        Path(os.environ["FIXED_CTRL_PRED"]),
        edge_threshold=0.50,
    )
    hold50 = run_split(
        "holdout8_edge_0_50_motion_off",
        HOLDOUT8,
        Path(os.environ["HOLD_CTRL_RAW"]),
        Path(os.environ["HOLD_CAP"]),
        Path(os.environ["HOLD_CTRL_PRED"]),
        edge_threshold=0.50,
    )
    combined = {
        "fixed8_edge_0_40": fixed40,
        "holdout8_edge_0_40": hold40,
        "fixed8_edge_0_50_motion_off": fixed50,
        "holdout8_edge_0_50_motion_off": hold50,
        "delta_counts_fixed8_0_40_minus_0_50": {
            k: fixed40["counts"][k] - fixed50["counts"][k] for k in fixed40["counts"]
        },
        "delta_counts_holdout8_0_40_minus_0_50": {
            k: hold40["counts"][k] - hold50["counts"][k] for k in hold40["counts"]
        },
    }
    (OUT / "combined_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    (OUT / "DONE").write_text("ok\n")
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
