#!/usr/bin/env python3
"""GPU edge_threshold sweep for 0.35 / 0.30 / 0.25 under motion-relink OFF."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import tracksdata as td

from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    _copy_raw_predictions,
    _estimated_total_nodes,
    _read_graph_tables,
    evaluate_postprocessed_predictions,
    evaluate_tables,
    find_fixed8_prediction_geffs,
    write_metric_outputs,
)
from biohub_pipeline.inference import (
    apply_spatial_d4_patch,
    build_predict_command,
    run_prediction,
)
from biohub_pipeline.submission import write_submission_from_geff

DATA_DIR = Path("data/competition/train").resolve()
SUPPORT = Path("data/support").resolve()
OUT_NFS = Path(os.environ["OUT_NFS"]).resolve()
OUT = Path("outputs/experiments/edge_gate_sweep_motion_off_v1").resolve()

CONTROL_FIXED = 0.9067252533426169
CONTROL_HOLD = 0.9603073913068769
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
THRESHOLDS = [0.35, 0.30, 0.25]


def thr_tag(thresh: float) -> str:
    return f"{thresh:.2f}".replace(".", "_")


def evaluate_custom(pred_csv, datasets, config):
    import pandas as pd

    predictions = pd.read_csv(pred_csv)
    scale = tuple(float(v) for v in config.postprocessing["voxel_scale_um"])
    max_distance_um = float(config.local_cv["max_match_um"])
    rows = []
    for dataset in datasets:
        part = predictions[predictions["dataset"] == dataset]
        pred_nodes = part[part["row_type"] == "node"].loc[:, ["node_id", "t", "z", "y", "x"]]
        pred_edges = part[part["row_type"] == "edge"].loc[:, ["source_id", "target_id"]]
        gt_nodes, gt_edges = _read_graph_tables(DATA_DIR / f"{dataset}.geff")
        rows.append(
            evaluate_tables(
                dataset,
                pred_nodes,
                pred_edges,
                gt_nodes,
                gt_edges,
                _estimated_total_nodes(DATA_DIR / f"{dataset}.geff"),
                scale=scale,
                max_distance_um=max_distance_um,
            )
        )
    return rows, {**official_spec_summarise(rows), "datasets": list(datasets)}


def count_raw_edges(raw_dir: Path, datasets) -> int:
    total = 0
    for dataset in datasets:
        geff = raw_dir / f"{dataset}.geff"
        loaded = td.graph.IndexedRXGraph.from_geff(geff)
        graph = loaded[0] if isinstance(loaded, tuple) else loaded
        total += sum(1 for _ in graph.edge_attrs().iter_rows(named=True))
    return total


def run_split(thresh: float, name: str, datasets: list[str], control_score: float) -> dict:
    cfg_path = Path(f"configs/experiments/edge_thresh_{thr_tag(thresh)}_motion_off_det0_96875.yaml")
    config = load_config(cfg_path)
    assert abs(float(config.inference["edge_threshold"]) - thresh) < 1e-12
    assert config.postprocessing["output_motion_relink"] is False

    out = OUT / f"thresh_{thr_tag(thresh)}" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    repo = work / "support_repo"
    shutil.copytree(SUPPORT / "repo", repo)
    if config.inference["spatial_d4_tta"]:
        apply_spatial_d4_patch(repo, str(config.inference["prediction_script"]))
    primary = SUPPORT / Path(str(config.inference["weights_relative"]))
    command, _ = build_predict_command(config, DATA_DIR, repo, primary, datasets)
    (out / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print("RUN", thresh, name, "has_flag", "--edge-threshold" in command, flush=True)
    started = time.perf_counter()
    run_prediction(command, repo)
    geffs = find_fixed8_prediction_geffs(repo, str(config.inference["method"]), datasets)
    raw_dir = _copy_raw_predictions(geffs, out, config_path=cfg_path)
    pred_csv = out / "predictions" / "postprocessed_submission.csv"
    write_submission_from_geff(geffs, config, DATA_DIR, pred_csv)
    if datasets == list(FIXED8_DATASETS):
        rows, aggregate = evaluate_postprocessed_predictions(pred_csv, DATA_DIR, config)
    else:
        rows, aggregate = evaluate_custom(pred_csv, datasets, config)
    admitted = count_raw_edges(raw_dir, datasets)
    runtime = time.perf_counter() - started
    per_dataset = [
        {
            "dataset": r["dataset"],
            "adj_edge_jaccard": float(r["adj_edge_jaccard"]),
            "edge_tp": int(r["edge_tp"]),
            "edge_fp": int(r["edge_fp"]),
            "edge_fn": int(r["edge_fn"]),
            "division_tp": int(r["division_tp"]),
            "division_fp": int(r["division_fp"]),
            "division_fn": int(r["division_fn"]),
        }
        for r in rows
    ]
    summary = {
        **aggregate,
        "label": f"{name}_edge_thresh_{thresh}_motion_off",
        "edge_threshold": thresh,
        "output_motion_relink": False,
        "detection_threshold": 0.96875,
        "delta_vs_motion_off_edge_0_5": float(aggregate["score"]) - control_score,
        "control_score_motion_off_edge_0_5": control_score,
        "raw_geff_admitted_edges": admitted,
        "runtime_seconds": runtime,
        "per_dataset": per_dataset,
    }
    write_metric_outputs(
        out,
        rows,
        summary,
        {
            "schema_version": 1,
            "experiment": "edge_gate_lower_sweep",
            "split": name,
            "edge_threshold": thresh,
            "config": str(cfg_path),
        },
    )
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(
        name,
        thresh,
        "score",
        summary["score"],
        "delta",
        summary["delta_vs_motion_off_edge_0_5"],
        "admitted",
        admitted,
        flush=True,
    )
    return summary


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    results: dict = {
        "thresholds": {},
        "control_fixed8": CONTROL_FIXED,
        "control_holdout8": CONTROL_HOLD,
    }
    for thr in THRESHOLDS:
        fixed = run_split(thr, "fixed8", list(FIXED8_DATASETS), CONTROL_FIXED)
        hold = run_split(thr, "holdout8", HOLDOUT8, CONTROL_HOLD)
        results["thresholds"][f"{thr:.2f}"] = {
            "fixed8": {
                "score": float(fixed["score"]),
                "adj_edge_jaccard": float(fixed["adj_edge_jaccard"]),
                "delta_vs_motion_off_edge_0_5": float(fixed["delta_vs_motion_off_edge_0_5"]),
                "edge_tp_fp_fn": [fixed["edge_tp"], fixed["edge_fp"], fixed["edge_fn"]],
                "div_tp_fp_fn": [
                    fixed["division_tp"],
                    fixed["division_fp"],
                    fixed["division_fn"],
                ],
                "raw_geff_admitted_edges": int(fixed["raw_geff_admitted_edges"]),
                "runtime_seconds": float(fixed["runtime_seconds"]),
                "per_dataset": fixed["per_dataset"],
            },
            "holdout8": {
                "score": float(hold["score"]),
                "adj_edge_jaccard": float(hold["adj_edge_jaccard"]),
                "delta_vs_motion_off_edge_0_5": float(hold["delta_vs_motion_off_edge_0_5"]),
                "edge_tp_fp_fn": [hold["edge_tp"], hold["edge_fp"], hold["edge_fn"]],
                "div_tp_fp_fn": [
                    hold["division_tp"],
                    hold["division_fp"],
                    hold["division_fn"],
                ],
                "raw_geff_admitted_edges": int(hold["raw_geff_admitted_edges"]),
                "runtime_seconds": float(hold["runtime_seconds"]),
                "per_dataset": hold["per_dataset"],
            },
        }

    (OUT / "lower_sweep_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "DONE_LOWER_SWEEP").write_text("ok\n", encoding="utf-8")

    OUT_NFS.mkdir(parents=True, exist_ok=True)
    for thr in THRESHOLDS:
        key = f"thresh_{thr_tag(thr)}"
        src = OUT / key
        dst = OUT_NFS / key
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("work"))
    shutil.copy2(OUT / "lower_sweep_results.json", OUT_NFS / "lower_sweep_results.json")
    (OUT_NFS / "DONE_LOWER_SWEEP").write_text("ok\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
