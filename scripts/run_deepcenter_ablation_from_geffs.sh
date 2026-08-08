#!/usr/bin/env bash
# Re-convert saved two-seed det=0.960 GEFFs with DeepCenter gating ON (no re-inference).
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-deepcenter-ablation}"
CONTROL_RAW="${CONTROL_RAW:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/raw_geff}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_det0_960_deepcenter}"
CONTROL_SUMMARY="${CONTROL_SUMMARY:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/summary.json}"
CONTROL_PER="${CONTROL_PER:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/per_dataset.csv}"
CONTROL_PRED="${CONTROL_PRED:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/predictions/postprocessed_submission.csv}"
CONFIG_REL="configs/experiments/two_seed_det0_960_deepcenter.yaml"
CONTROL_SCORE="${CONTROL_SCORE:-0.8879786102989093}"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

test -d "$CONTROL_RAW"
test "$(find "$CONTROL_RAW" -maxdepth 1 -name '*.geff' | wc -l)" -eq 8
test -f "$CONFIG_REL"
test -f data/deepcenter/weights/full_frame_center/checkpoint_last.pt

rm -rf "$OUT"
mkdir -p "$OUT/predictions" "$OUT/config"
cp -a "$CONFIG_REL" "$OUT/config/two_seed_det0_960_deepcenter.yaml"

CONTROL_SCORE="$CONTROL_SCORE" \
CONTROL_RAW="$CONTROL_RAW" \
OUT="$OUT" \
CONFIG_REL="$CONFIG_REL" \
CONTROL_SUMMARY="$CONTROL_SUMMARY" \
CONTROL_PER="$CONTROL_PER" \
CONTROL_PRED="$CONTROL_PRED" \
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import _node_match, _normalise_nodes
from biohub_pipeline.fixed8_cv import (
    MATERIAL_DELTA,
    evaluate_postprocessed_predictions,
    write_metric_outputs,
)
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

control_score = float(os.environ["CONTROL_SCORE"])
control_raw = Path(os.environ["CONTROL_RAW"])
out = Path(os.environ["OUT"])
config_path = Path(os.environ["CONFIG_REL"])
control_summary_path = Path(os.environ["CONTROL_SUMMARY"])
control_per_path = Path(os.environ["CONTROL_PER"])
control_pred_path = Path(os.environ["CONTROL_PRED"])
data_dir = Path("data/competition/train")

config = load_config(config_path)
assert float(config.inference["detection_threshold"]) == 0.960
assert float(config.inference["ensemble_alpha"]) == 0.5
assert config.postprocessing["output_gap2_recovery"] is False
assert config.postprocessing["use_deepcenter_veto"] is True
assert config.postprocessing["require_deepcenter_veto"] is True
assert config.postprocessing["deepcenter_gap_veto"] is True
assert config.postprocessing["deepcenter_safe_div_veto"] is True

raw_dst = out / "raw_geff_source"
raw_dst.mkdir(parents=True, exist_ok=True)
geffs = sorted(control_raw.glob("*.geff"))
assert len(geffs) == 8
for src in geffs:
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)

postprocessing.configure(config.postprocessing, data_dir)
bundle = postprocessing.load_deepcenter_veto_detector()
assert bundle is not None, "DeepCenter checkpoint failed to load"
print("DEEPCENTER_LOADED", bundle["path"])

control_edges_by_ds: dict[str, set[tuple[int, int]]] = defaultdict(set)
control_nodes_by_ds: dict[str, list[dict]] = defaultdict(list)
with control_pred_path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        ds = row["dataset"]
        if row["row_type"] == "edge":
            control_edges_by_ds[ds].add((int(row["source_id"]), int(row["target_id"])))
        else:
            control_nodes_by_ds[ds].append(
                {
                    "node_id": int(row["node_id"]),
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }
            )

started = time.perf_counter()
all_rows: list[dict] = []
stats_by_dataset: dict[str, dict] = {}
removed_edges: list[dict] = []
pred_dc: dict[str, set[tuple[int, int]]] = {}

for geff_path in sorted(raw_dst.glob("*.geff")):
    dataset = geff_path.stem
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(row["node_id"]): {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        probability = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if probability is None else float(probability),
            }
        )

    nodes_out, edges_out, stats = postprocessing.filter_output_graph(
        nodes,
        edges,
        dataset=dataset,
        deepcenter_bundle=bundle,
    )
    stats_by_dataset[dataset] = {
        k: int(v) for k, v in stats.items() if str(k).startswith("deepcenter_")
    }
    stats_by_dataset[dataset]["gap_pairs_selected"] = int(stats.get("gap_pairs_selected", 0))
    stats_by_dataset[dataset]["safe_divisions_added"] = int(stats.get("safe_divisions_added", 0))

    dc_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges_out}
    pred_dc[dataset] = dc_keys
    for src, tgt in sorted(control_edges_by_ds[dataset] - dc_keys):
        removed_edges.append({"dataset": dataset, "source_id": src, "target_id": tgt})

    all_rows.extend(graph_rows(dataset, nodes_out, edges_out, len(all_rows)))

pred_csv = out / "predictions" / "postprocessed_submission.csv"
write_rows(all_rows, pred_csv)
print("CONVERT_OK", validate_submission_file(pred_csv))

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
delta = float(aggregate["score"]) - control_score

# Classify removed control edges as TP/FP vs GT via control-node matching.
removed_tp = removed_fp = removed_unknown = 0
removed_by_ds: dict[str, dict[str, int]] = {}
for dataset in sorted(control_edges_by_ds):
    loaded = td.graph.IndexedRXGraph.from_geff(data_dir / f"{dataset}.geff")
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    gt_nodes = pd.DataFrame(
        [
            {
                "node_id": int(r["node_id"]),
                "t": int(r["t"]),
                "z": float(r["z"]),
                "y": float(r["y"]),
                "x": float(r["x"]),
            }
            for r in graph.node_attrs().iter_rows(named=True)
        ]
    )
    gt_edge_set = {
        (int(r["source_id"]), int(r["target_id"]))
        for r in graph.edge_attrs().iter_rows(named=True)
    }
    pred_nodes = _normalise_nodes(pd.DataFrame(control_nodes_by_ds[dataset]))
    gt_nodes_n = _normalise_nodes(gt_nodes)
    p2g, _, _ = _node_match(pred_nodes, gt_nodes_n)
    ds_tp = ds_fp = ds_unk = 0
    for src, tgt in control_edges_by_ds[dataset] - pred_dc[dataset]:
        gs = p2g.get(src)
        gt = p2g.get(tgt)
        if gs is None or gt is None:
            ds_unk += 1
            removed_unknown += 1
            continue
        if (int(gs), int(gt)) in gt_edge_set:
            ds_tp += 1
            removed_tp += 1
        else:
            ds_fp += 1
            removed_fp += 1
    removed_by_ds[dataset] = {"tp": ds_tp, "fp": ds_fp, "unknown": ds_unk}

removed_class = {
    "tp": removed_tp,
    "fp": removed_fp,
    "unknown": removed_unknown,
    "by_dataset": removed_by_ds,
    "n_removed_edges": len(removed_edges),
}

control_summary = json.loads(control_summary_path.read_text())
summary = {
    **aggregate,
    "detection_threshold": float(config.inference["detection_threshold"]),
    "ensemble_alpha": float(config.inference["ensemble_alpha"]),
    "output_gap2_recovery": False,
    "use_deepcenter_veto": True,
    "deepcenter_gap_veto": True,
    "deepcenter_safe_div_veto": True,
    "checkpoint_path": "reuse_two_seed_det0_960_raw_geff",
    "deepcenter_checkpoint": str(bundle["path"]),
    "runtime_seconds": runtime,
    "control_score": control_score,
    "delta_vs_control": delta,
    "control_difference_material": abs(delta) > MATERIAL_DELTA,
    "deepcenter_gap_rejected_total": sum(
        v.get("deepcenter_gap_rejected", 0) for v in stats_by_dataset.values()
    ),
    "deepcenter_safe_div_rejected_total": sum(
        v.get("deepcenter_safe_div_rejected", 0) for v in stats_by_dataset.values()
    ),
    "deepcenter_gap_checked_total": sum(
        v.get("deepcenter_gap_checked", 0) for v in stats_by_dataset.values()
    ),
    "deepcenter_safe_div_checked_total": sum(
        v.get("deepcenter_safe_div_checked", 0) for v in stats_by_dataset.values()
    ),
    "n_edges_removed_vs_control": len(removed_edges),
}
manifest = {
    "schema_version": 1,
    "experiment": "two_seed_det0_960_deepcenter",
    "config": str(config_path),
    "control_raw_geff": str(control_raw),
    "note": "DeepCenter gating only; gap2 off; no re-inference; control untouched",
}
write_metric_outputs(out, rows, summary, manifest)

control_per = {
    row["dataset"]: float(row["adj_edge_jaccard"])
    for row in csv.DictReader(control_per_path.open(newline="", encoding="utf-8"))
}
compare_rows = []
n_improved = n_worsened = n_unchanged = 0
severe = []
for row in rows:
    ds = row["dataset"]
    score = float(row["adj_edge_jaccard"])
    base = control_per[ds]
    d = score - base
    if d > 1e-12:
        n_improved += 1
    elif d < -1e-12:
        n_worsened += 1
    else:
        n_unchanged += 1
    if d < -0.005:
        severe.append({"dataset": ds, "delta": d})
    compare_rows.append(
        {
            "dataset": ds,
            "control_score": base,
            "deepcenter_score": score,
            "delta": d,
            "edge_tp": row["edge_tp"],
            "edge_fp": row["edge_fp"],
            "edge_fn": row["edge_fn"],
            "division_tp": row["division_tp"],
            "division_fp": row["division_fp"],
            "division_fn": row["division_fn"],
        }
    )

edge_delta = {
    "tp": int(summary["edge_tp"]) - int(control_summary["edge_tp"]),
    "fp": int(summary["edge_fp"]) - int(control_summary["edge_fp"]),
    "fn": int(summary["edge_fn"]) - int(control_summary["edge_fn"]),
}
div_delta = {
    "tp": int(summary["division_tp"]) - int(control_summary["division_tp"]),
    "fp": int(summary["division_fp"]) - int(control_summary["division_fp"]),
    "fn": int(summary["division_fn"]) - int(control_summary["division_fn"]),
}

promote = False
interesting = False
reject_reason = None
if delta >= 0.001 and n_improved >= 5 and not severe:
    promote = True
elif delta > 0 and edge_delta["fp"] < 0 and not severe:
    interesting = True
    if delta >= 0.001 and n_improved >= 4:
        promote = True
elif delta < 0:
    reject_reason = "negative score delta"
elif severe and delta < 0.001:
    reject_reason = "severe dataset regression without compensating aggregate gain"
elif n_improved <= 2 and delta > 0:
    reject_reason = "strongly concentrated gain"
else:
    reject_reason = "does not meet promote criteria"

decision = {
    "promote": promote,
    "interesting": interesting,
    "reject_reason": reject_reason,
    "verdict": "PROMOTE" if promote else ("CONSIDER" if interesting and not promote else "REJECT"),
    "delta": delta,
    "n_improved": n_improved,
    "n_unchanged": n_unchanged,
    "n_worsened": n_worsened,
    "edge_delta": edge_delta,
    "division_delta": div_delta,
    "severe_regressions": severe,
}

(out / "deepcenter_postprocess_stats.json").write_text(
    json.dumps(stats_by_dataset, indent=2, sort_keys=True) + "\n"
)
with (out / "edges_removed_vs_control.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "source_id", "target_id"])
    w.writeheader()
    w.writerows(removed_edges)
(out / "removed_edges_tp_fp.json").write_text(
    json.dumps(removed_class, indent=2, sort_keys=True) + "\n"
)
with (out / "per_dataset_vs_control.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(compare_rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(compare_rows, key=lambda r: r["dataset"]))

comparison = {
    "control_score": control_score,
    "deepcenter_score": float(summary["score"]),
    "delta": delta,
    "control_edges": {
        "tp": int(control_summary["edge_tp"]),
        "fp": int(control_summary["edge_fp"]),
        "fn": int(control_summary["edge_fn"]),
    },
    "deepcenter_edges": {
        "tp": int(summary["edge_tp"]),
        "fp": int(summary["edge_fp"]),
        "fn": int(summary["edge_fn"]),
    },
    "control_divisions": {
        "tp": int(control_summary["division_tp"]),
        "fp": int(control_summary["division_fp"]),
        "fn": int(control_summary["division_fn"]),
    },
    "deepcenter_divisions": {
        "tp": int(summary["division_tp"]),
        "fp": int(summary["division_fp"]),
        "fn": int(summary["division_fn"]),
    },
    "per_dataset": compare_rows,
    "deepcenter_stats": stats_by_dataset,
    "removed_edges_classification": removed_class,
    "decision": decision,
}
(out / "comparison_vs_control.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
(out / "DONE").write_text("ok\n", encoding="utf-8")

print(f"score:      {float(summary['score']):.12f}")
print(f"control:    {control_score:.12f}")
print(f"delta:      {delta:+.12f}")
print(
    "edges tp/fp/fn:",
    f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}",
)
print(
    "div tp/fp/fn:",
    f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}",
)
print(f"datasets improved/unchanged/worsened: {n_improved}/{n_unchanged}/{n_worsened}")
print(
    "deepcenter rejected gap/safe_div:",
    summary["deepcenter_gap_rejected_total"],
    summary["deepcenter_safe_div_rejected_total"],
)
print(
    "deepcenter checked gap/safe_div:",
    summary["deepcenter_gap_checked_total"],
    summary["deepcenter_safe_div_checked_total"],
)
print(
    "removed vs control:",
    len(removed_edges),
    "tp/fp/unk",
    removed_tp,
    removed_fp,
    removed_unknown,
)
print("severe:", severe)
print(f"VERDICT {decision['verdict']} promote={promote} reason={reject_reason}")
print("OUT", out)
PY

echo DEEPCENTER_ABLATION_DONE
