#!/usr/bin/env bash
# Full control postprocess (incl. safe-div + short-track), then late-strip safe_division
# edges only — no second short-track pass, no re-inference.
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-late-strip-safe-div}"
CONTROL_RAW="${CONTROL_RAW:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/raw_geff}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_det0_960_late_strip_safe_div}"
CONTROL_SUMMARY="${CONTROL_SUMMARY:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/summary.json}"
CONTROL_PER="${CONTROL_PER:-${HOME}/biohub-outputs/fixed8/two_seed_det_thresh_0_960/per_dataset.csv}"
CONFIG_REL="configs/experiments/two_seed_det0_960_late_strip_safe_div.yaml"
CONTROL_SCORE="${CONTROL_SCORE:-0.8879786102989093}"
NO_SAFE_DIV_SCORE="${NO_SAFE_DIV_SCORE:-0.8883300756289333}"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

test -d "$CONTROL_RAW"
test "$(find "$CONTROL_RAW" -maxdepth 1 -name '*.geff' | wc -l)" -eq 8
test -f "$CONFIG_REL"

rm -rf "$OUT"
mkdir -p "$OUT/predictions" "$OUT/config"
cp -a "$CONFIG_REL" "$OUT/config/"

CONTROL_SCORE="$CONTROL_SCORE" NO_SAFE_DIV_SCORE="$NO_SAFE_DIV_SCORE" \
CONTROL_RAW="$CONTROL_RAW" OUT="$OUT" CONFIG_REL="$CONFIG_REL" \
CONTROL_SUMMARY="$CONTROL_SUMMARY" CONTROL_PER="$CONTROL_PER" \
python - <<'PY'
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    MATERIAL_DELTA,
    evaluate_postprocessed_predictions,
    write_metric_outputs,
)
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

control_score = float(os.environ["CONTROL_SCORE"])
no_safe_div_score = float(os.environ["NO_SAFE_DIV_SCORE"])
control_raw = Path(os.environ["CONTROL_RAW"])
out = Path(os.environ["OUT"])
config_path = Path(os.environ["CONFIG_REL"])
control_summary_path = Path(os.environ["CONTROL_SUMMARY"])
control_per_path = Path(os.environ["CONTROL_PER"])
data_dir = Path("data/competition/train")

config = load_config(config_path)
assert float(config.inference["detection_threshold"]) == 0.960
assert config.postprocessing["output_safe_divisions"] is True
assert config.postprocessing["output_gap2_recovery"] is False
assert config.postprocessing["use_deepcenter_veto"] is False

raw_dst = out / "raw_geff_source"
raw_dst.mkdir(parents=True, exist_ok=True)
for src in sorted(control_raw.glob("*.geff")):
    dest = raw_dst / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)

postprocessing.configure(config.postprocessing, data_dir)

started = time.perf_counter()
all_rows: list[dict] = []
identity: dict[str, dict] = {}
stripped_edges: list[dict] = []
stats_by_dataset: dict[str, dict] = {}

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

    # Full control pipeline including safe-div + short-track + linefit.
    nodes_ctrl, edges_ctrl, stats = postprocessing.filter_output_graph(
        {k: dict(v) for k, v in nodes.items()},
        [dict(e) for e in edges],
        dataset=dataset,
    )

    ctrl_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges_ctrl}
    safe_edges = [
        e for e in edges_ctrl if int(e.get("safe_division", 0) or 0) == 1
    ]
    safe_keys = {(int(e["source_id"]), int(e["target_id"])) for e in safe_edges}
    ordinary_keys = ctrl_keys - safe_keys

    # Late strip ONLY — no second short-track / prune / linefit.
    edges_strip = [
        e for e in edges_ctrl if int(e.get("safe_division", 0) or 0) != 1
    ]
    strip_keys = {(int(e["source_id"]), int(e["target_id"])) for e in edges_strip}
    # Keep all control nodes (including orphans from stripped edges) so ordinary
    # edge endpoints remain; do not re-prune or re-filter short tracks.
    nodes_strip = nodes_ctrl

    for e in safe_edges:
        stripped_edges.append(
            {
                "dataset": dataset,
                "source_id": int(e["source_id"]),
                "target_id": int(e["target_id"]),
                "distance_um": e.get("distance_um"),
            }
        )

    identity[dataset] = {
        "n_edges_control": len(ctrl_keys),
        "n_edges_after_late_strip": len(strip_keys),
        "n_safe_division_edges_stripped": len(safe_keys),
        "n_ordinary_edges_control": len(ordinary_keys),
        "ordinary_edges_bit_identical": strip_keys == ordinary_keys,
        "n_ordinary_missing": len(ordinary_keys - strip_keys),
        "n_extra_edges": len(strip_keys - ordinary_keys),
        "safe_division_candidates": int(stats.get("safe_division_candidates", 0)),
        "safe_divisions_added": int(stats.get("safe_divisions_added", 0)),
        "n_nodes_control": len(nodes_ctrl),
        "n_nodes_after_strip": len(nodes_strip),
    }
    stats_by_dataset[dataset] = {
        "safe_division_candidates": int(stats.get("safe_division_candidates", 0)),
        "safe_divisions_added": int(stats.get("safe_divisions_added", 0)),
        "safe_division_edges_stripped": len(safe_keys),
    }

    all_rows.extend(graph_rows(dataset, nodes_strip, edges_strip, len(all_rows)))

pred_csv = out / "predictions" / "postprocessed_submission.csv"
write_rows(all_rows, pred_csv)
print("CONVERT_OK", validate_submission_file(pred_csv))

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
delta = float(aggregate["score"]) - control_score
delta_vs_no_safe = float(aggregate["score"]) - no_safe_div_score
ordinary_all_identical = all(v["ordinary_edges_bit_identical"] for v in identity.values())
n_stripped = sum(v["n_safe_division_edges_stripped"] for v in identity.values())

control_summary = json.loads(control_summary_path.read_text())
summary = {
    **aggregate,
    "detection_threshold": 0.960,
    "ensemble_alpha": 0.5,
    "output_safe_divisions": True,
    "late_strip_safe_division_edges": True,
    "output_gap2_recovery": False,
    "use_deepcenter_veto": False,
    "checkpoint_path": "reuse_two_seed_det0_960_raw_geff",
    "runtime_seconds": runtime,
    "control_score": control_score,
    "delta_vs_control": delta,
    "no_safe_div_score": no_safe_div_score,
    "delta_vs_no_safe_div": delta_vs_no_safe,
    "control_difference_material": abs(delta) > MATERIAL_DELTA,
    "safe_division_edges_stripped_total": n_stripped,
    "ordinary_edges_bit_identical_to_control": ordinary_all_identical,
}
manifest = {
    "schema_version": 1,
    "experiment": "two_seed_det0_960_late_strip_safe_div",
    "config": str(config_path),
    "note": (
        "Control postprocess through short-track+linefit, then remove only "
        "safe_division-flagged edges; no second short-track pass."
    ),
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
            "late_strip_score": score,
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

goal_hit = (
    int(summary["division_fp"]) == 0
    and ordinary_all_identical
    and edge_delta["tp"] == 0
    and edge_delta["fn"] == 0
)
decision = {
    "delta_vs_control": delta,
    "delta_vs_no_safe_div": delta_vs_no_safe,
    "n_improved": n_improved,
    "n_unchanged": n_unchanged,
    "n_worsened": n_worsened,
    "edge_delta": edge_delta,
    "division_delta": div_delta,
    "severe_regressions": severe,
    "ordinary_edges_bit_identical": ordinary_all_identical,
    "safe_division_edges_stripped": n_stripped,
    "eliminated_div_fp_without_cascade": goal_hit,
    "verdict": (
        "SUCCESS_NO_CASCADE"
        if goal_hit
        else (
            "PARTIAL"
            if int(summary["division_fp"]) == 0 and ordinary_all_identical
            else "NEEDS_INSPECT"
        )
    ),
}

(out / "ordinary_edge_identity.json").write_text(
    json.dumps({"all_identical": ordinary_all_identical, "by_dataset": identity}, indent=2, sort_keys=True)
    + "\n"
)
with (out / "safe_division_edges_stripped.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset", "source_id", "target_id", "distance_um"])
    w.writeheader()
    w.writerows(stripped_edges)
(out / "late_strip_stats.json").write_text(
    json.dumps(stats_by_dataset, indent=2, sort_keys=True) + "\n"
)
with (out / "per_dataset_vs_control.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(compare_rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(compare_rows, key=lambda r: r["dataset"]))

comparison = {
    "control_score": control_score,
    "late_strip_score": float(summary["score"]),
    "no_safe_div_score": no_safe_div_score,
    "delta_vs_control": delta,
    "delta_vs_no_safe_div": delta_vs_no_safe,
    "control_edges": {"tp": 3885, "fp": 273, "fn": 218},
    "late_strip_edges": {
        "tp": int(summary["edge_tp"]),
        "fp": int(summary["edge_fp"]),
        "fn": int(summary["edge_fn"]),
    },
    "control_divisions": {"tp": 0, "fp": 15, "fn": 7},
    "late_strip_divisions": {
        "tp": int(summary["division_tp"]),
        "fp": int(summary["division_fp"]),
        "fn": int(summary["division_fn"]),
    },
    "per_dataset": compare_rows,
    "ordinary_edge_identity": identity,
    "decision": decision,
}
(out / "comparison_vs_control.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
(out / "DONE").write_text("ok\n", encoding="utf-8")

print(f"score:      {float(summary['score']):.12f}")
print(f"control:    {control_score:.12f}")
print(f"delta:      {delta:+.12f}")
print(f"vs_no_safe: {delta_vs_no_safe:+.12f}")
print(
    "edges tp/fp/fn:",
    f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}",
)
print(
    "div tp/fp/fn:",
    f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}",
)
print(f"datasets improved/unchanged/worsened: {n_improved}/{n_unchanged}/{n_worsened}")
print(f"safe_div edges stripped: {n_stripped}")
print(f"ordinary_edges_bit_identical: {ordinary_all_identical}")
print("severe:", severe)
print(f"VERDICT {decision['verdict']} goal_hit={goal_hit}")
print("OUT", out)
PY

echo LATE_STRIP_SAFE_DIV_DONE
