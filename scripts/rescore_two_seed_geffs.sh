#!/usr/bin/env bash
# Re-convert saved two-seed GEFFs after the safe_division gate fix and score vs control.
set -euo pipefail

DEST="${DEST:-/tmp/${USER}/biohub-twoseed-rescored}"
OUT="${OUT:-${HOME}/biohub-outputs/fixed8/two_seed_alpha_0_5}"
RAW="${OUT}/raw_geff"
REF_SCORE="${REF_SCORE:-0.87892959136423}"

cd "$DEST"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
export PYTHONPATH="${DEST}/src"

test -d "$RAW"
test "$(find "$RAW" -maxdepth 1 -name '*.geff' | wc -l)" -eq 8

python - <<PY
from __future__ import annotations

import json
import time
from pathlib import Path

from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    CURRENT_V106_REFERENCE_SCORE,
    MATERIAL_DELTA,
    evaluate_postprocessed_predictions,
    write_metric_outputs,
)
from biohub_pipeline.submission import write_submission_from_geff

out = Path("${OUT}")
raw = out / "raw_geff"
geffs = sorted(raw.glob("*.geff"))
assert len(geffs) == 8, geffs

config = load_config(Path("configs/clean_v106_two_seed.yaml"))
data_dir = Path("data/competition/train")
pred_csv = out / "predictions" / "postprocessed_submission.csv"

started = time.perf_counter()
summary_counts = write_submission_from_geff(geffs, config, data_dir, pred_csv)
print("CONVERT_OK", summary_counts)

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
delta = float(aggregate["score"]) - CURRENT_V106_REFERENCE_SCORE
summary = {
    **aggregate,
    "detection_threshold": float(config.inference["detection_threshold"]),
    "checkpoint_path": "two_seed_saved_geff_reconversion",
    "runtime_seconds": runtime,
    "reference_score": CURRENT_V106_REFERENCE_SCORE,
    "delta_vs_reference": delta,
    "reference_difference_material": abs(delta) > MATERIAL_DELTA,
    "repair": "safe_division_one_child_per_parent",
    "source_geffs": "raw_geff",
}
manifest = {
    "schema_version": 1,
    "config": "configs/clean_v106_two_seed.yaml",
    "note": "Re-converted saved two-seed GEFFs after safe_division used_sources gate",
    "artifacts": {
        "summary": "summary.json",
        "per_dataset": "per_dataset.csv",
        "predictions": "predictions/postprocessed_submission.csv",
        "raw_geff": "raw_geff",
    },
}
write_metric_outputs(out, rows, summary, manifest)
(out / "DONE").write_text("ok\n", encoding="utf-8")
print(f"score:      {float(summary['score']):.12f}")
print(f"reference:  {CURRENT_V106_REFERENCE_SCORE:.12f}")
print(f"delta:      {float(summary['delta_vs_reference']):+.12f}")
print(
    "edges tp/fp/fn:",
    f"{summary['edge_tp']} / {summary['edge_fp']} / {summary['edge_fn']}",
)
print(
    "div tp/fp/fn:",
    f"{summary['division_tp']} / {summary['division_fp']} / {summary['division_fn']}",
)
print(f"node recall: {float(summary['node_recall']):.12f}")
print("SUMMARY_PATH", out / "summary.json")
print(json.dumps({"score": summary["score"], "delta": summary["delta_vs_reference"]}, sort_keys=True))
PY

echo RESCORE_DONE
