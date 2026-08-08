#!/usr/bin/env bash
# Exact-as-possible public Pilkwang dual-seed recipe on fixed-8 (full inference).
# Patch order matches notebook: spatial D4 TTA, then dual-seed ensemble replacements.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/public_two_seed_exact_fixed8.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting public recipe fixed-8 at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support data/deepcenter \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-04:00:00 \
  --chdir=/tmp \
  bash -lc '
set -euo pipefail
DEST=/tmp/${USER}/biohub-public-recipe-fixed8
OUT_NFS=${HOME}/biohub-outputs/fixed8/public_two_seed_exact
rm -rf "$DEST"
mkdir -p "$DEST" "$OUT_NFS"
tar xf - -C "$DEST"
cd "$DEST"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" "ilpy>=0.5.1" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow "rustworkx>=0.17.1" \
  "sqlalchemy>=2" "scikit-image>=0.24" "numcodecs>=0.13,<0.16" donfig >/dev/null
export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'"'"'PY'"'"'
import hashlib, json, shutil
from pathlib import Path
from biohub_pipeline.inference import apply_spatial_d4_patch

seed1 = Path("data/support/weights/unet_transformer/split_0/edge_predictor_best.pth")
seed2 = Path("data/support/weights/unet_transformer/seed_314159/edge_predictor_best.pth")
dc = Path("data/deepcenter/weights/full_frame_center/checkpoint_last.pt")
assert hashlib.sha256(seed1.read_bytes()).hexdigest() == "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
assert hashlib.sha256(seed2.read_bytes()).hexdigest() == "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
assert dc.is_file()

# Fresh predictor from support pack (avoid leftover patches)
repo = Path("data/support/repo")
pred = repo / "scripts/predict_unet_transformer.py"
# If a backup exists restore; otherwise we patch in place on staged copy
assert pred.is_file()

# 1) D4 TTA (notebook order; dual-seed replacements require /_nv preimage)
apply_spatial_d4_patch(repo, "scripts/predict_unet_transformer.py")
text0 = pred.read_text()
if "for _k in (1, 3):" not in text0:
    raise RuntimeError("D4 patch missing after apply_spatial_d4_patch")

# 2) Public dual-seed string replacements (require D4 /_nv preimage)
text = pred.read_text()
pairs = json.loads(Path("configs/audit/public_dual_seed_ensemble_replacements.json").read_text())
for i, item in enumerate(pairs):
    old, new = item["old"], item["new"]
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"replacement {i}: expected 1 match, found {n}")
    text = text.replace(old, new, 1)
pred.write_text(text)
import py_compile
py_compile.compile(str(pred), doraise=True)
print("public_dual_seed_patch_ok", len(pairs), flush=True)
PY

export BIOHUB_SECONDARY_WEIGHTS="${DEST}/data/support/weights/unet_transformer/seed_314159/edge_predictor_best.pth"
export BIOHUB_SECONDARY_EDGE_WEIGHT=0.15
export BIOHUB_SECONDARY_DETECTION_WEIGHT=0.475
export BIOHUB_SECONDARY_LINK_MODE=low_margin_consensus
export BIOHUB_SECONDARY_MIX_TEMPERATURE=1
export BIOHUB_SECONDARY_LOW_MARGIN_MAX=0.35
export BIOHUB_DUAL_SEED_EDGE_THRESHOLD=0.48

rm -rf data/support/repo/predictions

python - <<'"'"'PY'"'"'
import csv, json, os, shutil, time
from collections import defaultdict
from pathlib import Path

import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    CURRENT_V106_REFERENCE_SCORE,
    MATERIAL_DELTA,
    evaluate_postprocessed_predictions,
    find_fixed8_prediction_geffs,
    write_metric_outputs,
    _copy_raw_predictions,
)
from biohub_pipeline.inference import run_prediction
from biohub_pipeline.submission import (
    collect_outdegree_violations,
    format_outdegree_violation_report,
    graph_rows,
    validate_submission_file,
    write_rows,
)

out = Path("outputs/fixed8/public_two_seed_exact")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
cfg_path = Path("configs/experiments/public_two_seed_exact.yaml")
config = load_config(cfg_path)
assert config.inference.get("ensemble_weights_relative") is None
assert float(config.inference["detection_threshold"]) == 0.96875
assert float(config.inference["ilp_disappearance_weight"]) == 1.5
assert config.postprocessing["use_deepcenter_veto"] is True
assert config.postprocessing["deepcenter_gap_veto"] is True
assert config.postprocessing["deepcenter_safe_div_veto"] is False
assert config.postprocessing["output_safe_divisions"] is True
assert config.postprocessing["output_gap2_recovery"] is False
assert config.postprocessing["adaptive_short_track_rescue"] is False

data_dir = Path("data/competition/train")
repo = Path("data/support/repo")
weights = Path("data/support/weights/unet_transformer/split_0/edge_predictor_best.pth")

splits = repo / "clean_v106_test_splits.json"
splits.write_text(json.dumps([{"split": 0, "train": [], "test": list(FIXED8_DATASETS)}], indent=2))
cmd = [
    "python",
    "scripts/predict_unet_transformer.py",
    "--data-dir", str(data_dir.resolve()),
    "--splits", splits.name,
    "--split", "0",
    "--weights", os.path.relpath(weights.resolve(), repo.resolve()),
    "--unet-batch-size", "4",
    "--det-threshold", "0.96875",
    "--ilp-edge-weight", "-1.0",
    "--ilp-appearance-weight", "0.0",
    "--ilp-disappearance-weight", "1.5",
    "--ilp-division-weight", "1.0",
    "--use-ilp",
]
print("CMD", " ".join(cmd), flush=True)
for k in sorted(os.environ):
    if k.startswith("BIOHUB_"):
        print(f"ENV {k}={os.environ[k]}", flush=True)

started = time.perf_counter()
run_prediction(cmd, repo)
geffs = find_fixed8_prediction_geffs(repo, "unet_transformer")
_copy_raw_predictions(geffs, out, config_path=cfg_path)
(out / "config").mkdir(exist_ok=True)
shutil.copy2(cfg_path, out / "config" / cfg_path.name)

postprocessing.configure(config.postprocessing, data_dir)
bundle = postprocessing.load_deepcenter_veto_detector()
assert bundle is not None, "DeepCenter required for public recipe"

pred_csv = out / "predictions" / "postprocessed_submission.csv"
all_rows = []
stats_by_ds = {}
for geff_path in sorted(geffs):
    dataset = geff_path.stem
    loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    nodes = {
        int(r["node_id"]): {
            "node_id": int(r["node_id"]),
            "t": int(r["t"]),
            "z": float(r["z"]),
            "y": float(r["y"]),
            "x": float(r["x"]),
        }
        for r in graph.node_attrs().iter_rows(named=True)
    }
    edges = []
    for r in graph.edge_attrs().iter_rows(named=True):
        p = r.get("edge_prob") if hasattr(r, "get") else None
        edges.append({
            "source_id": int(r["source_id"]),
            "target_id": int(r["target_id"]),
            "edge_prob": None if p is None else float(p),
        })
    nodes, edges, stats = postprocessing.filter_output_graph(
        nodes, edges, dataset=dataset, deepcenter_bundle=bundle
    )
    stats_by_ds[dataset] = {k: int(v) if isinstance(v, (int, float)) else v for k, v in stats.items()}
    all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))

write_rows(all_rows, pred_csv)
validate_submission_file(pred_csv)
(out / "predictions" / "postprocess_stats.json").write_text(
    json.dumps(stats_by_ds, indent=2, sort_keys=True) + "\n"
)

rows, aggregate = evaluate_postprocessed_predictions(pred_csv, data_dir, config)
runtime = time.perf_counter() - started
summary = {
    **aggregate,
    "detection_threshold": 0.96875,
    "recipe": "public_two_seed_exact",
    "secondary_detection_weight": 0.475,
    "secondary_edge_weight": 0.15,
    "secondary_link_mode": "low_margin_consensus",
    "dual_seed_edge_threshold": 0.48,
    "ilp_disappearance_weight": 1.5,
    "runtime_seconds": runtime,
    "control_det960_score": 0.887978610299,
    "control_safeoff_score": 0.888330075629,
    "delta_vs_det960": float(aggregate["score"]) - 0.887978610299,
    "delta_vs_safeoff": float(aggregate["score"]) - 0.888330075629,
    "reference_score": CURRENT_V106_REFERENCE_SCORE,
    "delta_vs_reference": float(aggregate["score"]) - CURRENT_V106_REFERENCE_SCORE,
    "reference_difference_material": abs(float(aggregate["score"]) - CURRENT_V106_REFERENCE_SCORE) > MATERIAL_DELTA,
}
manifest = {
    "schema_version": 1,
    "experiment": "public_two_seed_exact",
    "config": str(cfg_path),
    "note": "Public Pilkwang dual-seed (D4 + BIOHUB_SECONDARY_* + DeepCenter gap gate); mapped postprocess",
    "prediction_command": cmd,
    "env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("BIOHUB_")},
}
write_metric_outputs(out, rows, summary, manifest)
(out / "DONE").write_text("ok\n")
print(f"score: {float(summary["score"]):.12f}")
print(f"edges: {summary["edge_tp"]}/{summary["edge_fp"]}/{summary["edge_fn"]}")
print(f"divs: {summary["division_tp"]}/{summary["division_fp"]}/{summary["division_fn"]}")
print(f"delta_vs_det960: {summary["delta_vs_det960"]:+.12f}")
print(f"delta_vs_safeoff: {summary["delta_vs_safeoff"]:+.12f}")
PY

rm -rf "$OUT_NFS"
cp -a outputs/fixed8/public_two_seed_exact "$OUT_NFS"
# also mirror into login workspace if mounted differently — handled by caller
date -Iseconds > "$OUT_NFS/DONE"
echo PUBLIC_RECIPE_FIXED8_DONE "$OUT_NFS"
' 2>&1 | tee -a "${LOG}"

# Mirror to workspace outputs if NFS copy succeeded
NFS="${HOME}/biohub-outputs/fixed8/public_two_seed_exact"
if [[ -f "${NFS}/summary.json" ]]; then
  mkdir -p "${SRC}/outputs/fixed8"
  rm -rf "${SRC}/outputs/fixed8/public_two_seed_exact"
  cp -a "${NFS}" "${SRC}/outputs/fixed8/public_two_seed_exact"
fi
