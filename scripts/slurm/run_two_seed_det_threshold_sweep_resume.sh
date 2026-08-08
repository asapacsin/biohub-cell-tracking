#!/usr/bin/env bash
# Resume two-seed det-threshold sweep, skipping thresholds that already have NFS DONE+summary.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
TRAIN_DIR="${SRC}/data/competition/train"
SUPPORT_DIR="${SRC}/data/support"
LOGIN_LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOGIN_LOG_DIR}"
LOG="${LOGIN_LOG_DIR}/two_seed_det_threshold_sweep_resume.log"

THRESHOLDS=(0.955 0.960 0.965 0.96875 0.9725 0.975 0.980)

FIXED8_DATASETS=(
  44b6_0113de3b
  44b6_0b24845f
  44b6_341df25f
  44b6_e57ff5c6
  6bba_05b6850b
  6bba_05db0fb1
  6bba_969618f6
  6bba_fc83837d
)

cd "${SRC}"
STAGE_INPUTS=()
for dataset in "${FIXED8_DATASETS[@]}"; do
  for suffix in zarr geff; do
    STAGE_INPUTS+=("data/competition/train/${dataset}.${suffix}")
  done
done

echo "Resume sweep at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='**/__pycache__' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='data/support/repo/predictions' \
  --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src data/support "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G -t 0-08:00:00 \
    bash -lc '
      set -euo pipefail
      DEST="/tmp/${USER}/biohub-twoseed-det-sweep"
      OUT_NFS="${HOME}/biohub-outputs/fixed8"
      SWEEP_NFS="${OUT_NFS}/two_seed_det_threshold_sweep"
      echo host=$(hostname)
      nvidia-smi -L | head -1
      rm -rf "${DEST}"
      mkdir -p "${DEST}" "${OUT_NFS}" "${SWEEP_NFS}"
      tar xf - -C "${DEST}"
      cd "${DEST}"

      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate biohub
      python -m pip install -U pip >/dev/null
      python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
      python -m pip install \
        --find-links=data/support/wheels \
        --prefer-binary \
        tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" "ilpy>=0.5.1" \
        pyscipopt polars blosc2 dask imagecodecs pyarrow "rustworkx>=0.17.1" \
        "sqlalchemy>=2" "scikit-image>=0.24" "numcodecs>=0.13,<0.16" donfig >/dev/null
      export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"

      python - <<'"'"'PY'"'"'
import hashlib
from pathlib import Path
import torch
assert torch.cuda.is_available()
seed1 = Path("data/support/weights/unet_transformer/split_0/edge_predictor_best.pth")
seed2 = Path("data/support/weights/unet_transformer/seed_314159/edge_predictor_best.pth")
assert hashlib.sha256(seed1.read_bytes()).hexdigest() == "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
assert hashlib.sha256(seed2.read_bytes()).hexdigest() == "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
print("hashes_ok", torch.cuda.get_device_name(0))
PY

      THRESHOLDS=(0.955 0.960 0.965 0.96875 0.9725 0.975 0.980)
      FAILED=0
      for t in "${THRESHOLDS[@]}"; do
        slug="${t//./_}"
        cfg="configs/sweeps/two_seed_det_thresh_${slug}.yaml"
        out_local="outputs/fixed8/two_seed_det_thresh_${slug}"
        out_nfs="${OUT_NFS}/two_seed_det_thresh_${slug}"
        if [[ -f "${out_nfs}/summary.json" && -f "${out_nfs}/DONE" ]]; then
          echo "=== SKIP threshold=${t} (already done) ==="
          continue
        fi
        echo "=== threshold=${t} slug=${slug} ==="
        rm -rf data/support/repo/predictions "${out_local}"
        set +e
        PYTHONUNBUFFERED=1 python -m biohub_pipeline.fixed8_cv \
          --config "${DEST}/${cfg}" \
          --data-dir "${DEST}/data/competition/train" \
          --weights-dir "${DEST}/data/support" \
          --support-dir "${DEST}/data/support" \
          --output-dir "${DEST}/${out_local}"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          echo "THRESHOLD_FAIL t=${t} rc=${rc}"
          FAILED=1
          mkdir -p "${out_nfs}"
          echo "failed rc=${rc}" > "${out_nfs}/FAILED"
          continue
        fi
        rm -rf "${out_nfs}"
        cp -a "${out_local}" "${out_nfs}"
        date -Iseconds > "${out_nfs}/DONE"
        echo "THRESHOLD_OK t=${t} ${out_nfs}"
        SUMMARY_JSON="${out_local}/summary.json" python - <<'"'"'PY'"'"'
import json, os
from pathlib import Path
s = json.loads(Path(os.environ["SUMMARY_JSON"]).read_text())
print("score={:.12f} edges={}/{}/{} div={}/{}/{}".format(
    s["score"], s["edge_tp"], s["edge_fp"], s["edge_fn"],
    s["division_tp"], s["division_fp"], s["division_fn"]))
PY
      done

      echo "=== writing comparison ==="
      python - <<'"'"'PY'"'"'
import csv
import json
from pathlib import Path

baseline_score = 0.8847464271589631
thresholds = ["0.955", "0.960", "0.965", "0.96875", "0.9725", "0.975", "0.980"]
out_nfs = Path.home() / "biohub-outputs" / "fixed8"
sweep = out_nfs / "two_seed_det_threshold_sweep"
sweep.mkdir(parents=True, exist_ok=True)
rows = []
for t in thresholds:
    slug = t.replace(".", "_")
    summary_path = out_nfs / f"two_seed_det_thresh_{slug}" / "summary.json"
    per_path = out_nfs / f"two_seed_det_thresh_{slug}" / "per_dataset.csv"
    if not summary_path.is_file():
        rows.append({
            "detection_threshold": float(t),
            "slug": slug,
            "status": "failed",
            "score": None,
            "delta_vs_two_seed_baseline": None,
            "edge_tp": None, "edge_fp": None, "edge_fn": None,
            "division_tp": None, "division_fp": None, "division_fn": None,
            "node_recall": None,
            "summary_path": str(summary_path),
            "per_dataset_path": str(per_path),
        })
        continue
    s = json.loads(summary_path.read_text())
    rows.append({
        "detection_threshold": float(t),
        "slug": slug,
        "status": "ok",
        "score": float(s["score"]),
        "delta_vs_two_seed_baseline": float(s["score"]) - baseline_score,
        "edge_tp": int(s["edge_tp"]),
        "edge_fp": int(s["edge_fp"]),
        "edge_fn": int(s["edge_fn"]),
        "division_tp": int(s["division_tp"]),
        "division_fp": int(s["division_fp"]),
        "division_fn": int(s["division_fn"]),
        "node_recall": float(s["node_recall"]),
        "summary_path": str(summary_path),
        "per_dataset_path": str(per_path),
        "artifacts_dir": str(out_nfs / f"two_seed_det_thresh_{slug}"),
    })

ok_rows = [r for r in rows if r["status"] == "ok"]
ok_rows_sorted = sorted(ok_rows, key=lambda r: r["score"], reverse=True)
best = ok_rows_sorted[0] if ok_rows_sorted else None

consistency = {}
baseline_per = out_nfs / "two_seed_alpha_0_5" / "per_dataset.csv"
ref_label = "two_seed_alpha_0_5"
if not baseline_per.is_file():
    baseline_per = out_nfs / "two_seed_det_thresh_0_96875" / "per_dataset.csv"
    ref_label = "sweep_0.96875"
if best and baseline_per.is_file():
    import pandas as pd
    best_per = pd.read_csv(out_nfs / f"two_seed_det_thresh_{best['slug']}" / "per_dataset.csv")
    base_df = pd.read_csv(baseline_per)
    merged = base_df[["dataset", "adj_edge_jaccard"]].merge(
        best_per[["dataset", "adj_edge_jaccard"]], on="dataset", suffixes=("_base", "_best")
    )
    merged["delta"] = merged["adj_edge_jaccard_best"] - merged["adj_edge_jaccard_base"]
    consistency = {
        "reference": ref_label,
        "best_threshold": best["detection_threshold"],
        "per_dataset_delta": {k: float(v) for k, v in merged.set_index("dataset")["delta"].items()},
        "n_improved": int((merged["delta"] > 0).sum()),
        "n_worsened": int((merged["delta"] < 0).sum()),
        "n_unchanged": int((merged["delta"] == 0).sum()),
        "max_abs_dataset": str(merged.loc[merged["delta"].abs().idxmax(), "dataset"]),
        "max_abs_delta": float(merged.loc[merged["delta"].abs().idxmax(), "delta"]),
    }

comparison = {
    "schema_version": 1,
    "experiment": "two_seed_det_threshold_sweep",
    "ensemble_alpha": 0.5,
    "two_seed_baseline_score": baseline_score,
    "two_seed_baseline_path": str(out_nfs / "two_seed_alpha_0_5"),
    "note": "Full re-inference required; detection_threshold applied before GEFF generation. Existing GEFFs cannot be rescored.",
    "rows_by_threshold": rows,
    "rows_sorted_by_score": ok_rows_sorted,
    "best": best,
    "consistency_vs_baseline": consistency,
}
(sweep / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
fields = [
    "detection_threshold", "status", "score", "delta_vs_two_seed_baseline",
    "edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
    "node_recall", "artifacts_dir",
]
with (sweep / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in sorted(ok_rows, key=lambda x: -x["score"]) + [r for r in rows if r["status"] != "ok"]:
        writer.writerow(r)
print("COMPARISON", sweep / "comparison.json")
if best:
    print("BEST threshold={} score={:.12f} delta={:+.12f}".format(
        best["detection_threshold"], best["score"], best["delta_vs_two_seed_baseline"]))
else:
    print("BEST none")
PY

      date -Iseconds > "${SWEEP_NFS}/DONE"
      echo "SWEEP_DONE failed=${FAILED}"
      exit ${FAILED}
    ' 2>&1 | tee -a "${LOG}"
