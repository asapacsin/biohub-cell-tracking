#!/usr/bin/env bash
# Runs inside the GPU allocation after the project is unpacked under /tmp/$USER/...
set -euo pipefail

OUT_NFS="${HOME}/biohub-outputs"
mkdir -p "${OUT_NFS}/artifacts/detector" "${OUT_NFS}/submission" artifacts/detector \
  outputs/architecture_learned logs/slurm

CONDA_ROOT="${HOME}/miniconda3"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
ENV_NAME=biohub
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi
conda activate "${ENV_NAME}"
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -e ".[dev]"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
PY

python - <<'PY'
from pathlib import Path
import yaml
path = Path("configs/training.yaml")
raw = yaml.safe_load(path.read_text())
raw["training"]["detector"]["epochs"] = 8
raw["training"]["detector"]["num_workers"] = 4
raw["training"]["detector"]["batch_size"] = 2
raw["training"]["detector"]["device"] = "cuda"
raw["training"]["detector"]["seeds"] = [42]
path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
print("epochs", 8, "seeds", [42])
PY

echo "=== train detector ==="
PYTHONUNBUFFERED=1 biohub-track train-detector-ensemble \
  --competition-root data/competition \
  --config configs/training.yaml \
  --output-dir artifacts/detector

python - <<'PY'
from pathlib import Path
import yaml
models = sorted(Path("artifacts/detector").glob("seed_*.ts"))
if not models:
    raise SystemExit("missing detector artifacts")
cfg_path = Path("configs/architecture.yaml")
raw = yaml.safe_load(cfg_path.read_text())
raw["detection"]["model_paths"] = [str(p) for p in models]
raw["detection"]["device"] = "cuda"
raw["postprocessing"] = {
    "minimum_track_length": 2,
    "remove_isolated_short_tracks": True,
}
opt = raw.setdefault("association", {}).setdefault("optimizer", {})
opt["ilp_event_limit"] = 40000
cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
print("models", raw["detection"]["model_paths"])
PY

echo "=== learned inference ==="
PYTHONUNBUFFERED=1 biohub-track run \
  --competition-root data/competition \
  --config configs/architecture.yaml \
  --output outputs/architecture_learned

python - <<'PY'
import pandas as pd
from pathlib import Path
from biohub_tracker.submission import validate_submission
path = Path("outputs/architecture_learned/submission.csv")
df = pd.read_csv(path)
validate_submission(df, "data/competition")
print("LEARNED_OK", "nodes", int((df.row_type == "node").sum()), "edges", int((df.row_type == "edge").sum()))
PY

cp -f outputs/architecture_learned/submission.csv "${OUT_NFS}/submission/submission_learned.csv"
cp -f artifacts/detector/seed_*.ts "${OUT_NFS}/artifacts/detector/" 2>/dev/null || true
cp -f artifacts/detector/*.json "${OUT_NFS}/artifacts/detector/" 2>/dev/null || true
cp -f artifacts/association/model.json "${OUT_NFS}/artifacts/" 2>/dev/null || true
date -Iseconds > "${OUT_NFS}/DONE"
echo "DONE results in ${OUT_NFS}"
ls -la "${OUT_NFS}/submission"
