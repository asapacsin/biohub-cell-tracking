#!/bin/bash
#SBATCH --job-name=biohub-train
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# NFS home path (GPU nodes). Must exist before this job starts.
PROJECT="${PROJECT_ROOT:-${HOME}/biohub-cell-tracking}"
cd "${PROJECT}"
mkdir -p logs/slurm artifacts/detector outputs/architecture_learned

echo "host=$(hostname) cuda=$(nvidia-smi -L | head -1)"
echo "project=${PROJECT}"

CONDA_ROOT="${HOME}/miniconda3"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

ENV_NAME="${BIOHUB_CONDA_ENV:-biohub}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi
conda activate "${ENV_NAME}"

python -m pip install -U pip
# Driver on gpu_batch is new enough for cu124 wheels; 2080 Ti = sm_75.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -e ".[dev]"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

# Faster first learned artifact; override via TRAIN_EPOCHS if needed.
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
python - <<'PY'
from pathlib import Path
import yaml
epochs = int(__import__("os").environ.get("TRAIN_EPOCHS", "8"))
path = Path("configs/training.yaml")
raw = yaml.safe_load(path.read_text())
raw["training"]["detector"]["epochs"] = epochs
raw["training"]["detector"]["num_workers"] = 4
raw["training"]["detector"]["batch_size"] = 2
raw["training"]["detector"]["device"] = "cuda"
# Single seed first for turnaround; ensemble can be re-run later.
raw["training"]["detector"]["seeds"] = [42]
path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
print("training epochs", epochs, "seeds", raw["training"]["detector"]["seeds"])
PY

echo "=== train detector ==="
PYTHONUNBUFFERED=1 biohub-track train-detector-ensemble \
  --competition-root data/competition \
  --config configs/training.yaml \
  --output-dir artifacts/detector

# Point architecture config at available TorchScript artifacts.
python - <<'PY'
from pathlib import Path
import yaml
models = sorted(Path("artifacts/detector").glob("seed_*.ts"))
if not models:
    raise SystemExit("No detector TorchScript artifacts found")
cfg_path = Path("configs/architecture.yaml")
raw = yaml.safe_load(cfg_path.read_text())
raw["detection"]["model_paths"] = [str(p) for p in models]
raw["detection"]["device"] = "cuda"
raw["postprocessing"] = {
    "minimum_track_length": 2,
    "remove_isolated_short_tracks": True,
}
raw["association"]["optimizer"]["ilp_event_limit"] = 40000
cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
print("using models", raw["detection"]["model_paths"])
PY

echo "=== learned architecture inference ==="
PYTHONUNBUFFERED=1 biohub-track run \
  --competition-root data/competition \
  --config configs/architecture.yaml \
  --output outputs/architecture_learned

# Stage Kaggle-facing copy on NFS home.
mkdir -p outputs/kaggle_submission
cp -f outputs/architecture_learned/submission.csv outputs/kaggle_submission/submission_learned.csv
# Keep classical as default submit name if present; else use learned.
if [[ -f outputs/kaggle_submission/submission.csv ]]; then
  cp -f outputs/architecture_learned/submission.csv outputs/kaggle_submission/submission_learned_latest.csv
else
  cp -f outputs/architecture_learned/submission.csv outputs/kaggle_submission/submission.csv
fi

python - <<'PY'
import pandas as pd
from pathlib import Path
from biohub_tracker.submission import validate_submission
path = Path("outputs/architecture_learned/submission.csv")
df = pd.read_csv(path)
validate_submission(df, "data/competition")
nodes = (df.row_type == "node").sum()
edges = (df.row_type == "edge").sum()
print(f"LEARNED_OK nodes={nodes} edges={edges} path={path}")
PY

echo "DONE"
