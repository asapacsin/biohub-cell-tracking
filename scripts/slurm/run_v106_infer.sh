#!/usr/bin/env bash
# Stage slim V106 inference bundle onto GPU /tmp and run clean_v106 pipeline.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOGIN_LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOGIN_LOG_DIR}"
LOG="${LOGIN_LOG_DIR}/v106_infer.log"

cd "${SRC}"
echo "Starting V106 GPU inference at $(date -Iseconds)" | tee "${LOG}"

# Slim tar: code + test zarrs + support pack (not full 82G train).
tar \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='**/__pycache__' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='data/competition/train' \
  --exclude='data/sample' \
  --exclude='artifacts' \
  --exclude='upstream_clean_v106' \
  -cf - \
  README.md LICENSE pyproject.toml configs src \
  data/competition/test data/competition/sample_submission.csv data/support \
| srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G -t 0-06:00:00 \
    bash -lc '
      set -euo pipefail
      DEST="/tmp/${USER}/biohub-v106"
      OUT_NFS="${HOME}/biohub-outputs/v106"
      echo host=$(hostname)
      nvidia-smi -L | head -1
      rm -rf "${DEST}"
      mkdir -p "${DEST}" "${OUT_NFS}"
      echo unpacking...
      tar xf - -C "${DEST}"
      cd "${DEST}"
      du -sh . data/competition/test data/support

      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate biohub
      python -m pip install -U pip
      # Offline wheels first; then remaining runtime deps.
      python -m pip install --no-deps data/support/wheels/*.whl || true
      python -m pip install \
        --find-links=data/support/wheels \
        --prefer-binary \
        tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" "ilpy>=0.5.1" \
        pyscipopt polars blosc2 dask imagecodecs pyarrow "rustworkx>=0.17.1" \
        "sqlalchemy>=2" "scikit-image>=0.24" "numcodecs>=0.13,<0.16" donfig
      export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"
      python - <<PY
import torch
assert torch.cuda.is_available(), "CUDA required"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
for m in ["biohub_pipeline","tracksdata","pyscipopt","geff","ilpy","polars","blosc2","zarr"]:
    __import__(m)
    print("ok", m)
PY

      echo "=== dry-run ==="
      PYTHONUNBUFFERED=1 python -m biohub_pipeline.run \
        --config "${DEST}/configs/clean_v106.yaml" \
        --data-dir "${DEST}/data/competition/test" \
        --weights-dir "${DEST}/data/support" \
        --support-dir "${DEST}/data/support" \
        --dry-run

      echo "=== full inference ==="
      mkdir -p outputs
      PYTHONUNBUFFERED=1 python -m biohub_pipeline.run \
        --config "${DEST}/configs/clean_v106.yaml" \
        --data-dir "${DEST}/data/competition/test" \
        --weights-dir "${DEST}/data/support" \
        --support-dir "${DEST}/data/support" \
        --output "${DEST}/outputs/submission_v106.csv"

      # Keep GEFF graphs on NFS so submission conversion can be retried without re-inference.
      rm -rf "${OUT_NFS}/predictions"
      cp -a data/support/repo/predictions "${OUT_NFS}/predictions" 2>/dev/null || true

      python - <<PY
import pandas as pd
from pathlib import Path
path = Path("outputs/submission_v106.csv")
df = pd.read_csv(path)
print("V106_OK", "rows", len(df), "nodes", int((df.row_type=="node").sum()), "edges", int((df.row_type=="edge").sum()), "datasets", df.dataset.nunique())
PY

      cp -f outputs/submission_v106.csv "${OUT_NFS}/submission_v106.csv"
      date -Iseconds > "${OUT_NFS}/DONE"
      echo "DONE ${OUT_NFS}"
      ls -la "${OUT_NFS}"
    ' 2>&1 | tee -a "${LOG}"
