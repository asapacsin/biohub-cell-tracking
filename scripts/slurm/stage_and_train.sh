#!/usr/bin/env bash
# Stream project tarball into one GPU allocation, unpack on local /tmp, train, infer,
# and copy submission/models to NFS $HOME/biohub-outputs.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOGIN_LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOGIN_LOG_DIR}"
LOG="${LOGIN_LOG_DIR}/stage_and_train.log"

cd "$(dirname "${SRC}")"
BASE="$(basename "${SRC}")"

echo "Starting combined stage+train at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude="${BASE}/.venv" \
  --exclude="${BASE}/.git" \
  --exclude="${BASE}/.mypy_cache" \
  --exclude="${BASE}/.ruff_cache" \
  --exclude="${BASE}/.pytest_cache" \
  --exclude="${BASE}/**/__pycache__" \
  --exclude="${BASE}/outputs/architecture_blob_ilp" \
  --exclude="${BASE}/outputs/architecture_blob_learned" \
  --exclude="${BASE}/outputs/public_test_baseline" \
  --exclude="${BASE}/logs/slurm/*.out" \
  --exclude="${BASE}/logs/slurm/*.err" \
  -cf - "${BASE}" \
| srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G -t 1-12:00:00 \
    bash -lc "
      set -euo pipefail
      BASE='${BASE}'
      DEST=\"/tmp/\${USER}/\${BASE}\"
      OUT_NFS=\"\${HOME}/biohub-outputs\"
      echo host=\$(hostname)
      nvidia-smi -L | head -1
      mkdir -p \"/tmp/\${USER}\" \"\${OUT_NFS}\"
      rm -rf \"\${DEST}\"
      echo unpacking...
      tar xf - -C \"/tmp/\${USER}\"
      cd \"\${DEST}\"
      date -Iseconds > .stage_complete
      du -sh . data/competition
      bash scripts/slurm/remote_train_only.sh
    " 2>&1 | tee -a "${LOG}"
