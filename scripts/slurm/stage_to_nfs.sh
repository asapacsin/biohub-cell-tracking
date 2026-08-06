#!/usr/bin/env bash
# Stage login01-local project onto the NFS home visible to GPU nodes.
# Run from login01. Uses srun to unpack on a gpu_batch node.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
# Destination path as seen on GPU / NFS home.
DST_NAME="biohub-cell-tracking"
MARKER="/home/mc46451/${DST_NAME}/.nfs_stage_complete"

echo "Staging ${SRC} -> NFS home/${DST_NAME} via gpu_batch"
cd "$(dirname "${SRC}")"
PARENT="$(pwd)"
BASE="$(basename "${SRC}")"

# Exclude local venv and caches; GPU job builds its own conda env.
tar \
  --exclude="${BASE}/.venv" \
  --exclude="${BASE}/.git" \
  --exclude="${BASE}/.mypy_cache" \
  --exclude="${BASE}/.ruff_cache" \
  --exclude="${BASE}/.pytest_cache" \
  --exclude="${BASE}/**/__pycache__" \
  --exclude="${BASE}/logs/slurm/*.out" \
  --exclude="${BASE}/logs/slurm/*.err" \
  -cf - "${BASE}" \
| srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 -t 04:00:00 --mem=16G \
    bash -lc "
      set -euo pipefail
      cd \"\${HOME}\"
      rm -rf \"${DST_NAME}.partial\" \"${DST_NAME}\"
      mkdir -p \"${DST_NAME}.partial\"
      tar xf - -C \"${DST_NAME}.partial\"
      mv \"${DST_NAME}.partial/${BASE}\" \"${DST_NAME}\"
      rm -rf \"${DST_NAME}.partial\"
      date -Iseconds > \"${DST_NAME}/.nfs_stage_complete\"
      du -sh \"${DST_NAME}\" \"${DST_NAME}/data/competition\" || true
      echo STAGE_OK
    "

echo "Stage finished. Marker will be at NFS ${MARKER}"
