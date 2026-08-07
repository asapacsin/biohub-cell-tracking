#!/usr/bin/env bash
# Stage exactly the fixed-8 train pairs onto GPU /tmp and run current V106 local CV.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
TRAIN_DIR="${SRC}/data/competition/train"
SUPPORT_DIR="${SRC}/data/support"
LOGIN_LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOGIN_LOG_DIR}"
LOG="${LOGIN_LOG_DIR}/v106_fixed8_cv.log"

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
[[ -d "${TRAIN_DIR}" ]] || { echo "Missing train directory: ${TRAIN_DIR}" >&2; exit 2; }
[[ -d "${SUPPORT_DIR}" ]] || { echo "Missing support directory: ${SUPPORT_DIR}" >&2; exit 2; }

STAGE_INPUTS=()
for dataset in "${FIXED8_DATASETS[@]}"; do
  for suffix in zarr geff; do
    path="data/competition/train/${dataset}.${suffix}"
    [[ -e "${SRC}/${path}" ]] || { echo "Missing fixed-8 input: ${SRC}/${path}" >&2; exit 2; }
    STAGE_INPUTS+=("${path}")
  done
done

echo "Starting V106 fixed-8 CV at $(date -Iseconds)" | tee "${LOG}"
echo "Staging ${#FIXED8_DATASETS[@]} zarr/geff pairs only" | tee -a "${LOG}"

tar \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='**/__pycache__' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='data/support/repo/predictions' \
  -cf - \
  README.md LICENSE pyproject.toml configs src data/support "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N 1 -n 1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=64G -t 0-08:00:00 \
    bash -lc '
      set -euo pipefail
      DEST="/tmp/${USER}/biohub-v106-fixed8"
      OUT_NFS="${HOME}/biohub-outputs/fixed8"
      echo host=$(hostname)
      nvidia-smi -L | head -1
      rm -rf "${DEST}"
      mkdir -p "${DEST}" "${OUT_NFS}"
      echo unpacking...
      tar xf - -C "${DEST}"
      cd "${DEST}"
      du -sh . data/competition/train data/support

      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate biohub
      python -m pip install -U pip
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
for name in ["biohub_pipeline", "tracksdata", "pyscipopt", "geff", "ilpy", "polars", "blosc2", "zarr"]:
    __import__(name)
    print("ok", name)
PY

      echo "=== fixed-8 current V106 ==="
      PYTHONUNBUFFERED=1 python -m biohub_pipeline.fixed8_cv \
        --config "${DEST}/configs/clean_v106.yaml" \
        --data-dir "${DEST}/data/competition/train" \
        --weights-dir "${DEST}/data/support" \
        --support-dir "${DEST}/data/support" \
        --output-dir "${DEST}/outputs/fixed8/current_v106"

      rm -rf "${OUT_NFS}/current_v106"
      cp -a outputs/fixed8/current_v106 "${OUT_NFS}/current_v106"
      date -Iseconds > "${OUT_NFS}/current_v106/DONE"
      echo "FIXED8_OK ${OUT_NFS}/current_v106"
      ls -la "${OUT_NFS}/current_v106"
    ' 2>&1 | tee -a "${LOG}"
