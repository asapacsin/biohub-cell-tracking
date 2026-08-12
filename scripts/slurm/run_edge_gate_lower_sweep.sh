#!/usr/bin/env bash
# GPU: edge_threshold sweep 0.35 / 0.30 / 0.25 under motion-relink OFF.
# Control: motion-OFF edge=0.5 (fixed 0.906725 / holdout 0.960307).
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/edge_gate_lower_sweep.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
base_path = Path('configs/experiments/recipe_c_motion_relink_off_det0_96875.yaml')
for thr in (0.35, 0.30, 0.25):
    cfg = yaml.safe_load(base_path.read_text())
    cfg['inference']['edge_threshold'] = thr
    assert cfg['postprocessing']['output_motion_relink'] is False
    tag = f'{thr:.2f}'.replace('.', '_')
    path = Path(f'configs/experiments/edge_thresh_{tag}_motion_off_det0_96875.yaml')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('wrote', path, thr)
# remove misnamed 0.3 file if present
bad = Path('configs/experiments/edge_thresh_0_3_motion_off_det0_96875.yaml')
if bad.exists():
    bad.unlink()
    print('removed', bad)
PY

STAGE_INPUTS=()
for d in "${FIXED8[@]}" "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting lower edge-gate sweep at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-08:00:00 \
  --chdir=/tmp \
  bash -lc '
set -euo pipefail
DEST=/tmp/${USER}/biohub-edge-gate-lower
OUT_NFS=${HOME}/biohub-outputs/experiments/edge_gate_sweep_motion_off_v1
export OUT_NFS

rm -rf "$DEST"
mkdir -p "$DEST"
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

python scripts/run_edge_gate_lower_sweep.py
' 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/edge_gate_sweep_motion_off_v1"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:10:00 --mem=8G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/edge_gate_sweep_motion_off_v1 && \
  tar cf - DONE_LOWER_SWEEP lower_sweep_results.json \
    thresh_0_35/fixed8/summary.json thresh_0_35/fixed8/per_dataset.csv \
    thresh_0_35/holdout8/summary.json thresh_0_35/holdout8/per_dataset.csv \
    thresh_0_30/fixed8/summary.json thresh_0_30/fixed8/per_dataset.csv \
    thresh_0_30/holdout8/summary.json thresh_0_30/holdout8/per_dataset.csv \
    thresh_0_25/fixed8/summary.json thresh_0_25/fixed8/per_dataset.csv \
    thresh_0_25/holdout8/summary.json thresh_0_25/holdout8/per_dataset.csv \
    2>/dev/null
' | tar xf - -C "${SRC}/outputs/experiments/edge_gate_sweep_motion_off_v1"
echo "LOWER_SWEEP_OK $(date -Iseconds)" | tee -a "${LOG}"
cat "${SRC}/outputs/experiments/edge_gate_sweep_motion_off_v1/lower_sweep_results.json"
