#!/usr/bin/env bash
# GPU: candidate-edge bottleneck diagnostic on independent holdout-8.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/candidate_bottleneck_holdout8.log"

HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting candidate bottleneck holdout-8 at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-03:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-candidate-bottleneck-holdout
OUT_NFS=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8
rm -rf \"\$DEST\"
mkdir -p \"\$DEST\"
tar xf - -C \"\$DEST\"
cd \"\$DEST\"

source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr \"geff>=1.1.3.1.1\" \"geff-spec<1.2\" \"ilpy>=0.5.1\" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow \"rustworkx>=0.17.1\" \
  \"sqlalchemy>=2\" \"scikit-image>=0.24\" \"numcodecs>=0.13,<0.16\" donfig >/dev/null
export PYTHONPATH=\"\${DEST}/src\${PYTHONPATH:+:\$PYTHONPATH}\"

if [[ -e \"\$OUT_NFS\" ]]; then
  echo \"ERROR: output already exists: \$OUT_NFS\" >&2
  exit 2
fi

OUT_LOCAL=outputs/experiments/candidate_edge_bottleneck_holdout8
WORK_LOCAL=outputs/experiments/candidate_edge_bottleneck_holdout8_work
rm -rf \"\$OUT_LOCAL\" \"\$WORK_LOCAL\"

set +e
python scripts/run_candidate_bottleneck_experiment.py \
  --data-dir data/competition/train \
  --support-dir data/support \
  --config configs/sweeps/two_seed_det_thresh_0_96875.yaml \
  --output-dir \"\$OUT_LOCAL\" \
  --work-dir \"\$WORK_LOCAL\" \
  --top-k 16 \
  --skip-fixed8-validation \
  --control-score 0.9590130516216544 \
  --datasets \
    44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d \
    6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
status=\$?
set -e

mkdir -p \"\$(dirname \"\$OUT_NFS\")\"
if [[ -d \"\$OUT_LOCAL\" ]]; then
  rm -rf \"\$OUT_NFS\"
  cp -a \"\$OUT_LOCAL\" \"\$OUT_NFS\"
  if [[ \$status -eq 0 ]]; then
    date -Iseconds > \"\$OUT_NFS/DONE\"
  else
    date -Iseconds > \"\$OUT_NFS/PARTIAL_FAIL\"
  fi
  echo HOLDOUT_BOTTLENECK_COPIED \"\$OUT_NFS\" status=\$status
fi
if [[ \$status -eq 0 ]]; then
  python - <<'PY'
import json
from pathlib import Path
decision = json.loads(Path(\"outputs/experiments/candidate_edge_bottleneck_holdout8/decision.json\").read_text())
score = json.loads(Path(\"outputs/experiments/candidate_edge_bottleneck_holdout8/score_summary.json\").read_text())
print(\"score\", score.get(\"score\"))
print(\"classification\", decision.get(\"classification\"))
print(\"shares\", decision.get(\"mechanism_shares\"))
print(\"next\", decision.get(\"recommended_next_action\"))
PY
  echo HOLDOUT_BOTTLENECK_DONE \"\$OUT_NFS\"
fi
exit \$status
" 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/candidate_edge_bottleneck_holdout8"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:10:00 --mem=8G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8 && \
  tar cf - \
    DONE decision.json score_summary.json metadata.json report.md \
    cause_summary.csv cause_by_dataset.csv candidate_edge_diagnostic.csv \
    metric_by_dataset.csv experiment_config.yaml command.txt \
    predictions/postprocessed_submission.csv 2>/dev/null
' | tar xf - -C "${SRC}/outputs/experiments/candidate_edge_bottleneck_holdout8" || true
echo "LOGIN_HOLDOUT_MIRROR_DONE"
