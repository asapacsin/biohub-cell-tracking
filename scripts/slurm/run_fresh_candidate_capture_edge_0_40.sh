#!/usr/bin/env bash
# Fresh instrumented candidate-edge capture under promoted recipe:
# motion_relink OFF, edge_threshold=0.40, det=0.96875 two-seed.
# Runs fixed-8 then holdout-8 sequentially on one GPU; new NFS dirs only.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/fresh_candidate_capture_edge_0_40.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}" "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting fresh candidate capture edge_0_40 at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-06:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-fresh-capture-edge040
OUT_ROOT=\${HOME}/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1
FIXED_NFS=\${OUT_ROOT}/fixed8
HOLD_NFS=\${OUT_ROOT}/holdout8
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

if [[ -e \"\$OUT_ROOT\" ]]; then
  echo \"ERROR: output already exists: \$OUT_ROOT\" >&2
  exit 2
fi
mkdir -p \"\$OUT_ROOT\"

CFG=configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml
python - <<'PY'
from pathlib import Path
from biohub_pipeline.config import load_config
cfg = load_config(Path('configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml'))
assert abs(float(cfg.inference['edge_threshold']) - 0.4) < 1e-12, cfg.inference.get('edge_threshold')
assert cfg.postprocessing['output_motion_relink'] is False
print('RECIPE_OK edge_threshold=', cfg.inference['edge_threshold'],
      'motion_relink=', cfg.postprocessing['output_motion_relink'],
      'det=', cfg.inference['detection_threshold'])
PY

FIXED_LOCAL=outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/fixed8
HOLD_LOCAL=outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8
mkdir -p outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1

run_one() {
  local name=\"\$1\"
  local out_local=\"\$2\"
  local out_nfs=\"\$3\"
  local control=\"\$4\"
  shift 4
  local work_local=\"\${out_local}_work\"
  rm -rf \"\$out_local\" \"\$work_local\"
  set +e
  if [[ \"\$name\" == \"fixed8\" ]]; then
    python scripts/run_candidate_bottleneck_experiment.py \\
      --data-dir data/competition/train \\
      --support-dir data/support \\
      --config \"\$CFG\" \\
      --output-dir \"\$out_local\" \\
      --work-dir \"\$work_local\" \\
      --top-k 16 \\
      --control-score \"\$control\"
  else
    python scripts/run_candidate_bottleneck_experiment.py \\
      --data-dir data/competition/train \\
      --support-dir data/support \\
      --config \"\$CFG\" \\
      --output-dir \"\$out_local\" \\
      --work-dir \"\$work_local\" \\
      --top-k 16 \\
      --skip-fixed8-validation \\
      --control-score \"\$control\" \\
      --datasets \"\$@\"
  fi
  local status=\$?
  set -e
  if [[ -d \"\$out_local\" ]]; then
    rm -rf \"\$out_nfs\"
    cp -a \"\$out_local\" \"\$out_nfs\"
    if [[ \$status -eq 0 ]]; then
      date -Iseconds > \"\$out_nfs/DONE\"
    else
      date -Iseconds > \"\$out_nfs/PARTIAL_FAIL\"
    fi
    echo \"CAPTURE_COPIED \$name -> \$out_nfs status=\$status\"
  fi
  return \$status
}

set +e
run_one fixed8 \"\$FIXED_LOCAL\" \"\$FIXED_NFS\" 0.9181439782806684
fixed_status=\$?
set -e
if [[ \$fixed_status -eq 0 ]]; then
  python - <<'PY'
import json
from pathlib import Path
base = Path('outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/fixed8')
s = json.loads((base / 'score_summary.json').read_text())
d = json.loads((base / 'decision.json').read_text())
print('FIXED8_SCORE', s.get('score'))
print('FIXED8_CLASS', d.get('classification'))
print('FIXED8_RANKING_SHARE', d.get('ranking_share'))
print('FIXED8_SHARES', d.get('mechanism_shares'))
PY
fi

set +e
run_one holdout8 \"\$HOLD_LOCAL\" \"\$HOLD_NFS\" 0.9646726188580379 \\
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d \\
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
hold_status=\$?
set -e
if [[ \$hold_status -eq 0 ]]; then
  python - <<'PY'
import json
from pathlib import Path
base = Path('outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8')
s = json.loads((base / 'score_summary.json').read_text())
d = json.loads((base / 'decision.json').read_text())
print('HOLDOUT8_SCORE', s.get('score'))
print('HOLDOUT8_CLASS', d.get('classification'))
print('HOLDOUT8_RANKING_SHARE', d.get('ranking_share'))
print('HOLDOUT8_SHARES', d.get('mechanism_shares'))
PY
fi

if [[ \$fixed_status -eq 0 && \$hold_status -eq 0 ]]; then
  date -Iseconds > \"\$OUT_ROOT/DONE\"
  echo FRESH_CAPTURE_DONE \"\$OUT_ROOT\"
else
  date -Iseconds > \"\$OUT_ROOT/PARTIAL_FAIL\"
  echo FRESH_CAPTURE_PARTIAL fixed=\$fixed_status hold=\$hold_status >&2
fi
exit \$(( fixed_status != 0 ? fixed_status : hold_status ))
" 2>&1 | tee -a "${LOG}"

# Mirror compact artifacts to login workspace.
# Login home ≠ compute NFS: tar on compute, extract on login (pipe outside srun).
MIRROR="${SRC}/outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1"
mkdir -p "${MIRROR}"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:20:00 --mem=16G --chdir=/tmp bash -lc '
  ROOT=${HOME}/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1
  cd "$ROOT"
  tar cf - DONE \
    fixed8/DONE fixed8/decision.json fixed8/score_summary.json fixed8/metadata.json \
    fixed8/report.md fixed8/cause_summary.csv fixed8/cause_by_dataset.csv \
    fixed8/candidate_edge_diagnostic.csv fixed8/metric_by_dataset.csv \
    fixed8/experiment_config.yaml fixed8/command.txt \
    fixed8/predictions/postprocessed_submission.csv \
    holdout8/DONE holdout8/decision.json holdout8/score_summary.json holdout8/metadata.json \
    holdout8/report.md holdout8/cause_summary.csv holdout8/cause_by_dataset.csv \
    holdout8/candidate_edge_diagnostic.csv holdout8/metric_by_dataset.csv \
    holdout8/experiment_config.yaml holdout8/command.txt \
    holdout8/predictions/postprocessed_submission.csv
' | tar xf - -C "${MIRROR}"
echo "LOGIN_MIRROR_DONE"
find "${MIRROR}" -type f | sort | head -40
