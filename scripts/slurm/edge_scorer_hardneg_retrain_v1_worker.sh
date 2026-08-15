#!/usr/bin/env bash
# Runs on the GPU node from the staged /tmp dest (code + 16 zarr/geff + support).
# Frozen recipe: motion-relink OFF, edge_threshold=0.40.
set -euo pipefail

HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

OUT_NFS=${HOME}/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1
CAP_NFS=${HOME}/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1
STATUS=${HOME}/biohub-outputs/logs/edge_scorer_hardneg_retrain_v1.status
JOB_ID="${SLURM_JOB_ID:-unknown}"
GIT="${HNRETR_GIT:-unknown}"

status() {
  # Compute-home quota is full; never fail the job on NFS log writes.
  echo "STATUS $(date -Iseconds) $*"
  mkdir -p "$(dirname "${STATUS}")" 2>/dev/null || true
  printf '%s\n' "$*" >> "${STATUS}" 2>/dev/null || true
}

rm -f "${HOME}/biohub-outputs/stage/edge_scorer_hardneg_retrain_v1.tar" 2>/dev/null || true

status "START host=$(hostname) job=${JOB_ID} git=${GIT} cwd=$(pwd)"
status "ETA_WINDOW_OPEN=$(date -Iseconds) ETA_HOURS=14"
echo "GPU=$(nvidia-smi -L | head -1 || true)"

if [[ ! -d "${CAP_NFS}/fixed8/candidate_capture" ]]; then
  status "ERROR missing capture at ${CAP_NFS}"
  exit 2
fi
if [[ -e "${OUT_NFS}" ]]; then
  status "ERROR output already exists: ${OUT_NFS}"
  exit 2
fi

status "Copying compact capture (no raw_geff) from compute NFS"
mkdir -p capture/fixed8 capture/holdout8
cp -a "${CAP_NFS}/fixed8/candidate_capture" capture/fixed8/
cp -a "${CAP_NFS}/fixed8/candidate_edge_diagnostic.csv" \
      "${CAP_NFS}/fixed8/metric_by_dataset.csv" \
      "${CAP_NFS}/fixed8/score_summary.json" \
      capture/fixed8/
cp -a "${CAP_NFS}/holdout8/candidate_edge_diagnostic.csv" \
      "${CAP_NFS}/holdout8/metric_by_dataset.csv" \
      "${CAP_NFS}/holdout8/score_summary.json" \
      capture/holdout8/

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr "geff>=1.1.3.1.1" "geff-spec<1.2" "ilpy>=0.5.1" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow "rustworkx>=0.17.1" \
  "sqlalchemy>=2" "scikit-image>=0.24" "numcodecs>=0.13,<0.16" donfig >/dev/null
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
from pathlib import Path
from biohub_pipeline.config import load_config
frozen = load_config(Path('configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml'))
cand = load_config(Path('configs/experiments/recipe_c_edge_0_40_hardneg_retrain_v1.yaml'))
assert abs(float(frozen.inference['edge_threshold']) - 0.4) < 1e-12
assert frozen.postprocessing['output_motion_relink'] is False
assert abs(float(cand.inference['edge_threshold']) - 0.4) < 1e-12
assert cand.postprocessing['output_motion_relink'] is False
assert cand.inference.get('pairwise_hardneg_w') in (None,)
assert cand.inference.get('margin_gated_dist_lambda') in (None, 0, 0.0)
assert 'hardneg_v1_split_0' in cand.inference['weights_relative']
assert 'hardneg_v1_seed_314159' in cand.inference['ensemble_weights_relative']
print('RECIPE_OK frozen+candidate gates unchanged; scorer paths swapped')
PY

status "TRAIN begin"
python scripts/train_edge_scorer_hardneg.py \
  --data-dir data/competition/train \
  --support-dir data/support \
  --capture-root capture/fixed8/candidate_capture \
  --diagnostic-csv capture/fixed8/candidate_edge_diagnostic.csv \
  --output-dir outputs/experiments/edge_scorer_hardneg_retrain_v1/train \
  --epochs 6 \
  --lr 3e-5
status "TRAIN done"

python - <<'PY'
from pathlib import Path
import hashlib
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
s1 = Path('data/support/weights/unet_transformer/hardneg_v1_split_0/edge_predictor_best.pth')
s2 = Path('data/support/weights/unet_transformer/hardneg_v1_seed_314159/edge_predictor_best.pth')
o1 = Path('data/support/weights/unet_transformer/split_0/edge_predictor_best.pth')
o2 = Path('data/support/weights/unet_transformer/seed_314159/edge_predictor_best.pth')
assert s1.is_file() and s2.is_file()
h1, h2, a1, a2 = sha(s1), sha(s2), sha(o1), sha(o2)
assert h1 != h2, 'retrained seeds must differ'
assert h1 != a1 and h2 != a2, 'retrain must change both checkpoints'
print('CKPT_OK', h1[:12], h2[:12])
Path('outputs/experiments/edge_scorer_hardneg_retrain_v1/train/checkpoint_sha256.json').write_text(
    __import__('json').dumps({'hardneg_v1_split_0': h1, 'hardneg_v1_seed_314159': h2,
                              'orig_split_0': a1, 'orig_seed_314159': a2}, indent=2) + '\n')
PY

CFG=configs/experiments/recipe_c_edge_0_40_hardneg_retrain_v1.yaml
EVAL_ROOT=outputs/experiments/edge_scorer_hardneg_retrain_v1
mkdir -p "${EVAL_ROOT}"

status "EVAL fixed-8 begin"
python scripts/run_candidate_bottleneck_experiment.py \
  --data-dir data/competition/train \
  --support-dir data/support \
  --config "${CFG}" \
  --output-dir "${EVAL_ROOT}/fixed8" \
  --work-dir "${EVAL_ROOT}/fixed8_work" \
  --top-k 16 \
  --control-score 0.9181439782806684
status "EVAL fixed-8 done"

status "EVAL holdout-8 begin"
python scripts/run_candidate_bottleneck_experiment.py \
  --data-dir data/competition/train \
  --support-dir data/support \
  --config "${CFG}" \
  --output-dir "${EVAL_ROOT}/holdout8" \
  --work-dir "${EVAL_ROOT}/holdout8_work" \
  --top-k 16 \
  --skip-fixed8-validation \
  --control-score 0.9646726188580379 \
  --datasets "${HOLDOUT8[@]}"
status "EVAL holdout-8 done"

python scripts/write_hardneg_retrain_report.py \
  --run-root "${EVAL_ROOT}" \
  --baseline-fixed-diag capture/fixed8/candidate_edge_diagnostic.csv \
  --baseline-hold-diag capture/holdout8/candidate_edge_diagnostic.csv \
  --baseline-fixed-metrics capture/fixed8/metric_by_dataset.csv \
  --baseline-hold-metrics capture/holdout8/metric_by_dataset.csv \
  --job-id "${JOB_ID}" \
  --git-commit "${GIT}"

date -Iseconds > "${EVAL_ROOT}/DONE"
echo "===== COMPARISON_JSON_BEGIN ====="
cat "${EVAL_ROOT}/comparison.json" || true
echo "===== COMPARISON_JSON_END ====="
echo "===== REPORT_MD_BEGIN ====="
cat "${EVAL_ROOT}/report.md" || true
echo "===== REPORT_MD_END ====="

status "Copying durable compact results to ${OUT_NFS} (best-effort; quota may block)"
export OUT_NFS
python - <<'PY' || true
import os, shutil
from pathlib import Path
src = Path('outputs/experiments/edge_scorer_hardneg_retrain_v1')
nfs = Path(os.environ['OUT_NFS'])
try:
    if nfs.exists():
        shutil.rmtree(nfs)
    shutil.copytree(src, nfs, ignore=shutil.ignore_patterns('*_work', 'work'))
    print('NFS_OK', nfs)
except OSError as exc:
    print('NFS_COPY_FAILED', exc)
PY

status "FINISHED job=${JOB_ID} at $(date -Iseconds)"
echo "Finished edge_scorer_hardneg_retrain_v1 at $(date -Iseconds) job=${JOB_ID}"
