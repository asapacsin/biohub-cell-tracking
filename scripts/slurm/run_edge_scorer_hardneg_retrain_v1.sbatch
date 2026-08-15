#!/bin/bash
#SBATCH -p gpu_batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-18:00:00
#SBATCH -J hnretrv1
#SBATCH --chdir=/tmp
#SBATCH -o /home/mc46451/biohub-outputs/logs/edge_scorer_hardneg_retrain_v1_%j.out
#SBATCH -e /home/mc46451/biohub-outputs/logs/edge_scorer_hardneg_retrain_v1_%j.err

# Compute-node worker. Login home is NOT mounted here.
# Expects a staged tarball on compute NFS (see submit_edge_scorer_hardneg_retrain_v1.sh).
# Frozen recipe: motion-relink OFF, edge_threshold=0.40.
set -euo pipefail

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

DEST=/tmp/${USER}/biohub-edge-hardneg-retrain-v1
STAGE_TAR=${HOME}/biohub-outputs/stage/edge_scorer_hardneg_retrain_v1.tar
OUT_NFS=${HOME}/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1
CAP_NFS=${HOME}/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1
STATUS=${HOME}/biohub-outputs/logs/edge_scorer_hardneg_retrain_v1.status
JOB_ID="${SLURM_JOB_ID:-unknown}"
GIT="${HNRETR_GIT:-unknown}"

status() {
  printf '%s\n' "$*" | tee -a "${STATUS}"
  echo "STATUS $(date -Iseconds) $*"
}

mkdir -p "$(dirname "${STATUS}")"
: > "${STATUS}"
status "START host=$(hostname) job=${JOB_ID} git=${GIT}"
status "ETA_WINDOW_OPEN=$(date -Iseconds) ETA_HOURS=14"
echo "GPU=$(nvidia-smi -L | head -1 || true)"

if [[ ! -f "${STAGE_TAR}" ]]; then
  status "ERROR missing stage tar ${STAGE_TAR}"
  exit 2
fi
if [[ ! -d "${CAP_NFS}/fixed8/candidate_capture" ]]; then
  status "ERROR missing capture at ${CAP_NFS}"
  exit 2
fi
if [[ -e "${OUT_NFS}" ]]; then
  status "ERROR output already exists: ${OUT_NFS}"
  exit 2
fi

rm -rf "${DEST}"
mkdir -p "${DEST}"
status "EXTRACT ${STAGE_TAR} -> ${DEST}"
tar xf "${STAGE_TAR}" -C "${DEST}"
cd "${DEST}"

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
export PYTHONPATH="${DEST}/src${PYTHONPATH:+:$PYTHONPATH}"
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
status "Copying durable results to ${OUT_NFS}"
export OUT_NFS
python - <<PY
import os, shutil
from pathlib import Path
src = Path('outputs/experiments/edge_scorer_hardneg_retrain_v1')
nfs = Path(os.environ['OUT_NFS'])
if nfs.exists():
    shutil.rmtree(nfs)
shutil.copytree(src, nfs, ignore=shutil.ignore_patterns('*_work', 'work'))
print('NFS_OK', nfs)
PY

status "FINISHED job=${JOB_ID} at $(date -Iseconds)"
echo "Finished edge_scorer_hardneg_retrain_v1 at $(date -Iseconds) job=${JOB_ID}"
