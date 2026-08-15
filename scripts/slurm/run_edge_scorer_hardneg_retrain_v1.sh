#!/usr/bin/env bash
# Login launcher: tar|srun to node-local /tmp (compute home quota cannot hold the 7G stage).
# Disconnect-safe via nohup. QOS allows one job; do not nest sbatch/srun.
# Frozen recipe: motion-relink OFF, edge_threshold=0.40.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/edge_scorer_hardneg_retrain_v1.log"
GIT=$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || echo unknown)
export HNRETR_GIT="${GIT}"

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

{
  echo "Starting edge_scorer_hardneg_retrain_v1 at $(date -Iseconds)"
  echo "ETA_WINDOW_OPEN=$(date -Iseconds) ETA_HOURS=14"
  echo "GIT=${GIT}"
} | tee "${LOG}"

if squeue -u "${USER}" -h | grep -q .; then
  echo "ERROR: a Slurm job is already active (QOS max 1); not duplicating" | tee -a "${LOG}"
  squeue -u "${USER}" | tee -a "${LOG}"
  exit 3
fi

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-18:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
export HNRETR_GIT='${GIT}'
DEST=/tmp/\${USER}/biohub-edge-hardneg-retrain-v1
rm -rf \"\$DEST\"
mkdir -p \"\$DEST\"
tar xf - -C \"\$DEST\"
cd \"\$DEST\"
bash scripts/slurm/edge_scorer_hardneg_retrain_v1_worker.sh
" 2>&1 | tee -a "${LOG}"

echo "Mirroring compact artifacts to login outputs/" | tee -a "${LOG}"
mkdir -p "${SRC}/outputs/experiments/edge_scorer_hardneg_retrain_v1" \
         "${SRC}/outputs/analysis"
srun -p gpu_batch -N1 -n1 --gres=gpu:0 --cpus-per-task=2 --mem=8G -t 0-00:20:00 --chdir=/tmp \
  bash -lc 'cd ${HOME}/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1 && tar -czf - comparison.json DONE report.md train/mine_summary.json train/train_summary.json train/checkpoint_sha256.json fixed8/score_summary.json fixed8/metric_by_dataset.csv fixed8/cause_summary.csv fixed8/candidate_edge_diagnostic.csv holdout8/score_summary.json holdout8/metric_by_dataset.csv holdout8/cause_summary.csv holdout8/candidate_edge_diagnostic.csv 2>/dev/null' \
  > /tmp/edge_scorer_hardneg_retrain_v1_mirror.tar.gz 2>>"${LOG}" || true
if [[ -s /tmp/edge_scorer_hardneg_retrain_v1_mirror.tar.gz ]]; then
  tar -xzf /tmp/edge_scorer_hardneg_retrain_v1_mirror.tar.gz \
    -C "${SRC}/outputs/experiments/edge_scorer_hardneg_retrain_v1" || true
  if [[ -f "${SRC}/outputs/experiments/edge_scorer_hardneg_retrain_v1/report.md" ]]; then
    cp -f "${SRC}/outputs/experiments/edge_scorer_hardneg_retrain_v1/report.md" \
      "${SRC}/outputs/analysis/edge_scorer_hardneg_retrain_v1_report.md"
  fi
fi
echo "Finished edge_scorer_hardneg_retrain_v1 at $(date -Iseconds)" | tee -a "${LOG}"
