#!/usr/bin/env bash
# Official test inference + Kaggle submission pack under the frozen recipe:
# motion_relink OFF, edge_threshold=0.40, det=0.96875, two-seed α=0.5.
# Does not change production YAML. Writes CSV + GEFFs to compute NFS, mirrors CSV to login.
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/frozen_recipe_test_submit.log"

cd "${SRC}"
echo "Starting frozen-recipe test submit at $(date -Iseconds)" | tee "${LOG}"
echo "ETA_WINDOW_OPEN=$(date -Iseconds) ETA_HOURS=1.5" | tee -a "${LOG}"
echo "CFG=configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml" | tee -a "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  --exclude='data/competition/train' --exclude='data/sample' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  data/competition/test \
  data/competition/sample_submission.csv \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0-04:00:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-frozen-test-submit
OUT_NFS=\${HOME}/biohub-outputs/kaggle/recipe_c_motion_off_edge_0_40_v1
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
mkdir -p \"\$OUT_NFS\"
export OUT_NFS DEST

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biohub_pipeline.config import load_config
from biohub_pipeline.inference import build_predict_command, list_stems
from biohub_pipeline.run import main as run_main
from biohub_pipeline.submission import validate_submission_file

cfg_path = Path('configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml')
cfg = load_config(cfg_path)
assert abs(float(cfg.inference['edge_threshold']) - 0.4) < 1e-12
assert abs(float(cfg.inference['detection_threshold']) - 0.96875) < 1e-12
assert abs(float(cfg.inference['ensemble_alpha']) - 0.5) < 1e-12
assert cfg.postprocessing['output_motion_relink'] is False
assert cfg.postprocessing['output_filter_short_tracks'] is True
assert cfg.inference.get('pairwise_hardneg_w') in (None, [], False)
assert not cfg.inference.get('margin_gated_dist_lambda')
print(
    'RECIPE_OK',
    'edge', cfg.inference['edge_threshold'],
    'det', cfg.inference['detection_threshold'],
    'motion', cfg.postprocessing['output_motion_relink'],
    'alpha', cfg.inference['ensemble_alpha'],
    flush=True,
)

data_dir = Path('data/competition/test').resolve()
support = Path('data/support').resolve()
stems = list_stems(data_dir)
expected = ['44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1']
if stems != expected:
    raise RuntimeError(f'test stems {stems} != {expected}')
print('TEST_STEMS', stems, flush=True)

out_nfs = Path(__import__('os').environ['OUT_NFS'])
csv_path = Path('outputs/kaggle_submission/submission_recipe_c_motion_off_edge_0_40.csv')
csv_path.parent.mkdir(parents=True, exist_ok=True)

# Dry-run then full inference via the same entry point as production.
rc = run_main([
    '--config', str(cfg_path),
    '--data-dir', str(data_dir),
    '--weights-dir', str(support),
    '--support-dir', str(support),
    '--dry-run',
])
if rc != 0:
    raise SystemExit(rc)

started = datetime.now(UTC)
rc = run_main([
    '--config', str(cfg_path),
    '--data-dir', str(data_dir),
    '--weights-dir', str(support),
    '--support-dir', str(support),
    '--output', str(csv_path),
])
if rc != 0:
    raise SystemExit(rc)
elapsed = (datetime.now(UTC) - started).total_seconds()

stats = validate_submission_file(csv_path)
print('VALIDATE_OK', json.dumps(stats), flush=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
primary = support / Path(str(cfg.inference['weights_relative']))
secondary = support / Path(str(cfg.inference['ensemble_weights_relative']))
geffs = sorted((support / 'repo' / 'predictions').glob('*/unet_transformer/split_0/*.geff'))
raw_dir = out_nfs / 'raw_geff'
raw_dir.mkdir(parents=True, exist_ok=True)
for geff in geffs:
    dest = raw_dir / geff.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(geff, dest)

shutil.copy2(csv_path, out_nfs / 'submission.csv')
shutil.copy2(cfg_path, out_nfs / 'experiment_config.yaml')
meta = {
    'schema_version': 1,
    'created_at_utc': datetime.now(UTC).isoformat(),
    'experiment': 'recipe_c_motion_off_edge_0_40_test_submit',
    'config': str(cfg_path),
    'git_commit': git.stdout.strip() or 'unknown',
    'datasets': stems,
    'validate': stats,
    'elapsed_seconds': elapsed,
    'primary_checkpoint_sha256': sha256(primary),
    'secondary_checkpoint_sha256': sha256(secondary),
    'csv_sha256': sha256(csv_path),
    'recipe': {
        'edge_threshold': float(cfg.inference['edge_threshold']),
        'detection_threshold': float(cfg.inference['detection_threshold']),
        'ensemble_alpha': float(cfg.inference['ensemble_alpha']),
        'output_motion_relink': bool(cfg.postprocessing['output_motion_relink']),
        'output_filter_short_tracks': bool(cfg.postprocessing['output_filter_short_tracks']),
        'output_min_track_len': int(cfg.postprocessing['output_min_track_len']),
        'output_safe_divisions': bool(cfg.postprocessing['output_safe_divisions']),
    },
    'local_cv_reference': {
        'fixed8': 0.9181439782806684,
        'holdout8': 0.9646726188580379,
    },
    'note': 'Official test inference. No local GT score. Upload submission.csv to Kaggle.',
}
(out_nfs / 'metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(out_nfs / 'DONE').write_text(datetime.now(UTC).isoformat() + '\n', encoding='utf-8')
print('TEST_SUBMIT_DONE', json.dumps({'stats': stats, 'elapsed_s': elapsed, 'out': str(out_nfs)}), flush=True)
PY

echo \"GPU frozen-recipe test submit finished at \$(date -Iseconds)\"
" 2>&1 | tee -a "${LOG}"

echo "Mirroring submission CSV to login ..." | tee -a "${LOG}"
MIRROR="${SRC}/outputs/kaggle_submission/recipe_c_motion_off_edge_0_40_v1"
mkdir -p "${MIRROR}"
srun -p gpu_batch -N1 -n1 --cpus-per-task=1 --mem=4G -t 0-00:15:00 --chdir=/tmp \
  bash -lc 'cd ${HOME}/biohub-outputs/kaggle/recipe_c_motion_off_edge_0_40_v1 && tar cf - submission.csv metadata.json experiment_config.yaml DONE' \
| tar xf - -C "${MIRROR}"
echo "LOGIN_MIRROR_DONE ${MIRROR}" | tee -a "${LOG}"
ls -la "${MIRROR}" | tee -a "${LOG}"
echo "Finished frozen-recipe test submit at $(date -Iseconds)" | tee -a "${LOG}"
