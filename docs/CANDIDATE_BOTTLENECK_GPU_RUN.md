# Candidate-edge bottleneck GPU run

This run executes the predeclared candidate-edge diagnostic without changing the production
tracking recipe. It requires the current GitHub branch, the competition data, the primary support
pack, the independent seed-314159 checkpoint, and one CUDA GPU.

## Kaggle inputs

Attach these three sources to a Kaggle notebook:

- competition: `biohub-cell-tracking-during-development`;
- dataset: `pilkwang/biohub-tracking-support-pack-50ep-v1`;
- dataset: `pilkwang/biohub-temporal-unet3d-seed314159-v1`.

Enable one T4 (or stronger) GPU and internet access for the GitHub clone. The launcher accepts both
the current `/kaggle/input/{competitions,datasets}/...` layout and Kaggle's legacy flat mounts. It
pins both checkpoint contents by SHA-256 and stages symlinks, so it does not copy the 3D volumes.

## Exact Kaggle command

Run this as one Bash cell:

```bash
set -euo pipefail

git clone --branch codex/cloud-candidate-bottleneck-prep --single-branch \
  https://github.com/asapacsin/biohub-cell-tracking.git \
  /kaggle/working/biohub-cell-tracking
cd /kaggle/working/biohub-cell-tracking

SUPPORT_WHEELS="$(find /kaggle/input -type d \
  -path '*/biohub-tracking-support-pack-50ep-v1/wheels' -print -quit)"
test -n "${SUPPORT_WHEELS}"

python -m pip install --no-index --find-links="${SUPPORT_WHEELS}" \
  tracksdata zarr 'geff>=1.1.3.1.1' 'geff-spec<1.2' 'ilpy>=0.5.1' \
  pyscipopt polars blosc2 dask imagecodecs pyarrow 'rustworkx>=0.17.1' \
  'sqlalchemy>=2' 'scikit-image>=0.24' 'numcodecs>=0.13,<0.16' donfig

nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

PYTHONPATH=src PYTHONUNBUFFERED=1 python -m biohub_pipeline.kaggle_candidate \
  --input-root /kaggle/input \
  --workspace /kaggle/working/biohub-candidate \
  --output-dir /kaggle/working/candidate_edge_bottleneck_v1 \
  --top-k 16
```

Do not add `--prepare-only` for the real run. For a CPU-only preflight, add it; the launcher still
runs both CUDA probes and prints the exact experiment command, but never starts inference.

## Expected durable outputs

Kaggle should preserve `/kaggle/working/candidate_edge_bottleneck_v1`, including:

- `report.md` and `decision.json`;
- `score_summary.json` and `metric_by_dataset.csv`;
- `candidate_edge_diagnostic.csv`, `cause_summary.csv`, and
  `cause_by_dataset.csv`;
- bounded `candidate_capture/` score exports;
- raw GEFFs and final postprocessed predictions;
- `metadata.json` with commit/checkpoint/config hashes and the equivalence check.

The run is valid only if the instrumented score exactly matches the historical control
`0.8847464271589631`. A mismatch means environment or instrumentation drift and must be resolved
before interpreting the causal classification.
