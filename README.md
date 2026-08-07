# Biohub clean V106 pipeline

This repository packages the strongest accessible clean public baseline selected under the
project's no-hacking rules: Yusuke Togashi's **“Clean Approach + Lightweight Local CV | No Hack,”
Version 106**. Kaggle reports a public score of **0.908** for V106 and a best score of **0.908 at
V96**. V106 was selected because it is the newest complete accessible source tied at that clean
best score and includes fixed-8 local validation. The score has **not** been reproduced locally.

Source: <https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack>

The preserved notebook is under `upstream_clean_v106/`. Its SHA-256 is
`5adc99aef3b61f2d8c5da5253eb1df13262986e8879bf6f630b5c1b5fa345d9d`. The notebook and the
vendored notebook-owned functions are Apache-2.0 licensed; see `LICENSE` and the source record.

## What the pipeline uses

- an external 3D U-Net detector and node-transformer edge scorer from
  `pilkwang/biohub-tracking-support-pack-50ep-v1`;
- `weights/unet_transformer/split_0/edge_predictor_best.pth`;
- spatial D4 detector test-time augmentation;
- ILP graph construction with the upstream appearance/disappearance/division weights;
- motion relinking, one-frame gap repair, density-adaptive gating, safe divisions, isolated-node
  pruning, six-node minimum component filtering, conservative five-node rescue, and line-fit
  smoothing;
- GEFF-to-submission conversion and the notebook's official-spec-lite evaluation utilities.

No replacement detector, fake weights, training path, or independently designed tracker is
included.

## Local setup and validation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m biohub_pipeline.run --help
.venv\Scripts\python -m biohub_pipeline.run --config configs\clean_v106.yaml --dry-run
.venv\Scripts\python -m pytest -q
```

Dry-run validates the configuration, paths, and dependency availability without loading a model.
It succeeds when the real data and weights are absent and reports exactly what is missing.

## Future Kaggle/server execution

After obtaining the competition test stores and the exact support artifact:

```powershell
python -m biohub_pipeline.run `
  --data-dir D:\path\to\competition\test `
  --weights-dir D:\path\to\support-artifact `
  --support-dir D:\path\to\support-artifact `
  --config configs\clean_v106.yaml `
  --output submission.csv
```

Install the `runtime` extra or use the support artifact's offline wheels. Full inference requires
CUDA. This local migration did not run training, full inference, or create a competition
submission.

## Fixed-8 CV on HPC

Fixed-8 CV runs the current V106 inference and postprocessing pipeline on the same eight labeled
training datasets, then evaluates the predictions with the packaged official-spec-lite metric.
Real execution requires CUDA and all eight matching train `.zarr` and `.geff` pairs.

From the HPC repository checkout, run:

```bash
bash scripts/slurm/run_v106_fixed8_cv.sh
```

The script validates and stages only the required eight pairs, plus the code and support artifact;
it does not copy the full training directory. Results are copied to
`~/biohub-outputs/fixed8/current_v106/` and include `per_dataset.csv`, `summary.json`,
`manifest.json`, and preserved raw and postprocessed predictions.

The module can also be invoked directly in a prepared GPU environment:

```bash
python -m biohub_pipeline.fixed8_cv \
  --data-dir /path/to/competition/train \
  --weights-dir /path/to/support \
  --support-dir /path/to/support \
  --config configs/clean_v106.yaml \
  --output-dir outputs/fixed8/current_v106
```

Fixed-8 CV does not improve the Kaggle score by itself. It provides a reproducible local
measurement for deciding whether future changes such as detector ensembling or threshold
calibration are actually better than the current V106 pipeline. The current reproduction target
recorded upstream is approximately `0.87892959136423`.

## Repository layout

- `upstream_clean_v106/`: preserved notebook, source conversion, attribution, and fingerprint.
- `src/biohub_pipeline/`: configuration, artifact checks, inference adapter, exact notebook graph
  postprocessing, submission validation, and local evaluation utilities.
- `configs/clean_v106.yaml`: upstream V106 defaults.
- `tests/test_clean_*.py`: source, CLI, configuration, graph, submission, and evaluation tests.
- `docs/HISTORICAL_BASELINE.md`: short record of the removed approximately 0.543 classical baseline.
- `docs/REMOVALS.md`: tracked cleanup inventory and reasons.

## Unverified without external assets

- support-artifact manifest and weight checksum;
- detector and learned-edge prediction equality;
- GEFF counts for the hidden test set;
- fixed-8 CV score and runtime;
- submission equality and the reported 0.908 public score.
