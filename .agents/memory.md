# Durable project memory

## Active clean pipeline (2026-08-06)

- Authoritative source is Yusuke Togashi's Kaggle notebook `Clean Approach + Lightweight Local CV
  | No Hack`, Version 106, Apache-2.0. Kaggle reports public score 0.908 and best score 0.908 at
  V96. V106 was selected as the newest complete accessible version tied at the clean best score.
- Exact notebook: `upstream_clean_v106/clean-approach-lightweight-local-cv-no-hack.ipynb`, SHA-256
  `5adc99aef3b61f2d8c5da5253eb1df13262986e8879bf6f630b5c1b5fa345d9d`.
- The notebook does not embed the detector/model implementation or weights. It requires
  `pilkwang/biohub-tracking-support-pack-50ep-v1`, including `repo/` and
  `weights/unet_transformer/split_0/edge_predictor_best.pth`. Never approximate these assets.
- Active package is `src/biohub_pipeline`; configuration is `configs/clean_v106.yaml`; entry point
  is `python -m biohub_pipeline.run`. Dry-run does not load a model and works without external
  assets.
- Notebook-owned graph postprocessing and official-spec-lite functions are mechanically vendored
  by `scripts/vendor_clean_v106.py` under a source fingerprint.
- The old blob, nearest-neighbour, custom training, experiment, and obsolete documentation paths
  were removed after a reference audit. They remain recoverable from Git history/tag
  `legacy_baseline`. The historical score record is `docs/HISTORICAL_BASELINE.md` (approximately
  0.543).
- Local verification after cleanup: 11 pytest tests, Python compilation, CLI help, dry-run, and
  Ruff all passed.
- No full data, model training, full inference, real submission, local accuracy validation, or
  leaderboard reproduction has been performed.
