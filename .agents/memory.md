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

## HPC V106 experiment start (2026-08-06 login01)

- No Slurm job was running when asked to cancel; queue was empty.
- Legacy I/O-fixed train job had already finished earlier: NFS `~/biohub-outputs/DONE`
  with `seed_42.ts` and `submission_learned.csv` (5528 nodes / 4672 edges) — far sparser
  than classical 38077/32917; fetched to `outputs/kaggle_submission/submission_learned.csv`.
- Active experiment is clean V106: downloaded
  `pilkwang/biohub-tracking-support-pack-50ep-v1` into `data/support` (340M; weights present).
- Dry-run reports `ready_for_full_inference: true` with 4 test zarrs; login `.venv` lacks
  runtime modules (torch/tracksdata/…); GPU conda `biohub` has torch 2.6.0+cu124.
- Job 4699 (`scripts/slurm/run_v106_infer.sh`) stages ~1.8G test + support to `/tmp` on
  um-gpu02 and runs `python -m biohub_pipeline.run` with `configs/clean_v106.yaml`.

## V106 public-test experiment (2026-08-06 login01)

- Support artifact downloaded to `data/support` from
  `pilkwang/biohub-tracking-support-pack-50ep-v1` (weights
  `weights/unet_transformer/split_0/edge_predictor_best.pth` present).
- GPU inference via `scripts/slurm/run_v106_infer.sh` (slim stage: test 1.8G + support
  340M to `/tmp`; conda env `biohub` + support wheels; `PYTHONPATH=src`).
- Bugs fixed during first runs: (1) relative `--data-dir` broke under `cwd=repo/` —
  resolve paths in `biohub_pipeline.run`; (2) `validate_graph` rejected slightly negative
  float coords — clamp with `max(0, int(round(...)))` before checks, matching upstream
  notebook write path.
- Result: `outputs/kaggle_submission/submission_v106.csv` — 120246 nodes / 115957 edges /
  4 datasets; `validate_submission_file` passed. NFS copy under `~/biohub-outputs/v106/`.
- Reported upstream public score for this notebook version is 0.908 (not re-verified on
  Kaggle from this host yet).

## Current V106 fixed-8 workflow (2026-08-07)

- `python -m biohub_pipeline.fixed8_cv` validates and runs exactly the eight fixed upstream train
  datasets; missing `.zarr` or `.geff` inputs are fatal and unrelated train datasets are ignored.
- Fixed-8 prediction conversion uses the same `write_submission_from_geff` path as normal V106,
  then evaluates through the existing `biohub_pipeline.evaluation` official-spec-lite functions.
- Outputs are `per_dataset.csv`, `summary.json`, `manifest.json`, raw prediction GEFFs, and the
  combined postprocessed prediction CSV. The current reference target is `0.87892959136423`.
- `scripts/slurm/run_v106_fixed8_cv.sh` stages only the eight required zarr/geff pairs, code, and
  support pack, and copies results to `~/biohub-outputs/fixed8/current_v106/`.
- Local verification is 19 passing tests plus Ruff, compileall, Bash syntax, and CLI help. Real
  fixed-8 reproduction remains unrun because it requires HPC training data, support assets, and CUDA.

## Fixed-8 current V106 control reproduced (2026-08-07 login01)

- Commit `3e6bf40982c7df3fa0a979ef1764bb95db68fb44`; job 5044 on um-gpu02
  (RTX 2080 Ti); `scripts/slurm/run_v106_fixed8_cv.sh`.
- Score `adj_edge_jaccard=0.87892959136423` matches reference exactly
  (`delta_vs_reference=0.0`, `reference_difference_material=false`).
- Aggregates: edge TP/FP/FN = 3852/287/251; division TP/FP/FN = 0/6/7;
  node_recall ≈ 0.9804; micro edge Jaccard ≈ 0.87745; division Jaccard = 0.0.
- Outputs: NFS `~/biohub-outputs/fixed8/current_v106/` and login copy
  `outputs/fixed8/current_v106/` (`summary.json`, `per_dataset.csv`,
  `manifest.json`, `predictions/`, `DONE`).
- Largest association error burden (edge FP+FN): `6bba_05db0fb1` (240),
  `6bba_fc83837d` (181), `44b6_e57ff5c6` (78).
