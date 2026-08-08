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

## Two-seed raw-logit ensemble plumbing (2026-08-07)

- Public support artifact `pilkwang/biohub-tracking-support-pack-50ep-v1` contains only
  `weights/unet_transformer/split_0/edge_predictor_best.pth` (SHA-256
  `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`) and a
  `checkpoint_last.pth` from the same split/training run. No independent second seed was found.
- Opt-in ensemble config is `ensemble_weights_relative` plus `ensemble_alpha`; null preserves the
  exact original prediction command and single-model behavior. Duplicate-content checkpoints are
  rejected even when stored at different paths.
- The deterministic support-source patch wraps two compatible models. It blends raw detector
  logits returned by `encode()` and raw edge logits returned by `predict_edges()` before the
  unchanged sigmoid/softmax, thresholds, ILP, and graph pipeline.
- The patch was applied after D4 and compiled successfully against the actual support predictor.
  Local verification: 26 tests, Ruff, compileall, and diff check pass.

## Public independent Seed 2 retrieved (2026-08-07)

- Pilkwang Kim's public notebook `biohub-cell-tracking-two-seeds-logit-blend` attaches dataset
  `pilkwang/biohub-temporal-unet3d-seed314159-v1` and pins its best checkpoint SHA-256 to
  `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`.
- The artifact manifest and training metadata record `base_seed=effective_seed=314159`, method
  `unet_transformer_alltrain_seed314159_v1`, 400 captured epochs, and best epoch 381. Its checkpoint
  hash differs from Seed 1 (`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`).
- Seed 2 is staged without replacing Seed 1 at
  `data/support/weights/unet_transformer/seed_314159/edge_predictor_best.pth`; required `config.json`
  and provenance manifests are alongside it. `data/support` remains intentionally gitignored.
- `configs/clean_v106_two_seed.yaml` enables this checkpoint at `ensemble_alpha: 0.5`; compared with
  the baseline config, only `inference.ensemble_weights_relative` differs.

## Two-seed fixed-8 failure diagnosis (2026-08-07/08)

- Job 5058 finished all 8 GEFF predictions then crashed in submission validation
  (`ValueError: a node has more than two children`). No `summary.json` / score.
- Raw GEFFs were later confirmed under
  `~/biohub-outputs/fixed8/two_seed_alpha_0_5/predictions/raw_geff/` and copied to
  top-level `~/biohub-outputs/fixed8/two_seed_alpha_0_5/raw_geff/` with `MANIFEST.json`.
- Re-conversion of those saved GEFFs (no re-inference) found exactly one violating parent:
  `6bba_05db0fb1` parent `24536` at `t=30` with 3 children at `t=31`:
  ordinary edge `24536->25365` (prob≈0.538, dist_um=4.875) plus two `safe_division`
  edges to `25345` and `25368`.
- Root cause in vendored `add_safe_divisions_postlink`: parents with out-degree 1 are
  eligible, but the accept loop does not mark a source as used after adding one
  safe-division child, so multiple proposals for the same parent can all be added.
- Persistence/diagnostics helpers now live in `fixed8_cv._copy_raw_predictions` and
  `submission.collect_outdegree_violations`.
- Fix applied in `add_safe_divisions_postlink`: track `used_sources` so a parent receives
  at most one safe-division child (out-degree never exceeds 2). This is an intentional
  local divergence from the vendored V106 accept loop; re-running `vendor_clean_v106.py`
  would overwrite it unless upstream is updated too.
- Re-conversion of saved two-seed GEFFs (no re-inference) scored
  `adj_edge_jaccard=0.8847464271589631` vs control `0.87892959136423`
  (`delta=+0.005816835794733133`). Aggregates: edge TP/FP/FN = 3878/283/225;
  division TP/FP/FN = 0/15/7; node_recall ≈ 0.9847. Artifacts:
  `~/biohub-outputs/fixed8/two_seed_alpha_0_5/` and login copy
  `outputs/fixed8/two_seed_alpha_0_5/`.

## Two-seed detection-threshold sweep (2026-08-08)

- `detection_threshold` is applied in support `predict_unet_transformer.py`
  `_detect_cells_pooled` (`sigmoid(logits) > det_threshold`) before edges/ILP/GEFF.
  Existing raw GEFFs cannot be rescored at a new threshold; no cached det logits.
  Cheapest valid sweep = full two-seed re-inference per threshold.
- Sweep thresholds: 0.955, 0.960, 0.965, 0.96875, 0.9725, 0.975, 0.980.
  Configs under `configs/sweeps/two_seed_det_thresh_*.yaml`. Baseline
  `two_seed_alpha_0_5` left unchanged.
- Best: **0.960 → 0.887978610299** (+0.003232 vs baseline 0.884746427159).
  Edges 3885/273/218 (Δ +7/−10/−7 vs 3878/283/225); divisions unchanged 0/15/7.
  Re-run at 0.96875 matched baseline score exactly.
- Per-dataset vs baseline mixed (4 improved / 4 worsened); largest gains
  `44b6_e57ff5c6` (+0.0174) and `44b6_341df25f` (+0.0133). Comparison:
  `outputs/fixed8/two_seed_det_threshold_sweep/comparison.json` and NFS twin.

## Gap-2 recovery ablation on two-seed det=0.960 (2026-08-08)

- Config `configs/experiments/two_seed_det0_960_gap2.yaml`: same as det=0.960 two-seed
  control with only `output_gap2_recovery: true`. DeepCenter vetoes remain false.
- Cheapest path: re-convert NFS `two_seed_det_thresh_0_960/raw_geff` (no NN re-inference).
  Control outputs left unmodified.
- Result score `0.888140530150` vs control `0.887978610299` (Δ=+0.000162).
  Edges 3888/273/215 (Δ +3/0/−3); divisions 0/14/7 (Δ 0/−1/0).
  Datasets: 3 improved / 0 unchanged / 5 worsened. 280 gap-2 bridges, 840 edges.
  Largest win `44b6_e57ff5c6` (+0.00514); largest loss `44b6_0b24845f` (−0.01860).
- Decision: **REJECT** (Δ ≪ +0.001 and not broadly distributed).
  Artifacts: `outputs/fixed8/two_seed_det0_960_gap2/` and NFS twin.

## DeepCenter gating ablation on two-seed det=0.960 (2026-08-08)

- Config `configs/experiments/two_seed_det0_960_deepcenter.yaml`: det=0.960 two-seed with
  `use_deepcenter_veto/require/gap/safe_div` true; `output_gap2_recovery` false.
  Checkpoint from Kaggle `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` staged at
  `data/deepcenter/` (gitignored). Postprocess-only on saved 0.960 raw GEFFs.
- Score `0.888330075629` vs control `0.887978610299` (Δ=+0.000351).
  Edges 3877/263/226 (Δ −8/−10/+8); divisions 0/0/7 (Δ 0/−15/0).
  Datasets: 6 improved / 0 unchanged / 2 worsened; no regression < −0.005.
  DeepCenter rejected 487/487 safe-div candidates; gap veto checked 0.
  652 edges present in control submission absent after DeepCenter.
- Decision: **CONSIDER** (positive Δ, clear FP drop, broad 6/8 gains) but not strong
  promote (Δ < +0.001). Artifacts: `outputs/fixed8/two_seed_det0_960_deepcenter/`.

## No-safe-div ablation on two-seed det=0.960 (2026-08-08)

- Config `configs/experiments/two_seed_det0_960_no_safe_div.yaml`: only
  `output_safe_divisions: false`; DeepCenter and gap2 remain off. Postprocess-only
  on saved 0.960 raw GEFFs.
- Score `0.888330075629` — **exact match to DeepCenter**. Δ vs control = +0.000351.
  Edges 3877/263/226; divisions 0/0/7 (identical to DeepCenter aggregates).
  Datasets: 6 improved / 0 unchanged / 2 worsened; no regression < −0.005.
- Removed 487 control safe-div candidates / 431 added safe-div edges.
  Ordinary edges are not bit-identical: after safe_div removal, short-track filtering
  cascades and drops additional ordinary edges (221 unexpected removals total).
- Conclusion: DeepCenter's observed benefit was entirely from suppressing false
  safe divisions. Prefer `output_safe_divisions=false` over DeepCenter (same
  score, no extra model). Artifacts: `outputs/fixed8/two_seed_det0_960_no_safe_div/`.

## Late-strip safe-div ablation (2026-08-08)

- Run full control postprocess (safe-div ON through short-track + linefit), then
  remove only edges with `safe_division=1` before submission write. No second
  short-track pass. Config/script:
  `configs/experiments/two_seed_det0_960_late_strip_safe_div.yaml`,
  `scripts/run_late_strip_safe_div_ablation_from_geffs.sh`.
- Score `0.888208474399` (Δ=+0.000230 vs control). Edges 3878/264/225;
  divisions 0/0/7. Stripped 431 safe-div edges.
  **Ordinary non-safe-div edges are bit-identical to control** (no 221-edge
  cascade). Datasets 3/2/3 improved/unchanged/worsened; no regression < −0.005.
- Slightly below early no-safe-div (`0.888330`) because cascade side-effects are
  absent; edge TP/FN shifts vs control are from removed safe-div edges only.
  Artifacts: `outputs/fixed8/two_seed_det0_960_late_strip_safe_div/`.

## Public recipe audit + generalization workflow (2026-08-08)

- Public candidate recipe A is Pilkwang `biohub-cell-tracking-two-seeds-logit-blend`
  tag `selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap`.
  Verified: det=**0.96875**, det secondary weight **0.475**, edge weight **0.15**,
  mode **`low_margin_consensus`**, edge τ=0.48, ILP disappearance **1.5**,
  DeepCenter gap gate ON, safe-div ON, gap2 OFF. Literal LB 0.911: **unknown**.
  Audit: `outputs/audit/public_two_seed_recipe.md`.
- Local fixed-8 reproduction of recipe A: score **0.877744529257**
  (edges 3863/298/240, div 0/10/7), Δ=−0.01023 vs local det=0.960 control.
  Artifacts: `outputs/fixed8/public_two_seed_exact/`.
- Phase-6 isolation (public GEFFs + local recipe-C postprocess): **0.878148**
  (+0.00040 vs public exact). Public gap is almost entirely ensemble/inference.
- Holdout-8 (unused for selecting 0.960): datasets
  `44b6_0c582fdc, 44b6_0db75fae, 44b6_12dfb391, 44b6_144b256d,
  6bba_062c8d37, 6bba_07477033, 6bba_07e24132, 6bba_085bf656`.
  det=0.960 score 0.958371 (1/8 wins) vs det=0.96875 **0.959013** (7/8 wins).
  **REJECT promoting 0.960.**
- Holdout safe-div OFF on det=0.960 GEFFs: 0.958650 (+0.000278) but regressions
  −0.0136 / −0.0061 on two 44b6 sets; zero div TP persists. **Do not promote OFF.**
- Recommended generalization-safe Kaggle config: local recipe C α=0.5,
  det=**0.96875**, safe-div ON, gap2 OFF, DeepCenter OFF.
  Final report: `outputs/analysis/final_experiment_report.md`.
