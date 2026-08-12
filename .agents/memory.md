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

## Association-density diagnostic (2026-08-09)

- Added reusable CPU-only analysis in `src/biohub_pipeline/association_density.py`
  and `scripts/run_association_density_diagnostic.py`; the production recipe was not
  changed and no training, detector inference, submission, or GPU job was run.
- Artifact coverage was limited to four public two-seed GEFFs that overlap competition
  GT: `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, and
  `6bba_05db0fb1`. The graphs are pre-final-postprocessing and contain only selected
  optimizer edges (`solution=True` throughout), so rejected candidate scores and true
  best-minus-second margins are unavailable.
- On 2,121 ordinary GT associations (division parents excluded), overall errors were
  89 (4.196%); attributed counts were 44 wrong associations, 28 missing edges,
  11 missed source nodes, and 6 missed target nodes. Overall node recall was 0.9950.
- A fixed 15-micrometer next-frame neighborhood gave density quartiles at 3/9/13
  predicted nodes. High-density error rate was 5.023% vs 2.177% at low density,
  ratio 2.3075. Association-attributed errors outnumbered detection-attributed errors
  72 to 17.
- Family error rates were 4.303% for `6bba` (87/2,022) and 2.020% for `44b6`
  (2/99); `6bba_05db0fb1` contributed 73 errors at 6.202%.
- Decision: **B PARTIALLY VERIFIED**. Density and association-error dominance are
  quantitatively supported, but the small-margin interaction and exact final fixed-8
  behavior cannot be verified from the saved artifact.
- Highest-value next experiment: on one already-planned inference/CV run, persist all
  pre-ILP `(source_id, target_id, edge_prob, solution)` candidate edges keyed to final
  node IDs, save exact final postprocessed fixed-8 graphs, and rerun this diagnostic.
  Do not implement a new tracker before that measurement.
- Outputs: `outputs/analysis/association_density_diagnostic.csv`,
  `association_density_summary.csv`, `association_density_by_dataset.csv`, and
  `association_density_report.md`.

## Selected-edge confidence calibration (2026-08-09)

- Added a CPU-only, predeclared 5% intervention-budget diagnostic in
  `src/biohub_pipeline/selected_edge_calibration.py` and
  `scripts/run_selected_edge_calibration.py`; no neural inference or production change was made.
- Of 2,121 ordinary associations in the source density diagnostic, 2,080 had a finite selected-edge
  score. These included 48 scored errors (44 wrong associations and 4 target-node-missed rows).
- The bottom-confidence 104 associations captured 13/48 errors (27.083% recall) at 12.50%
  intervention precision, a 5.42x lift over the 2.31% error prevalence. Correctness AUROC was
  0.752543. Both predeclared requirements failed: at least 40% error capture and AUROC at least 0.80.
- `6bba_05db0fb1` contained 41/48 scored errors and had correctness AUROC 0.701617; its independent
  bottom-5% budget captured 13/41 errors. `6bba_05b6850b` had 7 errors and AUROC 0.823620 but its
  bottom-5% budget captured only 1/7. Neither `44b6` overlap contained a scored error.
- Decision: **D. HYPOTHESIS REJECTED**. Selected-edge confidence is enriched for error but does not
  capture enough of the remaining error burden to justify confidence-only calibration as the next
  production lever. Keep the generalization-safe recipe unchanged and target edge-ranking quality.
- Durable outputs are under `outputs/experiments/selected_edge_calibration/`.

## Candidate-edge bottleneck GPU runs (2026-08-09)

- Codex left `scripts/run_candidate_bottleneck_experiment.py` unrun. Executed on HPC with
  launcher `scripts/slurm/run_candidate_bottleneck_fixed8.sh` using recipe C
  (`configs/sweeps/two_seed_det_thresh_0_96875.yaml`).
- Fixed-8 instrumented score **0.884746427159** (exact control match). Causal errors 193.
  Shares: ranking 37.3%, postprocessing_removed 29.5%, threshold 16.1%, ILP 10.9%,
  rematch 6.2%. Classification **D. HYPOTHESIS REJECTED**.
- Holdout-8 repeat (same 8 unused sequences as prior thresh/safe-div holdout): score
  **0.959013051622** (exact prior holdout match). Causal errors 67. Shares:
  postprocessing_removed **34.3%**, ranking 32.8%, ILP 13.4%, threshold 11.9%, rematch 7.5%.
  Classification **D. HYPOTHESIS REJECTED**; recommended next is one targeted
  postprocessing retention change on saved GEFFs.
- Do not retrain the edge scorer as the immediate next lever. Production recipe remains
  recipe C α=0.5 det=0.96875 safe-div ON.
- Artifacts: `outputs/experiments/candidate_edge_bottleneck_v1/`,
  `outputs/experiments/candidate_edge_bottleneck_holdout8/`,
  `outputs/analysis/candidate_bottleneck_synthesis.md`.

## Short-track filter ablation (2026-08-09)

- Postprocess-only on saved recipe-C det=0.96875 raw GEFFs (fixed-8 from
  `candidate_edge_bottleneck_v1/raw_geff`, holdout from
  `holdout8/det0_96875_safeon/raw_geff`). No re-inference / no scorer retrain.
- Control reproduced exactly: fixed-8 **0.884746427159**, holdout **0.959013051622**.
  Short-track removed 6637 / 6485 nodes respectively.
- Filter OFF: fixed-8 **−0.003097**, holdout **−0.001159**. Less aggressive
  `min_track_len` 4/3/2 also all negative on both sets. `min_len=2` ≡ filter OFF.
- Per-dataset OFF helps `44b6_0113de3b` but regresses most `6bba_*` (net FP rise).
- Decision: **KEEP CONTROL** short-track (`output_filter_short_tracks=true`,
  `output_min_track_len=6`, adaptive rescue ON). Do not promote retaining short tracks.
- Implication for bottleneck: `postprocessing_removed` is not a reason to disable
  short-track; next should stage-attribute those removals or pursue other mechanisms
  with short-track left ON.
- Artifacts: `outputs/experiments/shorttrack_ablation_det0_96875/`,
  `outputs/analysis/shorttrack_ablation_report.md`.

## Postprocess stage attribution + motion-relink OFF (2026-08-09/10)

- Traced `filter_output_graph` on bottleneck `postprocessing_removed` edges
  (`src/biohub_pipeline/postprocess_stage_trace.py`). Fixed-8: motion_relink
  46/57 (80.7%), short-track 11/57. Holdout: motion 13/23, short-track 10/23.
  No other stage was first-loss.
- Postprocess-only ablation: `output_motion_relink=false` → fixed-8
  **0.906725 (+0.021979)**, holdout **0.960307 (+0.001294)** vs motion-ON
  control 0.884746 / 0.959013. Mainly FP collapse (283→195 on fixed-8).
  Single-parent OFF is score-identical. **PROMOTE motion-relink OFF**.
- Historical baseline YAMLs with motion ON kept for reproducibility. Recommended
  config starts at `configs/experiments/recipe_c_motion_relink_off_det0_96875.yaml`.
- Artifacts: `outputs/experiments/postprocess_stage_attribution_v1/`,
  `outputs/experiments/ppstage_ablation_det0_96875/`.

## Edge-threshold lowering under motion OFF (2026-08-10)

- Under motion OFF, remaining `postprocessing_removed` is **100% short-track**
  (keep filter ON). Rediagnosis shifted burden toward ranking / candidate_threshold.
- Added optional `inference.edge_threshold` + predictor CLI patch
  (`apply_edge_threshold_cli_patch`). Default behavior unchanged when unset.
- edge 0.45: fixed-8 **0.915193 (+0.00847)**, holdout **0.962739 (+0.00243)**
  vs motion-OFF edge 0.5. **PROMOTE**.
- edge 0.40: fixed-8 **0.918144 (+0.00295)**, holdout **0.964673 (+0.00193)**
  vs edge 0.45. **PROMOTE**. Stop further blind threshold sweeps.
- Cumulative vs original recipe C (motion ON, edge 0.5): fixed-8 **+0.03340**,
  holdout **+0.00566**.
- Current recommended recipe:
  `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`
  (α=0.5, det=0.96875, edge_threshold=0.40, motion_relink OFF, safe-div ON,
  short-track ON). Do not silently overwrite
  `configs/sweeps/two_seed_det_thresh_0_96875.yaml`.
- Next major step: fresh candidate-edge capture on the promoted recipe before
  ranking retrain or ILP changes. Report:
  `outputs/analysis/autonomous_cycle_report.md`.

## Edge-gate sweep under motion-relink OFF (2026-08-10/12)

- Control: motion_relink OFF + edge_threshold=0.50 → fixed-8 **0.906725**, holdout **0.960307**.
- Full sweep 0.50/0.45/0.40/0.35/0.30/0.25 (only gate changed). Reused prior GPU
  results for 0.50–0.40; new GPU runs for 0.35–0.25 via
  `scripts/slurm/run_edge_gate_lower_sweep.sh`.
- Best robust gate: **0.40** → fixed **0.918144 (+0.01142)**, holdout **0.964673 (+0.00437)**.
- 0.35 improves fixed (+0.00110 vs 0.40) but regresses holdout (−0.00078) → **do not promote**.
- 0.30/0.25 worse than 0.40 on both sets; FP rise dominates.
- Band diagnostic (top-k capture): precision_proxy falls from ~0.75% (0.45–0.50) to
  ~0.19% (0.25–0.30). Useful recoverable signal concentrated in 0.40–0.50.
- Bottleneck rediagnosis at edge 0.40: candidate_threshold collapses (48→13 fixed,
  16→3 holdout). Ranking ~54%/50%. Remaining postprocessing_removed are 100%
  missing-endpoint (short-track); do not disable short-track again.
- Promoted config remains
  `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`.
  Historical motion-ON / edge-0.5 YAMLs unchanged.
- Next: fresh candidate-edge capture on the promoted recipe before ranking retrain.
- Artifacts: `outputs/experiments/edge_gate_sweep_motion_off_v1/`,
  `outputs/experiments/bottleneck_rediagnose_edge_0_40_v1/`,
  `outputs/analysis/edge_gate_sweep_report.md`.

## Fresh candidate capture @ promoted recipe (2026-08-12)

- Fresh instrumented top-16 captures under
  `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`
  (motion_relink OFF, edge_threshold=0.40, det=0.96875, α=0.5).
- Scores exact match: fixed-8 **0.9181439782806684**, holdout-8 **0.9646726188580379**.
- Causal ordinary-association ranking shares: fixed **54.3%** (70/129), holdout
  **50.0%** (26/52), combined **53.0%** (96/181). **SUPPORTED** for ~50% ranking claim
  among endpoint-matched failures.
- Rank-2 near misses: 46/70 fixed, 21/26 holdout. All `postprocessing_removed` still
  missing final endpoints (do not disable short-track).
- Including detection_miss, ranking is only 42%/30%/38% of all ordinary errors;
  holdout Candidate/gating (detection_miss) is 40% of all-error mass.
- NFS (compute home, not login): `~/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`.
  Login mirror: `outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`.
  Report: `outputs/analysis/fresh_candidate_capture_edge_0_40_report.md`.
- Recommended next (not run): one targeted edge-scorer ranking improvement under
  frozen recipe; no further gate sweeps / short-track OFF.

## Margin-gated distance rank REJECT (2026-08-12)

- Rank-2 OA failures: GT ~2.3× farther than competitor (~88% GT farther); dense scenes;
  softmax gap small vs successes. ILP has no distance term.
- Intervention D: pre-softmax margin-gated `logit += 0.1*dist_um` when top1−top2 < 0.15 and
  dens≥8 (15 µm). Frozen recipe unchanged (motion OFF, edge=0.40).
- Offline capture fit overstated benefit; GPU fixed-8 **0.917136 (−0.00101)** and holdout
  **0.964022 (−0.00065)** vs 0.918144 / 0.964673. Edge TP fell on both splits.
- **REJECT.** Keep promoted config without `margin_gated_dist_*`. Opt-in code retained for
  reproducibility only.
- Next: learned pairwise / hard-neg ranking under the same frozen recipe — not another
  hand-crafted distance bonus.
