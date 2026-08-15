# Running experiment log

## E1 — Short-track retention ablation (2026-08-09)
- **Hypothesis:** Harmful `postprocessing_removed` errors come from short-track filtering; relaxing it recovers true tracks.
- **Change:** `output_filter_short_tracks` OFF / min_len 4/3/2 vs control min_len=6.
- **Compute:** postprocess-only on saved GEFFs (srun GPU node, no re-inference).
- **Control:** recipe C α=0.5 det=0.96875 safe-div ON — fixed-8 **0.884746** / holdout **0.959013**.
- **Result:** all relaxations worse (OFF fixed-8 −0.00310, holdout −0.00116).
- **Conclusion:** **REJECT.** Keep short-track ON.
- **Next:** stage-attribute `postprocessing_removed`.

## E2 — Postprocess stage attribution (2026-08-09)
- **Hypothesis:** A non-short-track stage causes most harmful removals.
- **Change:** traced `filter_output_graph` checkpoints on bottleneck `postprocessing_removed` edges.
- **Compute:** postprocess-only.
- **Result:** fixed-8 motion_relink 46/57 (80.7%), short-track 11/57; holdout motion 13/23, short-track 10/23. No other stage.
- **Conclusion:** **SUPPORTED** for motion-relink as dominant removal stage among these labels.
- **Next:** ablate motion-relink OFF.

## E3 — Motion-relink / single-parent ablation (2026-08-10)
- **Hypothesis:** Disabling motion-relink recovers true edges net-positive.
- **Change:** `output_motion_relink=false` (+ single-parent OFF checks).
- **Compute:** postprocess-only on saved GEFFs.
- **Control:** same as E1.
- **Result:** motion OFF fixed-8 **0.906725 (+0.02198)**, holdout **0.960307 (+0.00129)**; FP 283→195. Single-parent OFF noop.
- **Conclusion:** **ACCEPT / PROMOTE** motion-relink OFF (new recommended recipe). Do not silently rewrite historical baseline YAMLs.
- **Next:** rediagnose causes under motion OFF.

## E4 — Bottleneck rediagnosis + remaining-stage attr (2026-08-10)
- **Hypothesis:** After motion OFF, remaining postprocess removals are short-track; ranking/threshold dominate.
- **Change:** re-run cause taxonomy on motion-OFF finals; attribute remaining `postprocessing_removed`.
- **Result:** remaining pp_removed = **100% short_track** (fixed 23, holdout 14). Ranking ~35–42%; candidate_threshold rose (fixed 48, holdout 16) with probs in [0.25,0.50).
- **Conclusion:** postprocess retention path closed. Next lever = pre-ILP edge gate.
- **Next:** edge_threshold 0.5→0.45 under motion OFF (GPU).

## E5 — Edge threshold 0.45 (2026-08-10)
- **Hypothesis:** Lowering edge_threshold to 0.45 recovers near-threshold true ordinary associations net-positive.
- **Change:** `inference.edge_threshold=0.45`, motion_relink OFF; optional `--edge-threshold` CLI patch.
- **Compute:** full two-seed GPU inference fixed-8 + holdout-8.
- **Control:** motion-OFF edge=0.5 scores from E3 (0.906725 / 0.960307).
- **Result:** fixed-8 **0.915193 (+0.00847)**; holdout **0.962739 (+0.00243)**.
- **Conclusion:** **ACCEPT / PROMOTE** edge_threshold=0.45.
- **Next:** one step to 0.40 vs this new control.

## E6 — Edge threshold 0.40 (2026-08-10)
- **Hypothesis:** Further lowering 0.45→0.40 recovers remaining sub-0.45 true edges net-positive.
- **Change:** `edge_threshold=0.40`, motion_relink OFF.
- **Compute:** full two-seed GPU inference fixed-8 + holdout-8.
- **Control:** E5 scores (0.915193 / 0.962739).
- **Result:** fixed-8 **0.918144 (+0.00295)**; holdout **0.964673 (+0.00193)**.
- **Conclusion:** **ACCEPT / PROMOTE** edge_threshold=0.40. Stop further threshold steps (diminishing returns).
- **Next:** fresh bottleneck capture on promoted recipe before ranking/ILP work.

## E7 — Full edge-gate sweep 0.50→0.25 (2026-08-12)
- **Hypothesis:** Useful near-threshold true edges exist in 0.25–0.50; lowering the gate recovers them net-positive.
- **Change:** only `inference.edge_threshold` under motion_relink OFF.
- **Compute:** reused 0.50/0.45/0.40 GPU results; new GPU for 0.35/0.30/0.25; band diagnostic from saved capture.
- **Control:** motion-OFF edge 0.5 (0.906725 / 0.960307).
- **Result:** best robust = **0.40** (0.918144 / 0.964673). 0.35 overfits (holdout −0.00078 vs 0.40).
- **Conclusion:** **KEEP edge_threshold=0.40**. Stop further gate sweeps.
- **Next:** fresh capture on promoted recipe for ranking validation.

## E8 — Fresh candidate-edge capture @ edge=0.40 motion OFF (2026-08-12)
- **Hypothesis:** Ranking accounts for ~50% of remaining ordinary-association (causal) failures under the promoted recipe.
- **Change:** none to recipe; new instrumented top-16 captures on fixed-8 + holdout-8.
- **Compute:** GPU `scripts/slurm/run_fresh_candidate_capture_edge_0_40.sh`.
- **Control / expected:** fixed **0.918144** / holdout **0.964673**.
- **Result:** exact score match both splits. Causal ranking **54.3% / 50.0% / 53.0%** combined. Rank-2 near misses 66%/81% of ranking errors. All pp_removed still missing-endpoint.
- **Conclusion:** **SUPPORTED** (~50% ranking among causal ordinary failures).
- **Next:** one targeted scorer ranking improvement under frozen recipe; do not reopen short-track OFF or gate sweeps.
- **Artifacts:** NFS `~/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`; login `outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`; report `outputs/analysis/fresh_candidate_capture_edge_0_40_report.md`.

## E9 — Margin-gated distance rank term (2026-08-12)
- **Hypothesis:** Softmax over-prefers proximal distractors on rank-2 OA near misses; a margin-gated `+λ·dist` in dense near-tie columns recovers them net-positive under frozen recipe.
- **Change:** opt-in pre-softmax `λ=0.1`, `Δ=0.15`, dens≥8 / 15µm; recipe motion OFF + edge=0.40 unchanged.
- **Compute:** GPU fixed-8 + holdout-8 (`scripts/slurm/run_margin_gated_dist_rank_v1.sh`); offline fit from fresh capture first.
- **Control:** 0.918144 / 0.964673.
- **Result:** fixed **0.917136 (−0.00101)**; holdout **0.964022 (−0.00065)**. Edges fixed 3941/185/162 (Δ −8/−2/+8); holdout 4035/77/91 (Δ −2/+1/+2).
- **Conclusion:** **REJECT.** Offline top-k flips did not survive full softmax+ILP; net FN increase on both splits.
- **Next:** learned pairwise / hard-neg ranking improvement (not another hand distance bonus).

## E10 — Pairwise hard-neg ranking v1 (2026-08-13)
- **Hypothesis:** Appearance-aware pairwise reweight (`w·φ` on seed disagreement / min, dist, density) in dense near-tie columns recovers rank-2 GT edges net-positive under the frozen recipe.
- **Change:** opt-in pre-softmax `w` from capture fit, dens≥8, gap_max=1.5 / 15µm; motion OFF + edge=0.40 unchanged. Not a distance-only bonus.
- **Compute:** GPU fixed-8 + holdout-8 (`scripts/slurm/run_pairwise_hardneg_rank_v1.sh`, job 6572, ~33 min).
- **Control:** 0.918144 / 0.964673.
- **Result:** fixed **0.924211 (+0.00607)**; holdout **0.963846 (−0.00083)**. Edges fixed 3946/167/157 (Δ −3/−20/+3); holdout 4032/78/94 (Δ −5/+2/+5). Holdout loss concentrated in `44b6_12dfb391` (−0.0101).
- **Conclusion:** **REJECT.** Fixed-8 FP collapse does not generalize; do not enable pairwise_hardneg in production.
- **Next:** one 0.5×-weight salvage under the same gates (`pairwise_hardneg_rank_v1_scale0_5`); if that also fails holdout, close linear pairwise reweight.

## E11 — Pairwise hard-neg ranking 0.5× salvage (2026-08-13)
- **Hypothesis:** Halving v1 pairwise weights keeps the fixed-8 dense-scene FP collapse without the `44b6_12dfb391` holdout hit.
- **Change:** same dens≥8 / gap_max=1.5 gates; `pairwise_hardneg_w` ×0.5. Motion OFF + edge=0.40 unchanged.
- **Compute:** GPU fixed-8 + holdout-8 (`scripts/slurm/run_pairwise_hardneg_rank_v1_scale0_5.sh`, job 6691, ~33 min). Did not duplicate; awaited in-flight job.
- **Control:** 0.918144 / 0.964673. Also vs v1 0.924211 / 0.963846.
- **Result:** fixed **0.924009 (+0.00587)**; holdout **0.963839 (−0.00083)**. `44b6_12dfb391` −0.00561 (half of v1's −0.0101, still net-negative).
- **Conclusion:** **REJECT.** Close linear pairwise reweight. Do not scale weights further.
- **Next (recommend only, not executed):** edge-scorer retrain with pairwise hard-neg mining under the frozen recipe.

## E12 — Edge-scorer hard-neg retrain v1 (2026-08-15)
- **Hypothesis:** Fine-tuning both two-seed edge transformers with pairwise ranking loss on fixed-8 rank-2 mines recovers remaining OA ranking failures net-positive under the frozen recipe.
- **Change:** UNet/detect_head frozen; holdout not mined; linear pairwise reweight off. Eval config `recipe_c_edge_0_40_hardneg_retrain_v1.yaml`.
- **Compute:** slurm 7104, um-gpu01, 54m53s.
- **Control:** 0.918144 / 0.964673.
- **Result:** fixed **0.915593 (−0.00255)**; holdout **0.960067 (−0.00461)**. `44b6_12dfb391` 0.934707 → 0.930609.
- **Conclusion:** **REJECT.** Do not install hardneg_v1 weights. Do not start another edge-scorer retrain.

## E13 — OA ILP / candidate-set audit on holdout 44b6_12dfb391 (2026-08-15)
- **Hypothesis:** Remaining OA failures on this holdout killer are ILP occupancy or a candidate-set hole, not scorer ranking inside the gated graph.
- **Change:** none to recipe. CPU audit of frozen-recipe capture + raw GEFF.
- **Result:** 62.5% of causal errors never enter ILP (softmax ≤ 0.40). Only 1/15 ranking rows is gated. Source-top-1 admit would add 5670 extras vs 7 GTs (810:1) — **REJECTED offline**. All 4 ILP errors are source_taken by a non-GT closer extra detection.
- **Conclusion:** **Close ILP / candidate-set branch.** Frozen recipe unchanged.
- **Next (not started):** detector-scope leap — holdout detection miss on `6bba_07e24132` (22/34). Not another ranking or ILP job.

## E14 — Detection-miss audit + frozen z-shift on 6bba_07e24132 (2026-08-15)
- **Hypothesis:** Holdout OA detection misses are missing peaks, recoverable by a global +z convention fix without a detector retrain.
- **Change:** none to recipe. CPU nearest-peak audit; z-shift sweep Δ∈{−2..+4} voxels on frozen postprocessed submissions (edges unchanged).
- **Control:** 0.918144 / 0.964673.
- **Result:** 17 unmatched GT, all with a nearby peak (median 8.64 µm, mostly −z). +1 voxel helps this dataset (recall 0.936→0.953) but fixed **−0.0140** / holdout **−0.0079**. Intensity z-COM does not walk toward GT.
- **Conclusion:** **REJECT** global z-shift. Do not retrain the detector for these misses. Frozen recipe unchanged.
- **Next:** stop. No remaining cheap both-split promote path on association or detection coordinates.
