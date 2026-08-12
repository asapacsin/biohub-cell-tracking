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
