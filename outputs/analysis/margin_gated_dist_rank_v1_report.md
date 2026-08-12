# Margin-gated distance rank term — REJECT

**Experiment:** `margin_gated_dist_rank_v1`  
**Date:** 2026-08-12  
**Frozen recipe:** `recipe_c_motion_off_edge_0_40_det0_96875` (motion OFF, edge=0.40 unchanged)

## Decision: **REJECT**

Do not enable the margin-gated distance term. Both fixed-8 and holdout-8 scores fell.

## Compact weakness (from fresh capture)

Among ordinary-association **rank-2** failures (fixed 46 / holdout 21):

| Signal | Rank-2 fails | Successful rank-1 |
| --- | ---: | ---: |
| Median GT dist (µm) | ~6.1 | (much shorter typical) |
| Median competitor dist (µm) | ~2.3 | — |
| GT farther than competitor | **~88%** | — |
| Local density median | 9–10.5 | ~3–5 |
| Softmax top1−top2 logit gap median | 0.74 / 0.85 | 3.65 / 4.35 |

**Weakness:** column-softmax ranking prefers proximal distractors; ILP uses only `edge_prob` (no distance). Global `+λ·dist` destroys short true edges; only near-tie dense columns are candidates for a deterministic correction.

Offline fit (capture top-k only) suggested `λ=0.1`, `Δ=0.15`, dens≥8: ~3/46 + 3/21 rank-2 flips with ≤4 success drops. That did **not** survive full inference.

## Intervention (option D)

Pre-softmax, per target column:

- if `(top1_logit − top2_logit) < 0.15` and local detection density (≥15 µm) `≥ 8`
- then `logit ← logit + 0.1 · dist_um`

Implemented as opt-in support-patch + config keys (promoted recipe YAML untouched).

Config: `configs/experiments/recipe_c_edge_0_40_margin_gated_dist_v1.yaml`  
Runner: `scripts/slurm/run_margin_gated_dist_rank_v1.sh`

## Results

| Split | Baseline | Candidate | Δ |
| --- | ---: | ---: | ---: |
| fixed-8 | 0.9181439782806684 | **0.9171355476259768** | **−0.001008** |
| holdout-8 | 0.9646726188580379 | **0.9640223216403617** | **−0.000650** |

| Split | Baseline edge TP/FP/FN | Candidate TP/FP/FN | Δ |
| --- | --- | --- | --- |
| fixed-8 | 3949 / 187 / 154 | 3941 / 185 / 162 | −8 / −2 / +8 |
| holdout-8 | 4037 / 76 / 89 | 4035 / 77 / 91 | −2 / +1 / +2 |

Per-dataset: fixed 2↑ / 4↓ / 2→; holdout 2↑ / 3↓ / 3→. Largest fixed regression `44b6_e57ff5c6` (−0.0129); largest holdout gain `44b6_0c582fdc` (+0.026) insufficient to offset losses.

Recipe flags verified on command: `--edge-threshold 0.4`, motion OFF in config, `--margin-gated-dist-lambda 0.1`.

## Causal interpretation

Offline top-k re-ranking overstated benefit. Full-matrix softmax renormalization + ILP/postprocess made the term **net harmful**: true edges lost (FN↑, TP↓) on both splits. Near-tie distance boosts also elevate wrong long distractors, not only GT rank-2 cases. No evidence of fixed↑/holdout↓ overfit asymmetry — both declined together.

No further coeff tweak justified: smaller λ offline already had almost no holdout flips; larger gates increase success drops sharply.

## Artifacts

- Login: `outputs/experiments/margin_gated_dist_rank_v1/`
- NFS: `~/biohub-outputs/experiments/margin_gated_dist_rank_v1/`
- Offline diagnosis: `outputs/analysis/rank2_near_miss_v1/`, `outputs/analysis/margin_gated_distance_fit_v1/`
- Code: `src/biohub_pipeline/margin_gated_distance_rank.py`, patch in `inference.py` (opt-in only)

## Recommended next experiment (not executed)

**Pairwise / hard-negative edge-scorer fine-tune** (or a learned pairwise re-ranker) targeting dense scenes where GT is the longer of the top-2 — using capture logits/features as mining seed. Keep frozen recipe (motion OFF, edge=0.40). Avoid further hand-crafted distance bonuses without appearance features.
