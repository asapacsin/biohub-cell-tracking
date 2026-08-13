# Pairwise hard-neg ranking v1 — REJECT

**Experiment:** `pairwise_hardneg_rank_v1`  
**Date:** 2026-08-13  
**Frozen recipe:** `recipe_c_motion_off_edge_0_40_det0_96875` (motion OFF, edge=0.40 unchanged)  
**GPU:** Slurm 6572 on um-gpu01, 2026-08-12T18:18–18:50 UTC (~33 min). Compact login mirror recovered 2026-08-13 via `srun tar` (job 6574 tar-exclude failed; NFS still held full scores).

## Decision: **REJECT**

Do not enable pairwise hard-neg reweight in the production recipe. Fixed-8 improved; holdout-8 regressed. Same promote rule as edge-gate 0.35: both splits must not go negative.

## Intervention

Pre-softmax, per target column, if density (≥15 µm) `≥ 8` and `(top1−top2) < 1.5`:

`score = blended_logit + w · φ(seed_disagree, seed_min, dist_n, log_dens, dist×disagree, dist×seed_min)`

Weights from capture-fit (fixed-8 hard-neg search, offline holdout check):

| feature | w |
| --- | ---: |
| seed_disagree | +0.0605 |
| seed_min | −0.2087 |
| dist_n | +0.3215 |
| log_dens | +0.0103 |
| dist_x_disagree | +0.0501 |
| dist_x_seedmin | +0.0892 |

Offline (capture top-k only): rank-2 flips 4/46 fixed, 2/21 holdout; success drops 0/0. Applied also to 137/3853 fixed and 90/3974 holdout **success** columns — too loose a gap gate for full softmax+ILP.

Config: `configs/experiments/recipe_c_edge_0_40_pairwise_hardneg_v1.yaml`  
Runner: `scripts/slurm/run_pairwise_hardneg_rank_v1.sh`  
Promoted recipe YAML untouched. Opt-in code retained for reproducibility only.

CLI verified: `--edge-threshold 0.4`, `--pairwise-hardneg-weights […]`, `--det-threshold 0.96875`, no `--margin-gated-dist-lambda`, motion OFF in config.

## Results

| Split | Baseline | Candidate | Δ |
| --- | ---: | ---: | ---: |
| fixed-8 | 0.9181439782806684 | **0.9242113721438376** | **+0.006067** |
| holdout-8 | 0.9646726188580379 | **0.9638461817004692** | **−0.000826** |

| Split | Baseline edge TP/FP/FN | Candidate TP/FP/FN | Δ |
| --- | --- | --- | --- |
| fixed-8 | 3949 / 187 / 154 | 3946 / 167 / 157 | −3 / −20 / +3 |
| holdout-8 | 4037 / 76 / 89 | 4032 / 78 / 94 | −5 / +2 / +5 |

Divisions unchanged in TP (0); fixed div FP 10→13; holdout div FP 5→6.

Per-dataset adj_edge_jaccard vs edge-0.40 control:

- **fixed-8:** 6↑ / 1↓ / 1→. Largest gains `6bba_fc83837d` (+0.0199, FP 70→55) and `44b6_e57ff5c6` (+0.0155). Only regression `44b6_341df25f` (−0.0046, FP 0→1).
- **holdout-8:** 5↑ / 1↓ / 2→. Entire net loss is `44b6_12dfb391` (−0.0101; TP 745→740, FP 41→45, FN 28→33). Largest holdout gain `6bba_07e24132` (+0.0116, FP 18→16).

## Causal interpretation

Offline top-k flips overstated generalization. Full-matrix softmax + ILP converted a dense-scene FP collapse on fixed-8 (especially `6bba_fc83837d`) into a **one-dataset holdout regression** on `44b6_12dfb391`. Dominant weight is `dist_n` (+0.32): same long-distractor risk as the rejected hand-crafted distance bonus, now appearance-gated but still over-applied (`gap_max=1.5` touches many success columns).

Not an ambiguous promote: standing rule requires non-negative holdout. Magnitude matches the rejected edge-0.35 holdout dip (−0.00078).

## Artifacts

- Login compact: `outputs/experiments/pairwise_hardneg_rank_v1/` (`comparison.json`, `fixed8/` + `holdout8/` summaries)
- NFS (compute home): `~/biohub-outputs/experiments/pairwise_hardneg_rank_v1/` (includes raw GEFFs + submissions)
- Offline fit: `outputs/analysis/pairwise_hardneg_rank_v1/`
- Logs: `logs/slurm/pairwise_hardneg_rank_v1.log`
- Code: `src/biohub_pipeline/pairwise_hardneg_rank.py`, opt-in patch in `inference.py`

## Recommended next experiment (not a recipe change)

**`pairwise_hardneg_rank_v1_scale0_5`:** identical frozen recipe and gates (`dens≥8`, `gap_max=1.5`), weights ×0.5. Tests whether the fixed-8 FP collapse can be kept without the `44b6_12dfb391` holdout hit. One GPU fixed-8+holdout-8 eval; do not grid-search coefficients.

If that also fails holdout, close linear pairwise reweight (two ranking interventions REJECTED) and move to the next remaining causal mass: holdout detection_miss (40% of all ordinary errors) or assignment/consistency (ILP + rematch, 38% of causal). Do not reopen short-track OFF or edge-gate sweeps.
