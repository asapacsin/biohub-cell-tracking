# Pairwise hard-neg ranking 0.5× salvage — REJECT

**Experiment:** `pairwise_hardneg_rank_v1_scale0_5`  
**Date:** 2026-08-13  
**Frozen recipe:** `recipe_c_motion_off_edge_0_40_det0_96875` (motion OFF, edge=0.40 unchanged)  
**GPU:** Slurm 6691 on um-gpu01, 2026-08-13T05:22–05:55 UTC (~33 min). Compact login mirror via in-job `srun tar`.

## Decision: **REJECT**

Do not enable 0.5× pairwise hard-neg reweight. Fixed-8 still improves; holdout-8 still regresses. Close linear pairwise reweight (v1 and 0.5× both fail the same promote rule). Do not keep scaling weights.

## Intervention

Same dens/gap gates as v1 (`dens≥8`, `gap_max=1.5`, 15 µm). Weights ×0.5:

| feature | w (v1) | w (×0.5) |
| --- | ---: | ---: |
| seed_disagree | +0.0605 | +0.0302 |
| seed_min | −0.2087 | −0.1044 |
| dist_n | +0.3215 | +0.1608 |
| log_dens | +0.0103 | +0.0051 |
| dist_x_disagree | +0.0501 | +0.0251 |
| dist_x_seedmin | +0.0892 | +0.0446 |

Config: `configs/experiments/recipe_c_edge_0_40_pairwise_hardneg_v1_scale0_5.yaml`  
Runner: `scripts/slurm/run_pairwise_hardneg_rank_v1_scale0_5.sh`  
Promoted recipe YAML untouched. Opt-in code retained for reproducibility only.

CLI verified: `--edge-threshold 0.4`, `--pairwise-hardneg-weights […]` (halved), `--det-threshold 0.96875`, no `--margin-gated-dist-lambda`, motion OFF.

## Results

| Split | Frozen baseline | v1 | 0.5× | 0.5× vs frozen | 0.5× vs v1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-8 | 0.9181439782806684 | 0.9242113721438376 | **0.924009074688947** | **+0.005865** | −0.000202 |
| holdout-8 | 0.9646726188580379 | 0.9638461817004692 | **0.963838933678889** | **−0.000834** | −0.000007 |

| Split | Baseline TP/FP/FN | v1 | 0.5× | 0.5× Δ vs frozen |
| --- | --- | --- | --- | --- |
| fixed-8 | 3949 / 187 / 154 | 3946 / 167 / 157 | 3954 / 169 / 149 | +5 / −18 / −5 |
| holdout-8 | 4037 / 76 / 89 | 4032 / 78 / 94 | 4035 / 79 / 91 | −2 / +3 / +2 |

Divisions: fixed FP 10→11 (v1 was 13); holdout FP 5→6 (same as v1). TP still 0.

Per-dataset adj_edge_jaccard vs frozen edge-0.40 control:

- **fixed-8:** 6↑ / 1↓ / 1→. Gain is almost entirely `6bba_fc83837d` (+0.0228; TP 857→862, FP 70→53). Tiny regressions/gains elsewhere; `44b6_e57ff5c6` keeps baseline TP/FP (168/24) and loses v1's +0.0155.
- **holdout-8:** 4↑ / 1↓ / 3→. Entire net loss is still `44b6_12dfb391` (−0.00561). Other holdout datasets are flat or slightly up; `6bba_07e24132` only +0.0028 vs v1's +0.0116.

## Causal interpretation (`44b6_12dfb391`)

| | adj | TP / FP / FN | Δ adj vs frozen |
| --- | ---: | --- | ---: |
| frozen | 0.934707 | 745 / 41 / 28 | — |
| v1 | 0.924583 | 740 / 45 / 33 | −0.01012 |
| 0.5× | 0.929101 | 743 / 44 / 30 | **−0.00561** |

Halving weights recovered about half of the v1 damage on `44b6_12dfb391` (FN 33→30, FP 45→44, TP 740→743) but **did not restore the frozen baseline**. The leftover −0.0056 on this one dense FOV still outweighs the small holdout gains, so net holdout stays ~v1 (−0.00083).

Same mechanism as v1: dominant `dist_n` term (now +0.16) still over-applies in dense softmax+ILP on this embryo FOV. Scaling is not a new causal lever — it interpolates between frozen and v1 and lands on the wrong side of the promote rule.

Linear pairwise reweight is **closed**. Further ×0.25 / coefficient grids would be the same interpolation.

## Artifacts

- Login compact: `outputs/experiments/pairwise_hardneg_rank_v1_scale0_5/` (`comparison.json`, `fixed8/` + `holdout8/` summaries)
- NFS (compute home): `~/biohub-outputs/experiments/pairwise_hardneg_rank_v1_scale0_5/`
- Logs: `logs/slurm/pairwise_hardneg_rank_v1_scale0_5.log`
- Job: slurm **6691**, nohup PID **2122368** (COMPLETED 0:0)

## Recommended next experiment (not executed)

**Learned edge-scorer retrain with pairwise hard-negative mining** under the frozen recipe (`motion OFF`, `edge_threshold=0.40`). Ranking remains ~50% of causal ordinary failures after endpoints are detected; two linear logit add-ons (margin-dist, pairwise `w·φ`) both failed holdout on `44b6_12dfb391`. A retrained scorer is a major scope leap — **recommend only, do not launch from this session**.

Do not reopen short-track OFF, edge-gate sweeps, or further linear pairwise weight scales.
