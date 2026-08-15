# Edge-scorer hard-neg retrain v1 — REJECT

**Experiment:** `edge_scorer_hardneg_retrain_v1`
**Frozen recipe:** motion-relink OFF, `edge_threshold=0.40` (unchanged)
**Job:** slurm `7104`  **Commit:** `e21070f8f7270d17aa3a8ab1b96f22b0708dee32`

## Decision: **REJECT**

Do not start another edge-scorer retrain. Inspect remaining ordinary-association failures on holdout 44b6_12dfb391 with frozen detections (ILP / candidate set), not another ranking-weight or architecture sweep.

## What changed

Fine-tuned both two-seed **edge transformers** with pairwise ranking loss on
rank-2 near-misses + controls mined from the frozen-recipe capture
(`candidate_capture_recipe_c_edge_0_40_v1`, fixed-8 only). UNet / detect_head
frozen. Holdout-8 was not used for training. Linear pairwise reweight was not
re-enabled.

Eval config: `configs/experiments/recipe_c_edge_0_40_hardneg_retrain_v1.yaml`

## Scores

| Split | Frozen baseline | Hard-neg retrain | Δ |
| --- | ---: | ---: | ---: |
| fixed-8 | 0.9181439782806684 | 0.9155928318042139 | -0.002551 |
| holdout-8 | 0.9646726188580379 | 0.9600674680962610 | -0.004605 |

Edges TP/FP/FN: fixed [3965, 198, 138] ; holdout [4042, 88, 84]

## Causal ranking

| Split | ranking (base → new) | rank-2 (base → new) | causal OA ranking share |
| --- | --- | --- | --- |
| fixed-8 | 70 → 51 | 46 → 36 | 0.5426356589147286 → 0.4636363636363636 |
| holdout-8 | 26 → 23 | 21 → 18 | 0.5 → 0.4791666666666667 |

### `44b6_12dfb391`

- baseline adj=0.9347073959325116 TP/FP/FN=745/41/28
- new adj=0.9306093037330831 TP/FP/FN=746/43/27

## Artifacts

- NFS: `~/biohub-outputs/experiments/edge_scorer_hardneg_retrain_v1/`
- Login compact: `outputs/experiments/edge_scorer_hardneg_retrain_v1/`
- Report: `outputs/analysis/edge_scorer_hardneg_retrain_v1_report.md`
