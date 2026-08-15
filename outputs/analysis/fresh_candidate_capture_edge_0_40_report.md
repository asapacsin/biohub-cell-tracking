# Fresh candidate-edge capture — recipe C edge=0.40 motion OFF

**Experiment:** `candidate_capture_recipe_c_edge_0_40_v1`  
**Date:** 2026-08-12  
**Question:** Does ranking account for ~50% of remaining ordinary-association failures under this exact recipe?

## Recipe verified

| Setting | Value |
| --- | --- |
| Config | `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml` |
| `output_motion_relink` | **false** |
| `edge_threshold` | **0.40** (`--edge-threshold 0.4` on CLI) |
| `detection_threshold` | 0.96875 |
| Ensemble | α=0.5 two-seed |
| Primary ckpt SHA-256 | `12f6881ee3620a83…` (split_0) |
| Secondary ckpt SHA-256 | `9bac2fa0dadc4a6f…` (seed_314159) |
| Instrumentation | top-k=16 pre-gate capture; scores bit-identical to control |

**Fixed-8 command** (via `scripts/run_candidate_bottleneck_experiment.py`):

```text
scripts/slurm/run_fresh_candidate_capture_edge_0_40.sh
→ python scripts/run_candidate_bottleneck_experiment.py \
    --config configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml \
    --top-k 16 --control-score 0.9181439782806684
```

Predictor CLI includes `--edge-threshold 0.4`, `--det-threshold 0.96875`,
`--ensemble-alpha 0.5`, `--use-ilp`, `--edge-diagnostic-dir … --edge-diagnostic-top-k 16`.

**Holdout-8 command:** same launcher, second stage with
`--skip-fixed8-validation --control-score 0.9646726188580379` and the eight holdout stems
`44b6_0c582fdc … 6bba_085bf656`.

## Scores

| Split | Score | Expected | Match |
| --- | ---: | ---: | --- |
| fixed-8 | **0.9181439782806684** | 0.9181439782806684 | exact (`prediction_equivalent_to_control=true`) |
| holdout-8 | **0.9646726188580379** | 0.9646726188580379 | exact |

Instrumentation did not change graph construction or scores.

## Fresh attribution

Taxonomy map (ordinary associations only):

| User A–E | Native cause(s) |
| --- | --- |
| Candidate/gating | `detection_miss` |
| Ranking | `scorer_ranking` |
| Threshold | `candidate_threshold` |
| Assignment/consistency | `ilp_global` + `postprocessing_removed` + `postprocessing_rematch` |
| Other | residual (none observed) |

### A. All ordinary-association errors (includes detection misses)

| Category | fixed-8 | holdout-8 | combined % |
| --- | ---: | ---: | ---: |
| Candidate/gating | 37 (22.3%) | 35 (40.2%) | 28.5% (72/253) |
| Ranking | 70 (42.2%) | 26 (29.9%) | 37.9% (96/253) |
| Threshold | 13 (7.8%) | 3 (3.4%) | 6.3% (16/253) |
| Assignment/consistency | 46 (27.7%) | 23 (26.4%) | 27.3% (69/253) |
| Other | 0 (0%) | 0 (0%) | 0% |

### B. Causal / endpoint-matched errors (prior ~50% claim definition)

Detection misses excluded; shares over ordinary associations where both GT endpoints were detected.

| Category | fixed-8 | holdout-8 | combined % |
| --- | ---: | ---: | ---: |
| Candidate/gating | 0 (0%) | 0 (0%) | 0% |
| Ranking | **70 (54.3%)** | **26 (50.0%)** | **53.0%** (96/181) |
| Threshold | 13 (10.1%) | 3 (5.8%) | 8.8% (16/181) |
| Assignment/consistency | 46 (35.7%) | 23 (44.2%) | 38.1% (69/181) |
| Other | 0 (0%) | 0 (0%) | 0% |

Causal error counts: fixed **129**, holdout **52**, combined **181**.

Native breakdown (fixed / holdout): ranking 70/26, threshold 13/3, ILP 9/9,
pp_removed 19/8, pp_rematch 18/6. All `pp_removed` rows still have a missing final
endpoint (19/19 and 8/8) — short-track cascade, not edge-level deletion with both
nodes retained.

## Ranking diagnostics

### GT target-rank distribution among ranking errors

| Rank | fixed-8 | holdout-8 |
| ---: | ---: | ---: |
| 2 | 46 (65.7%) | 21 (80.8%) |
| 3 | 16 | 4 |
| 4 | 4 | 1 |
| 5+ | 4 | 0 |

### Near-miss / score-margin (ranking errors)

| Metric | fixed-8 | holdout-8 |
| --- | ---: | ---: |
| Rank-2 near misses | 46 / 70 (65.7%) | 21 / 26 (80.8%) |
| Competitor−GT margin median | 0.406 | 0.437 |
| Margin mean | 0.429 | 0.434 |
| Margin p25 / p75 | 0.175 / 0.699 | 0.210 / 0.623 |

### Ranking-error rate conditional on GT edge existing in capture

Among endpoint-matched ordinary associations whose GT pair was recorded in the
top-k ∪ gated capture:

| Split | Recorded GT edges | Ranking errors | Rate |
| --- | ---: | ---: | ---: |
| fixed-8 | 4052 | 70 | **1.73%** |
| holdout-8 | 4081 | 26 | **0.64%** |

Top-1 / top-2 / top-3 rates among recorded GT edges: fixed ≈ high (see
`fresh_attribution.json` ranking_diagnostics); ranking failures are a small
absolute rate but dominate the remaining causal error mass.

## Consistency checks

- Recipe flags present in both `experiment_config.yaml` copies.
- Both CLIs contain `--edge-threshold 0.4` and diagnostic flags.
- Score exact match to promoted recipe; `prediction_equivalent_to_control=true`.
- Recomputed causal ranking shares match `decision.json` exactly.
- Sample rows: ranking errors have `target_rank≥2`; threshold errors have rank=1 and
  `blended_prob≤0.40`; ILP errors have rank=1 and above threshold but not ILP-selected;
  pp_rematch has `final_pair_present=true` but wrong final mapping.

## Conclusion

**SUPPORTED** (causal / endpoint-matched definition used by the prior ~50% claim):

| Split | Ranking share of causal ordinary failures |
| --- | ---: |
| fixed-8 | **54.3%** (70/129) |
| holdout-8 | **50.0%** (26/52) |
| combined | **53.0%** (96/181) |

Fresh captures under the exact promoted recipe reproduce the prior rediagnosis
numbers. Ranking is the largest single causal mechanism on both splits.

Caveat: if **all** ordinary errors including detection misses are pooled, ranking
is 42.2% / 29.9% / 37.9% — then Candidate/gating is large on holdout (40.2%).
The ~50% statement applies to remaining failures **after endpoints are detected**.

## Recommended next experiment

**One targeted edge-scorer ranking improvement** (retrain or re-rank) evaluated on
fixed-8 + holdout-8 under the frozen recipe
`recipe_c_motion_off_edge_0_40_det0_96875` (motion OFF, edge=0.40). Do not reopen
short-track OFF or further edge-gate sweeps. Prefer interventions that lift GT
pairs currently at rank 2 (65–81% of ranking errors).

*(Not executed in this session.)*

## Artifacts

- NFS (compute home): `~/biohub-outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`
  (full captures + raw GEFFs; login home ≠ this NFS)
- Login mirror: `outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/`
- Attribution JSON: `…/fresh_attribution.json`
- This report: `outputs/analysis/fresh_candidate_capture_edge_0_40_report.md`
- Launcher: `scripts/slurm/run_fresh_candidate_capture_edge_0_40.sh`
- Attribution script: `scripts/attribute_fresh_candidate_capture.py`
