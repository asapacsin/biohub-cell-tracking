# Selected-edge confidence calibration

## Final classification: D. HYPOTHESIS REJECTED

## Hypothesis and predeclared rule

Selected-edge confidence alone can identify a sufficiently concentrated intervention
set to justify confidence calibration as the next production lever. The fixed budget
is 5.0% of scored ordinary associations. Support
requires at least 40.0% error capture and correctness
AUROC of at least 0.80.

## Control and experiment

- Control: the existing selected-edge scores from the four labeled public two-seed
  overlaps in `association_density_diagnostic.csv`.
- Experiment: rank scored ordinary associations from lowest to highest selected-edge
  confidence and inspect the predeclared bottom-budget set.
- Neural inference required: no; this is a CPU-only analysis of saved artifacts.
- Production inference and postprocessing were not changed.

## Result

- Scored ordinary associations: 2,080
- Scored errors: 48
- Intervention budget: 104 associations
- Captured errors: 13 / 48
  (27.08%)
- Intervention precision: 12.50%
- Precision lift over random: 5.42x
- Correctness AUROC: 0.752543
- Maximum selected score inside the intervention set:
  0.657264173

The low-confidence tail is enriched for errors, but it misses nearly three quarters
of scored errors and fails both predeclared practical thresholds. Confidence-only
calibration is therefore not the next production lever.

## Per dataset

| group | scored_associations | scored_errors | error_prevalence | budget_count | captured_errors | error_capture | intervention_precision | correctness_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44b6_0113de3b | 50 | 0 | 0 | 3 | 0 | 0 | 0 |  |
| 44b6_0b24845f | 47 | 0 | 0 | 3 | 0 | 0 | 0 |  |
| 6bba_05b6850b | 838 | 7 | 0.00835322 | 42 | 1 | 0.142857 | 0.0238095 | 0.82362 |
| 6bba_05db0fb1 | 1145 | 41 | 0.0358079 | 58 | 13 | 0.317073 | 0.224138 | 0.701617 |

## Decision

Do not change the production/Kaggle recipe. Keep the production recipe unchanged and target edge-ranking quality rather than confidence-only calibration.
