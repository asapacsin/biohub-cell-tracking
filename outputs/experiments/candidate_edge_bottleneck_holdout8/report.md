# Candidate-edge bottleneck experiment

## Final classification: D. HYPOTHESIS REJECTED

## Hypothesis and predeclared rule

The learned edge scorer is the dominant ordinary-association failure mechanism, rather
than node detection, the 0.5 candidate threshold, ILP/global assignment, or final
postprocessing. Support required at least 20 endpoint-matched causal errors, at least
60% attributed to scorer ranking, and a lead of at least 20 percentage points over the
next mechanism. Ranking below 40%, or any competing mechanism at 40% or more, rejected it.

## Control and experiment

- Control: generalization-safe two-seed recipe C, alpha=0.5, det=0.96875, safe divisions
  ON, gap2 OFF, DeepCenter OFF; reference score `0.959013051622`.
- Experiment: identical inference/postprocessing with opt-in top-16 pre-gate blended and
  per-seed score capture. Instrumentation does not alter candidate construction or ILP.
- Instrumented score: `0.959013051622`; delta from control
  `+0.000000000000`.
- Runtime: `17.25` minutes.
- GPU inference required: yes, one fixed-8 run.

## Main causal result

- Ordinary associations: 4116
- Endpoint-matched causal errors: 67
- Scorer-ranking failures: 22
  (32.84%)
- Candidate-threshold failures: 8
  (11.94%)
- ILP/global failures: 9
  (13.43%)
- Postprocessing removals: 23
  (34.33%)
- Detection misses: 35
- Candidate recall with detected endpoints: 97.43%
- Final ordinary-edge recall with detected endpoints: 98.36%

## Dense-scene and family checks

| group_type | group | ordinary_associations | errors | error_rate | causal_errors | scorer_ranking | candidate_threshold | ilp_global | postprocessing_removed | candidate_recall | target_top1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_family | 44b6 | 1111 | 38 | 0.0342034 | 35 | 15 | 5 | 5 | 7 | 0.942238 | 0.958484 |
| dataset_family | 6bba | 3005 | 64 | 0.0212978 | 32 | 7 | 3 | 4 | 16 | 0.986209 | 0.988227 |
| density_bin | medium_high | 650 | 14 | 0.0215385 | 12 | 3 | 1 | 2 | 5 | 0.964506 | 0.973765 |
| density_bin | medium_low | 595 | 20 | 0.0336134 | 7 | 2 | 0 | 2 | 3 | 0.9811 | 0.986254 |
| density_bin | high | 999 | 46 | 0.046046 | 40 | 15 | 7 | 4 | 11 | 0.941591 | 0.956697 |
| density_bin | low | 1872 | 22 | 0.0117521 | 8 | 2 | 0 | 1 | 4 | 0.993003 | 0.993003 |

## Per dataset

| group | ordinary_associations | errors | error_rate | scorer_ranking | candidate_threshold | ilp_global | postprocessing_removed | detected_node_recall | final_node_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44b6_0c582fdc | 70 | 2 | 0.0285714 | 1 | 1 | 0 | 0 | 1 | 1 |
| 44b6_0db75fae | 151 | 7 | 0.0463576 | 0 | 0 | 0 | 7 | 1 | 0.955414 |
| 44b6_12dfb391 | 771 | 29 | 0.0376135 | 14 | 4 | 5 | 0 | 0.997462 | 0.996193 |
| 44b6_144b256d | 119 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 6bba_062c8d37 | 896 | 16 | 0.0178571 | 1 | 0 | 1 | 7 | 0.995699 | 0.990323 |
| 6bba_07477033 | 602 | 8 | 0.013289 | 2 | 1 | 2 | 2 | 1 | 0.998369 |
| 6bba_07e24132 | 341 | 35 | 0.102639 | 4 | 2 | 0 | 6 | 0.952646 | 0.941504 |
| 6bba_085bf656 | 1166 | 5 | 0.00428816 | 0 | 0 | 1 | 1 | 0.998331 | 0.996661 |

## Engineering decision

This experiment changes no production setting. The result determines whether the next
single action should target learned edge ranking or a different stage.

**Next action:** Hold raw GEFF predictions fixed and test one targeted postprocessing retention change on fixed-8 plus an independent holdout.

See the machine-readable `decision.json` for the predeclared classification inputs.
