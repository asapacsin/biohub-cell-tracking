# Candidate-edge bottleneck experiment

## Final classification: D. HYPOTHESIS REJECTED

## Hypothesis and predeclared rule

The learned edge scorer is the dominant ordinary-association failure mechanism, rather
than node detection, the 0.5 candidate threshold, ILP/global assignment, or final
postprocessing. Support required at least 20 endpoint-matched causal errors, at least
60% attributed to scorer ranking, and a lead of at least 20 percentage points over the
next mechanism. Ranking below 40%, or any competing mechanism at 40% or more, rejected it.

## Control and experiment

- Control: generalization-safe two-seed fixed-8, alpha=0.5, det=0.96875, safe divisions
  ON, gap2 OFF, DeepCenter OFF; historical score `0.884746427159`.
- Experiment: identical inference/postprocessing with opt-in top-16 pre-gate blended and
  per-seed score capture. Instrumentation does not alter candidate construction or ILP.
- Instrumented score: `0.884746427159`; delta from control
  `+0.000000000000`.
- Runtime: `18.68` minutes.
- GPU inference required: yes, one fixed-8 run.

## Main causal result

- Ordinary associations: 4089
- Endpoint-matched causal errors: 193
- Scorer-ranking failures: 72
  (37.31%)
- Candidate-threshold failures: 31
  (16.06%)
- ILP/global failures: 21
  (10.88%)
- Postprocessing removals: 57
  (29.53%)
- Detection misses: 37
- Candidate recall with detected endpoints: 94.89%
- Final ordinary-edge recall with detected endpoints: 95.24%

## Dense-scene and family checks

| group_type | group | ordinary_associations | errors | error_rate | causal_errors | scorer_ranking | candidate_threshold | ilp_global | postprocessing_removed | candidate_recall | target_top1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_family | 44b6 | 492 | 41 | 0.0833333 | 37 | 9 | 5 | 4 | 19 | 0.940574 | 0.963115 |
| dataset_family | 6bba | 3597 | 189 | 0.0525438 | 156 | 63 | 26 | 17 | 38 | 0.950056 | 0.965769 |
| density_bin | medium_low | 1061 | 50 | 0.0471254 | 34 | 12 | 6 | 6 | 7 | 0.957895 | 0.972249 |
| density_bin | medium_high | 906 | 69 | 0.0761589 | 56 | 18 | 13 | 7 | 15 | 0.927212 | 0.955207 |
| density_bin | low | 1219 | 24 | 0.0196883 | 19 | 9 | 1 | 1 | 5 | 0.978583 | 0.984349 |
| density_bin | high | 903 | 87 | 0.0963455 | 84 | 33 | 11 | 7 | 30 | 0.92 | 0.942222 |

## Per dataset

| group | ordinary_associations | errors | error_rate | scorer_ranking | candidate_threshold | ilp_global | postprocessing_removed | detected_node_recall | final_node_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44b6_0113de3b | 50 | 3 | 0.06 | 0 | 0 | 0 | 3 | 1 | 0.942308 |
| 44b6_0b24845f | 49 | 2 | 0.0408163 | 0 | 0 | 0 | 2 | 1 | 1 |
| 44b6_341df25f | 207 | 1 | 0.00483092 | 1 | 0 | 0 | 0 | 1 | 1 |
| 44b6_e57ff5c6 | 186 | 35 | 0.188172 | 8 | 5 | 4 | 14 | 0.989637 | 0.948187 |
| 6bba_05b6850b | 845 | 11 | 0.0130178 | 4 | 0 | 0 | 3 | 0.998839 | 0.998839 |
| 6bba_05db0fb1 | 1177 | 83 | 0.0705183 | 36 | 9 | 3 | 22 | 0.995932 | 0.992677 |
| 6bba_969618f6 | 652 | 5 | 0.00766871 | 2 | 1 | 1 | 0 | 1 | 0.995582 |
| 6bba_fc83837d | 923 | 90 | 0.0975081 | 21 | 16 | 13 | 13 | 0.982236 | 0.959248 |

## Engineering decision

This experiment changes no production setting. The result determines whether the next
single action should target learned edge ranking or a different stage.

**Next action:** Repeat the ranking diagnostic on an independent holdout before changing the production recipe.

See the machine-readable `decision.json` for the predeclared classification inputs.
