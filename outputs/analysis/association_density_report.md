# Association density diagnostic

## Scope and provenance

- Artifact: public two-seed alpha=0.5 det=0.96875 selected-edge GEFF; four-dataset overlap with competition GT; pre-postprocessing graph
- Evaluated set: 4 datasets (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1)
- Missing from the requested fixed-eight prediction artifact: 44b6_341df25f, 44b6_e57ff5c6, 6bba_969618f6, 6bba_fc83837d
- Physical scale (z, y, x): (1.625, 0.40625, 0.40625) micrometers per voxel
- Node matching gate: 7.0 micrometers
- Local density radius: 15.0 micrometers around the GT target
- Density population: selected predicted nodes in the target's next frame
- Ordinary association: GT sources with exactly one outgoing edge; GT divisions are excluded
- Density quartiles: q25=3.000, q50=9.000, q75=13.000

## Predeclared decision rule

`A VERIFIED` requires all of: high-density error rate at least 1.5x low density;
small-margin error rate at least 1.5x large margin; high-density/small-margin cases
contain at least 40% of association errors; and association errors outnumber errors
with a missing GT source/target detection. `B PARTIALLY VERIFIED` requires the density
effect and association-error dominance when complete candidate margins are unavailable
or another full-verification condition fails. Otherwise the result is `C REJECTED`.

## Result: B PARTIALLY VERIFIED

- High- versus low-density error ratio: 2.3075
- High-density error rate: 0.0502
- Low-density error rate: 0.0218
- Small- versus large-margin error ratio: unavailable
- Dense + small-margin error share: unavailable
- Association errors (wrong/missing edge): 72
- Detection-attributed errors (source/target node unmatched): 17
- Overall node recall: 0.9950

The saved public graph is the learned/ILP association graph before the repository's
final motion relinking, gap repair, safe-division addition, short-track filtering,
and line-fit smoothing. Therefore these numbers diagnose the learned ordinary
association stage, not an exact rescore of the final generalization-safe submission.
No detector inference, training, submission generation, or GPU job was run.

## Density and dataset-family comparisons

| group_type | group | associations | errors | error_rate | edge_tp | edge_fp | edge_fn | edge_precision | edge_recall | edge_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_family | 44b6 | 99 | 2 | 0.020202 | 97 | 0 | 2 | 1 | 0.979798 | 0.979798 |
| dataset_family | 6bba | 2022 | 87 | 0.0430267 | 1935 | 48 | 87 | 0.975794 | 0.956973 | 0.934783 |
| density_bin | medium_low | 381 | 19 | 0.0498688 | 362 | 12 | 19 | 0.967914 | 0.950131 | 0.92112 |
| density_bin | medium_high | 521 | 31 | 0.059501 | 490 | 16 | 31 | 0.968379 | 0.940499 | 0.912477 |
| density_bin | low | 781 | 17 | 0.021767 | 764 | 7 | 17 | 0.990921 | 0.978233 | 0.969543 |
| density_bin | high | 438 | 22 | 0.0502283 | 416 | 13 | 22 | 0.969697 | 0.949772 | 0.922395 |

## Margin comparison

| group | associations | errors | error_rate | margin_available | median_candidate_margin |
| --- | --- | --- | --- | --- | --- |
| unavailable | 2121 | 89 | 0.0419613 | 0 |  |

## Candidate-margin limitation

Candidate margins available: **false**. A true
top-one/top-two margin requires rejected candidate edges and their `edge_prob` values.
The saved GEFFs used here retain only optimizer-selected edges, so `candidate_count`,
`best_candidate_score`, `second_candidate_score`, and `candidate_margin` are null.
`selected_edge_score` remains available but is not presented as a margin proxy.

Minimal future export: before ILP solution filtering, persist `(source_id, target_id,
edge_prob, solution)` for every next-frame candidate edge, keyed to the final GEFF node
IDs. Re-running this CPU-only script will then populate all margin bins.

This is the highest-value next experiment: export candidate alternatives during one
already-planned inference/CV run, then rerun this diagnostic on the exact eight final
prediction graphs. Do not build a new tracker until the small-margin interaction is
measured and the final postprocessed fixed-eight graphs are available.

## By dataset

| dataset | associations | errors | error_rate | edge_tp | edge_fp | edge_fn | node_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 44b6_0113de3b | 50 | 0 | 0 | 50 | 0 | 0 | 1 |
| 44b6_0b24845f | 49 | 2 | 0.0408163 | 47 | 0 | 2 | 1 |
| 6bba_05b6850b | 845 | 14 | 0.016568 | 831 | 7 | 14 | 0.997677 |
| 6bba_05db0fb1 | 1177 | 73 | 0.0620221 | 1104 | 41 | 73 | 0.992677 |

## Top 20 highest-confidence wrong ordinary associations

| dataset | t | gt_source_id | gt_target_id | outcome | predicted_target_gt_id | selected_edge_score | candidate_margin | selected_edge_distance_um | local_density_pred | density_bin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6bba_05db0fb1 | 66 | 67001312 | 68001331 | wrong_association |  | 0.962153 |  | 2.81458 | 9 | medium_low |
| 6bba_05db0fb1 | 20 | 21000311 | 22000331 | wrong_association |  | 0.961717 |  | 2.81458 | 10 | medium_high |
| 6bba_05db0fb1 | 82 | 83001622 | 84001644 | wrong_association |  | 0.951666 |  | 0 | 11 | medium_high |
| 6bba_05db0fb1 | 21 | 22000331 | 23000351 | wrong_association |  | 0.945038 |  | 0 | 13 | medium_high |
| 6bba_05db0fb1 | 22 | 23000342 | 24000358 | wrong_association |  | 0.937916 |  | 2.2981 | 10 | medium_high |
| 6bba_05db0fb1 | 46 | 47000873 | 48000894 | wrong_association |  | 0.935706 |  | 1.625 | 14 | high |
| 6bba_05db0fb1 | 61 | 62001198 | 63001221 | wrong_association |  | 0.931318 |  | 2.2981 | 11 | medium_high |
| 6bba_05db0fb1 | 80 | 81001589 | 82001604 | wrong_association |  | 0.921825 |  | 5.1387 | 7 | medium_low |
| 6bba_05b6850b | 71 | 72000669 | 73000682 | wrong_association |  | 0.901868 |  | 2.2981 | 2 | low |
| 6bba_05db0fb1 | 22 | 23000356 | 24000373 | wrong_association |  | 0.900334 |  | 1.625 | 11 | medium_high |
| 6bba_05db0fb1 | 32 | 33000538 | 34000560 | wrong_association |  | 0.896096 |  | 4.875 | 13 | medium_high |
| 6bba_05db0fb1 | 73 | 74001434 | 75001454 | wrong_association |  | 0.886797 |  | 1.625 | 15 | high |
| 6bba_05db0fb1 | 80 | 81001577 | 82001599 | wrong_association |  | 0.886242 |  | 6.08019 | 7 | medium_low |
| 6bba_05db0fb1 | 20 | 21000318 | 22000338 | wrong_association |  | 0.867852 |  | 2.2981 | 13 | medium_high |
| 6bba_05db0fb1 | 58 | 59001141 | 60001162 | wrong_association |  | 0.851737 |  | 3.63361 | 7 | medium_low |
| 6bba_05db0fb1 | 54 | 55001051 | 56001073 | wrong_association |  | 0.836992 |  | 1.625 | 8 | medium_low |
| 6bba_05b6850b | 64 | 65000595 | 66000606 | wrong_association |  | 0.836855 |  | 3.25 | 3 | low |
| 6bba_05db0fb1 | 56 | 57001093 | 58001119 | wrong_association |  | 0.821243 |  | 0 | 9 | medium_low |
| 6bba_05db0fb1 | 37 | 38000665 | 39000687 | wrong_association |  | 0.818205 |  | 1.625 | 16 | high |
| 6bba_05db0fb1 | 50 | 51000957 | 52000981 | wrong_association |  | 0.806759 |  | 2.2981 | 11 | medium_high |

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_association_density_diagnostic.py `
  --predictions-dir "data\cache\public_two_seed_raw\tracking_repo\predictions\unknown\unet_transformer\split_0" `
  --ground-truth-dir "data\cache\competition_fixed8_gt\structured" `
  --output-dir "outputs\analysis" `
  --datasets 44b6_0113de3b 44b6_0b24845f 6bba_05b6850b 6bba_05db0fb1 `
  --artifact-label "public two-seed alpha=0.5 det=0.96875 selected-edge GEFF; four-dataset overlap with competition GT; pre-postprocessing graph"
```

Machine-readable conclusion:

```json
{
  "association_dominates": true,
  "association_errors": 72,
  "burden_signal": false,
  "conclusion": "B PARTIALLY VERIFIED",
  "density_error_ratio_high_vs_low": 2.3075476766048886,
  "density_signal": true,
  "detection_errors": 17,
  "high_density_error_rate": 0.0502283105022831,
  "high_density_small_margin_error_share": null,
  "large_margin_error_rate": null,
  "low_density_error_rate": 0.02176696542893726,
  "margin_available": false,
  "margin_error_ratio_small_vs_large": null,
  "margin_signal": false,
  "small_margin_error_rate": null
}
```
