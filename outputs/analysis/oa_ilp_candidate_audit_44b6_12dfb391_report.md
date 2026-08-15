# OA ILP / candidate-set audit — `44b6_12dfb391`

**Experiment:** `oa_ilp_candidate_audit_v1`  
**Date:** 2026-08-15  
**Frozen recipe:** motion-relink OFF, `edge_threshold=0.40`  
**Detections / scorer weights:** unchanged (CPU audit of the promoted-recipe capture)  
**Question:** After ranking retrains failed to transfer, are remaining ordinary-association failures a candidate-set hole (GT never admitted to ILP) or an ILP occupancy conflict?

## Decision: **close the ILP / candidate-set branch**

Do not change the production recipe. Do not start another edge-scorer retrain. Do not admit every source-top-1 below 0.40. Do not lower the global edge gate. Do not reopen short-track OFF or motion-relink ON.

The candidate-set hole is real, but the obvious repair is a false-positive bomb under the current ILP disappearance cost. The ILP occupancy errors are proximal extra detections stealing the parent — the same geometry already rejected as a ranking term.

## Dataset burden (frozen recipe capture)

| | |
| --- | ---: |
| Ordinary associations | 771 |
| Errors | 27 |
| Causal (both endpoints detected) | 24 |
| Edge TP/FP/FN (this dataset) | 745 / 41 / 28 |
| `adj_edge_jaccard` | 0.934707 |

Capture: `outputs/experiments/candidate_capture_recipe_c_edge_0_40_v1/holdout8/`  
Raw GEFF: `outputs/experiments/oa_ilp_candidate_audit_v1/raw_geff/44b6_12dfb391.geff`

## Causal buckets

Native bottleneck labels split by whether GT ever enters the ILP graph (`softmax > 0.40`):

| Bucket | Count | Share of causal | Meaning |
| --- | ---: | ---: | --- |
| ranking_below_gate | 14 | 58.3% | Rank > 1 **and** softmax ≤ 0.40 — ILP never sees GT |
| candidate_threshold | 1 | 4.2% | Rank-1 but softmax 0.288 ≤ 0.40 |
| ranking_gated | 1 | 4.2% | Rank-2, softmax 0.462 — only ranking error ILP could have flipped |
| ilp_global | 4 | 16.7% | Rank-1 **and** gated; source taken by another child |
| postprocessing_rematch | 4 | 16.7% | ILP kept the pred pair; final identity mapping missed GT |
| detection_miss | 3 | (excluded) | Frozen; not an ILP/candidate lever |
| candidate_absent | 0 | 0% | Every endpoint-matched GT pair is in the top-16 ∪ gated capture |

**62.5% of causal errors never enter ILP.** That is why in-sample ranking lifts from hard-neg retrain did not transfer: most remaining misses on this dataset are not alternative rows inside the gated graph.

Only **1 / 15** native `scorer_ranking` rows is above the gate.

## Source-top-1 admit — REJECTED offline

Admitting each source's top-1 continuation even when softmax ≤ 0.40 would put those GTs into ILP. Census of this capture:

| | |
| --- | ---: |
| Source-top-1 pairs below 0.40 | **5670** |
| Of those, OA error GTs | 7 |
| Extra candidates per recoverable GT | **810** |
| Median softmax of excluded top-1 | 0.124 |

ILP disappearance weight is **1.575** vs edge cost `-p`. Any admitted pair with `p > 0` is cheaper than disappearing, so ILP would take nearly all 5670 extras. That is the same FP mechanism as the rejected 0.35/0.30/0.25 gate sweep, just restricted to one pair per source.

A floor (e.g. admit only if `p ≥ 0.25`) still adds **1448** extras to recover **4** GTs. Do not run this on GPU.

## ILP occupancy — extra detection theft, not a broken constraint

All 4 `ilp_global` rows are `source_taken`. GT is `target_rank=1` and gated (`p` 0.53–0.77) but `source_rank=2`. The source's preferred child is **not a GT association** (4/4 stolen children unmatched to any ordinary GT pair).

| t | GT `p` / dist µm | Stolen `p` / dist µm | Stolen↔GT µm |
| ---: | --- | --- | ---: |
| 14 | 0.617 / 4.88 | 0.845 / 1.63 | 6.08 |
| 37 | 0.766 / 2.30 | 0.801 / 2.30 | 4.60 |
| 45 | 0.534 / 5.63 | 0.944 / 1.63 | 4.88 |
| 58 | 0.623 / 5.39 | 0.931 / **0.00** | 5.39 |

Stolen and GT targets are nearest neighbors of each other (~5–6 µm). Recovering GT means either a false division (t=14 has **three** `target_rank=1` gated children) or suppressing the extra peak. Preferring the closer child is what ILP already does; a distance bonus was **REJECTED** on this same dataset.

Rematch: 2/4 rows share a node with an ILP/ranking conflict (cascade from t=14 `7281` and t=45 `22111`).

## Recoverability (all 27 errors)

| Lever | Count | Status |
| --- | ---: | --- |
| admit_source_top1 | 7 | **REJECTED offline** (5670 extras) |
| needs_ranking_or_lower_gate | 8 | Closed (retrain + gate sweep) |
| ilp_contention_repair | 4 | Same proximal-distractor geometry; stolen child is extra detection |
| ilp_override_rank2 | 1 | Too small; ILP already saw both |
| needs_identity_repair | 4 | 2 are ILP cascades |
| needs_detection | 3 | Frozen |

## Production

Unchanged: `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`  
Scores to beat remain fixed-8 **0.918144** / holdout-8 **0.964673**.

## Artifacts

- Login: `outputs/experiments/oa_ilp_candidate_audit_v1/` (`error_table.csv`, `summary.json`, this report)
- Raw GEFF mirror: `outputs/experiments/oa_ilp_candidate_audit_v1/raw_geff/44b6_12dfb391.geff`
- Code: `src/biohub_pipeline/oa_ilp_candidate_audit.py`, `scripts/run_oa_ilp_candidate_audit.py`
- Tests: `tests/test_oa_ilp_candidate_audit.py` (4 passed)

## Next (not started)

Ranking, linear pairwise reweight, edge-scorer retrain, global gate, and ILP/candidate-set expansion are closed on this holdout killer. Remaining holdout ordinary-error mass that is **not** this dataset's ranking-below-gate pattern is detection miss on `6bba_07e24132` (22/34 OA errors there; 40% of holdout all-error mass in the fresh capture). That is a detector-scope leap, not another edge-scorer or ILP job.
