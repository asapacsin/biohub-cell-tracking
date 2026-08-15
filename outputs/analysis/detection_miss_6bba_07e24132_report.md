# Detection-miss audit — `6bba_07e24132` + frozen z-shift sweep

**Experiments:** `detection_miss_audit_6bba_07e24132`, `detection_z_shift_v1`  
**Date:** 2026-08-15  
**Frozen recipe:** motion-relink OFF, `edge_threshold=0.40`, `detection_threshold=0.96875`  
**Compute:** CPU only (capture nodes + postprocessed submissions). No GPU. No detector retrain.

## Decision: **REJECT** global z-shift; **do not retrain the detector** for these misses

The 22 ordinary-association `detection_miss` rows on holdout `6bba_07e24132` are **17 unmatched GT nodes**. They are not empty frames. A free predicted peak sits 7–10 µm away, almost entirely as a **negative z offset** (bright plane vs GT center). A global +z voxel shift recovers this dataset’s node recall and **loses on both fixed-8 and holdout-8**. Raw-intensity z-centroid does not move those peaks toward GT.

Production stays `configs/experiments/recipe_c_motion_off_edge_0_40_det0_96875.yaml`.

## Anatomy of the 22 OA detection misses

| | |
| --- | ---: |
| OA errors on this dataset | 34 |
| `detection_miss` | **22** (15 source / 16 target / 9 both) |
| Unique unmatched GT nodes | **17** (all of them are OA endpoints) |
| Detection-time node recall | 0.9526 (342/359) |
| Final node recall | 0.9359 |

Nearest predicted peak to each unmatched GT:

| Band | Count | Interpretation |
| --- | ---: | --- |
| ≤ 7 µm, already matched to another GT | 3 | matching contention |
| 7–10 µm, **free** (not matched to any GT) | 11 | localization just outside the 7 µm gate |
| 10–13 µm, free | 3 | same pattern, slightly farther |

**No unmatched GT lacks a nearby peak.** Median nearest-peak distance 8.64 µm. Offsets of the free peaks are systematically `dz ≈ −6.5 to −9.8 µm` (4–6 z voxels at 1.625 µm) with small y/x. The same pattern appears on fixed-8 `6bba_fc83837d` (also 17 unmatched GT, mostly `dz ≈ −5 to −8 µm`).

This is GT-center vs fluorescence-maximum, not a missing detection.

`det=0.960` already improved this dataset slightly (0.831 vs 0.827) and was **REJECTED** on holdout-8 overall. Do not reopen that sweep.

## Global z-shift on frozen submissions — REJECT

Same nodes and edges; only predicted node `z` += Δ voxels; official 7 µm matching.

| Δz voxels | fixed-8 Δ | holdout-8 Δ | `6bba_07e24132` adj / recall |
| ---: | ---: | ---: | --- |
| −2 | −0.1119 | −0.0601 | 0.769 / 0.914 |
| −1 | −0.0230 | −0.0058 | 0.803 / 0.928 |
| **0** | **0** | **0** | **0.827 / 0.936** |
| +1 | −0.0140 | −0.0079 | 0.836 / **0.953** |
| +2 | −0.0572 | −0.0301 | 0.815 / 0.958 |
| +3 | −0.1847 | −0.1173 | 0.703 / 0.955 |
| +4 | −0.3541 | −0.2978 | 0.491 / 0.919 |

+1 voxel raises this dataset’s node recall to the detection-time 0.953 and adj +0.009, but FP rises on both splits (fixed 187→229, holdout 76→102). **No nonzero shift is nonnegative on both splits.**

## Intensity z-COM probe (negative)

On `6bba_07e24132.zarr` (TZYX `(100,64,256,256)`, spacing 1.625/0.40625/0.40625 from `zarr.json`): intensity-weighted z in a ±6 voxel column at each unmatched nearest peak **does not walk toward GT** (4/17 slightly better; none newly under 7 µm except the three already-inside contention cases). Matched sample: 19/30 get *worse*. The detector is already sitting on the bright plane.

## Closed levers

- Edge-scorer retrain / pairwise / distance rank / source-top-1 admit / global edge gate / short-track OFF / motion-relink ON
- `detection_threshold=0.960`
- Global detection z offset
- Intensity z-centroid on raw fluorescence
- Detector retrain for these 17 nodes (wrong geometry: appearance peak ≠ GT center; large compute, both-split risk)

## Artifacts

- `outputs/experiments/detection_z_shift_v1/` (`sweep.csv`, `decision.json`, per-shift per-dataset CSVs)
- This report: `outputs/analysis/detection_miss_6bba_07e24132_report.md`

## Next

Frozen recipe remains the best defensible local result (fixed-8 **0.918144**, holdout-8 **0.964673**). Remaining errors are mixed localization/ranking on a few hard sequences. There is no pending cheap association or detection-coordinate experiment with a both-split promote path. Stop unless a new independent hypothesis appears.
