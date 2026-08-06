# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.006707, "end_time": "2026-07-23T00:37:14.940821+00:00", "exception": false, "start_time": "2026-07-23T00:37:14.934114+00:00", "status": "completed"}
# # 🧬 Biohub 132 | Clean Short-Track Rescue + Light Local CV | No Hack
#
# ## Clean baseline 0.908 → recover only high-confidence five-node tracks
#
# Kaggle has addressed the historical metric exploit, so **0.908 is treated as the legitimate clean baseline**. Biohub 132 targets a real model weakness: valid short trajectories may be removed by the global minimum-track-length filter.
#
# This notebook keeps the attached baseline pipeline intact and enables a deliberately conservative adaptive rescue. Only existing components of length five are eligible, and they must have high learned-edge confidence and short mean physical displacement. No nodes or edges are fabricated.
#
# **No hack:** no artificial hubs, fake forks, negative-time nodes, out-of-volume coordinates, cross-clip edges, or hidden-test labels.
#

# %% [markdown] papermill={"duration": 0.00534, "end_time": "2026-07-23T00:37:14.951849+00:00", "exception": false, "start_time": "2026-07-23T00:37:14.946509+00:00", "status": "completed"}
# > **Attach exactly these inputs before running:**
# > 1. `Biohub - Cell Tracking During Development`
# > 2. `pilkwang/biohub-tracking-support-pack-50ep-v1`
# >
# > The clean hidden-test `submission.csv` is written and validated first. The fixed-8 CV then runs under a guard and cannot overwrite the submission file.
#

# %% _kg_hide-input=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 0.015944, "end_time": "2026-07-23T00:37:14.973431+00:00", "exception": false, "start_time": "2026-07-23T00:37:14.957487+00:00", "status": "completed"} tags=["hide-input"]
from pathlib import Path
from IPython.display import Image, display

cover_image_path = Path("/kaggle/input/datasets/pilkwang/pilkwang-public-dataset-for-notebooks-figures/biohub_lineage.png")
if cover_image_path.exists():
    display(Image(filename=str(cover_image_path)))


# %% [markdown] papermill={"duration": 0.005752, "end_time": "2026-07-23T00:37:14.984904+00:00", "exception": false, "start_time": "2026-07-23T00:37:14.979152+00:00", "status": "completed"}
# ## Controlled experiment
#
# This is a clean post-processing A/B test from the legitimate `0.908` baseline.
#
# | Setting | 0.908 baseline | Biohub 132 |
# |---|---:|---:|
# | Motion relaxed gate | `9.5 µm` | `9.5 µm` |
# | Minimum retained track length | `6` | `6` |
# | Adaptive rescue | disabled | **enabled** |
# | Eligible rescued length | none | **exactly 5 nodes** |
# | Minimum mean learned-edge probability | n/a | **`0.90`** |
# | Maximum mean edge distance | n/a | **`2.75 µm`** |
# | Rescue budget | n/a | **min(`0.6%` of nodes, `60`)** |
#
# **Hypothesis:** fixed-8 CV retains meaningful node-recall and edge-FN headroom. A small set of highly coherent five-node components may be genuine trajectories incorrectly removed by the length-six cutoff.
#

# %% [markdown] papermill={"duration": 0.008007, "end_time": "2026-07-23T00:37:14.998123+00:00", "exception": false, "start_time": "2026-07-23T00:37:14.990116+00:00", "status": "completed"}
# ## Pipeline overview
#
# 1. **3D detector** proposes candidate cell centers in each frame.
# 2. **Learned edge scorer** estimates which cells in adjacent frames should be linked.
# 3. **ILP graph builder** chooses a globally consistent set of detections and links.
# 4. **Post-processing** repairs short gaps, removes weak isolated fragments, smooths trajectories, and adds only geometrically safe divisions.
# 5. **Schema validation and diagnostics** check that the final `submission.csv` has valid rows, IDs, and graph statistics.
#
# The main lesson behind this notebook is that cell-tracking competitions are graph problems. A small change to graph construction can matter as much as a detector threshold change.
#

# %% [markdown] papermill={"duration": 0.005193, "end_time": "2026-07-23T00:37:15.008526+00:00", "exception": false, "start_time": "2026-07-23T00:37:15.003333+00:00", "status": "completed"}
# ## Selected settings
#
# | Setting | Value |
# |---|---:|
# | Clean leaderboard baseline | `0.908` |
# | Detection threshold | `0.96875` |
# | ILP appearance/disappearance cost | `0.0 / 1.575` |
# | Motion tight/relaxed gate | `6.0 / 9.5 µm` |
# | Minimum track length | `6` |
# | Adaptive short-track rescue | **enabled** |
# | Rescue eligible length | **`5` only** |
# | Minimum mean edge probability | **`0.90`** |
# | Maximum mean edge distance | **`2.75 µm`** |
# | Maximum rescued nodes | **`60` or `0.6%`** |
# | Safe-division parent/sister radii | `4.66 / 8.5 µm` |
#

# %% _kg_hide-input=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 0.016846, "end_time": "2026-07-23T00:37:15.030518+00:00", "exception": false, "start_time": "2026-07-23T00:37:15.013672+00:00", "status": "completed"} tags=["hide-input"]
import os
BIOHUB_PRESET = 'biohub_132_clean_short_track_rescue_lightcv_nohack'
BIOHUB_SCORE_AXIS = 'Clean recall test: conservatively rescue high-confidence five-node track components from the legitimate 0.908 baseline'

# Graph calibration preset.
os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.96875"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.0"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.0"
os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "9.5"

# Main experiment: tune the ILP birth/death tradeoff.
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.575"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.8"
os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
os.environ["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
os.environ["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
os.environ["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
os.environ["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = '8.5'
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

# Keep this disabled so the experiment stays focused on ILP cost calibration.
os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC"] = "0.10"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "5"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.90"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "2.75"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.006"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "60"

# Auxiliary branch disabled: the public notebook should run from the attached support artifact only.
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = '0'
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = '0'
os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = '0'
os.environ["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.0"
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = ''
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = '0'
os.environ["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.20"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = '0'
os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"

print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)

# %% _kg_hide-input=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 0.837012, "end_time": "2026-07-23T00:37:15.872758+00:00", "exception": false, "start_time": "2026-07-23T00:37:15.035746+00:00", "status": "completed"} tags=["hide-input"]
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import time
from pathlib import Path

import pandas as pd

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TEST_DIR = COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "biohub_132_clean_short_track_rescue_lightcv_nohack"
TARGET_ARTIFACT_SLUG = os.environ.get("BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1")
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get(
    "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
))
ALLOW_ARTIFACT_FALLBACK = os.environ.get("BIOHUB_ALLOW_ARTIFACT_FALLBACK", "0") != "0"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.99"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))

# Empty for a real submission. Useful for local smoke tests, e.g. BIOHUB_SLICE=:1.
SLICE = os.environ.get("BIOHUB_SLICE", "").strip()

# If dependencies are not already installed and no offline wheels are attached,
# this controls whether the notebook attempts PyPI installation.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"
RUN_OUTPUT_DIAGNOSTICS = os.environ.get("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "1") != "0"
RUN_VISUAL_EDA = os.environ.get("BIOHUB_RUN_VISUAL_EDA", "1") != "0"

# Output-level graph post-processing.
OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0"
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0"
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0"
OUTPUT_PRUNE_ISOLATED = os.environ.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0"
OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"
MOTION_RELINK_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0"))
MOTION_RELINK_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0"))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600"))

OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_DENSITY_ADAPTIVE = os.environ.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0"
GAP_DENSITY_REFERENCE_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5"))
GAP_DENSITY_GAIN = float(os.environ.get("BIOHUB_GAP_DENSITY_GAIN", "0.040"))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125"))
GAP_DENSITY_NEIGHBORS = int(os.environ.get("BIOHUB_GAP_DENSITY_NEIGHBORS", "3"))
GAP_CLOSE_REUSE_EXISTING = os.environ.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0"
GAP_CLOSE_REUSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2"))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05"))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000"))
GAP_REFINE_SYNTHETIC = os.environ.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0"
GAP_REFINE_WIN_Z = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_Z", "1"))
GAP_REFINE_WIN_YX = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_YX", "3"))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2"))

OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6"))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0"
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10"))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4"))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82"))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25"))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018"))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180"))

OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))

OUTPUT_GAP2_RECOVERY = os.environ.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0"
GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2"))
GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4"))
GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045"))
GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180"))
GAP2_REQUIRE_CONTEXT = os.environ.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0"
GAP2_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006"))

OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))

# DeepCenter support is retained for compatibility, but this selected run keeps it disabled.
USE_DEEPCENTER_VETO = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "1") != "0"
REQUIRE_DEEPCENTER_VETO = os.environ.get("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1") != "0"
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json",
)
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT",
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt",
)
DEEPCENTER_RELATIVE = os.environ.get("BIOHUB_DEEPCENTER_RELATIVE", "weights/full_frame_center/checkpoint_last.pt")
DEEPCENTER_GAP_VETO = os.environ.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0"
DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10"))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0"))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0"))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12"))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1"))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2"))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8"))

CONFIG_DISPLAY = {
    "experiment_tag": EXPERIMENT_TAG,
    "method": METHOD,
    "weights": WEIGHTS_RELATIVE,
    "target_artifact_slug": TARGET_ARTIFACT_SLUG,
    "primary_artifact_manifest": str(PRIMARY_ARTIFACT_MANIFEST),
    "allow_artifact_fallback": ALLOW_ARTIFACT_FALLBACK,
    "det_threshold": DET_THRESHOLD,
    "unet_batch_size": UNET_BATCH_SIZE,
    "use_ilp": USE_ILP,
    "ilp_edge_weight": ILP_EDGE_WEIGHT,
    "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
    "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
    "ilp_division_weight": ILP_DIVISION_WEIGHT,
    "slice": SLICE,
    "allow_pip_install": ALLOW_PIP_INSTALL,
    "run_visual_eda": RUN_VISUAL_EDA,
    "output_edge_max_um": OUTPUT_EDGE_MAX_UM,
    "output_enforce_next_frame": OUTPUT_ENFORCE_NEXT_FRAME,
    "output_single_parent_repair": OUTPUT_SINGLE_PARENT_REPAIR,
    "output_single_child_repair": OUTPUT_SINGLE_CHILD_REPAIR,
    "output_prune_isolated": OUTPUT_PRUNE_ISOLATED,
    "output_motion_relink": OUTPUT_MOTION_RELINK,
    "motion_relink_tight_um": MOTION_RELINK_TIGHT_UM,
    "motion_relink_relaxed_um": MOTION_RELINK_RELAXED_UM,
    "motion_relink_velocity_weight": MOTION_RELINK_VELOCITY_WEIGHT,
    "motion_relink_learned_bonus": MOTION_RELINK_LEARNED_BONUS,
    "motion_relink_max_frame_nodes": MOTION_RELINK_MAX_FRAME_NODES,
    "output_division_geometry_filter": OUTPUT_DIVISION_GEOMETRY_FILTER,
    "div_parent_max_um": DIV_PARENT_MAX_UM,
    "div_sister_max_um": DIV_SISTER_MAX_UM,
    "div_drop_to_single_if_bad": DIV_DROP_TO_SINGLE_IF_BAD,
    "output_gap_close": OUTPUT_GAP_CLOSE,
    "gap_close_max_gap": GAP_CLOSE_MAX_GAP,
    "gap_close_effective_max_gap": min(GAP_CLOSE_MAX_GAP, 1),
    "gap_close_um": GAP_CLOSE_UM,
    "gap_density_adaptive": GAP_DENSITY_ADAPTIVE,
    "gap_density_reference_um": GAP_DENSITY_REFERENCE_UM,
    "gap_density_gain": GAP_DENSITY_GAIN,
    "gap_density_max_step_delta_um": GAP_DENSITY_MAX_STEP_DELTA_UM,
    "gap_density_neighbors": GAP_DENSITY_NEIGHBORS,
    "gap_close_reuse_existing": GAP_CLOSE_REUSE_EXISTING,
    "gap_close_reuse_um": GAP_CLOSE_REUSE_UM,
    "gap_close_max_added_frac": GAP_CLOSE_MAX_ADDED_FRAC,
    "gap_close_max_added_abs": GAP_CLOSE_MAX_ADDED_ABS,
    "gap_refine_synthetic": GAP_REFINE_SYNTHETIC,
    "gap_refine_win_z": GAP_REFINE_WIN_Z,
    "gap_refine_win_yx": GAP_REFINE_WIN_YX,
    "gap_refine_max_shift_um": GAP_REFINE_MAX_SHIFT_UM,
    "output_filter_short_tracks": OUTPUT_FILTER_SHORT_TRACKS,
    "output_min_track_len": OUTPUT_MIN_TRACK_LEN,
    "output_keep_division_components": OUTPUT_KEEP_DIVISION_COMPONENTS,
    "adaptive_short_track_rescue": ADAPTIVE_SHORT_TRACK_RESCUE,
    "short_track_rescue_trigger_removed_frac": SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC,
    "short_track_rescue_min_len": SHORT_TRACK_RESCUE_MIN_LEN,
    "short_track_rescue_min_mean_edge_prob": SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB,
    "short_track_rescue_max_mean_edge_dist_um": SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM,
    "short_track_rescue_max_nodes_frac": SHORT_TRACK_RESCUE_MAX_NODES_FRAC,
    "short_track_rescue_max_nodes_abs": SHORT_TRACK_RESCUE_MAX_NODES_ABS,
    "output_linefit_smooth": OUTPUT_LINEFIT_SMOOTH,
    "output_linefit_weight": OUTPUT_LINEFIT_WEIGHT,
    "output_linefit_window": OUTPUT_LINEFIT_WINDOW,
    "output_gap2_recovery": OUTPUT_GAP2_RECOVERY,
    "gap2_max_total_um": GAP2_MAX_TOTAL_UM,
    "gap2_max_step_um": GAP2_MAX_STEP_UM,
    "gap2_max_links_frac": GAP2_MAX_LINKS_FRAC,
    "gap2_max_links_abs": GAP2_MAX_LINKS_ABS,
    "gap2_require_context": GAP2_REQUIRE_CONTEXT,
    "gap2_frame_frac_cap": GAP2_FRAME_FRAC_CAP,
    "output_safe_divisions": OUTPUT_SAFE_DIVISIONS,
    "safe_div_max_um": SAFE_DIV_MAX_UM,
    "safe_div_sister_max_um": SAFE_DIV_SISTER_MAX_UM,
    "safe_div_existing_child_max_um": SAFE_DIV_EXISTING_CHILD_MAX_UM,
    "safe_div_frame_frac_cap": SAFE_DIV_FRAME_FRAC_CAP,
    "safe_div_global_frac_cap": SAFE_DIV_GLOBAL_FRAC_CAP,
    "use_deepcenter_add_only_gate": USE_DEEPCENTER_VETO,
    "deepcenter_gap_add_gate": DEEPCENTER_GAP_VETO,
    "deepcenter_safe_div_add_gate": DEEPCENTER_SAFE_DIV_VETO,
    "deepcenter_gap_threshold": DEEPCENTER_GAP_THRESHOLD,
    "deepcenter_expected_epoch": DEEPCENTER_EXPECTED_EPOCH,
    "deepcenter_gap_confirm_min_span_um": DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM,
    "deepcenter_safe_div_threshold": DEEPCENTER_SAFE_DIV_THRESHOLD,
    "deepcenter_checkpoint_default": DEEPCENTER_CHECKPOINT_DEFAULT,
}

print("Biohub learned UNet + node-transformer + ILP submission")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent=2, sort_keys=True))

# %% [markdown] papermill={"duration": 0.005302, "end_time": "2026-07-23T00:37:15.883747+00:00", "exception": false, "start_time": "2026-07-23T00:37:15.878445+00:00", "status": "completed"}
# ## Artifact and Dependency Setup
#
# The notebook expects a compact support artifact containing the inference source,
# trained weights, and optionally offline dependency wheels. The primary Kaggle
# attachment path is:
#
# ```text
# /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json
# ```
#
# The setup cell first uses already-installed modules, then attached wheels, and
# only attempts an internet install when explicitly enabled for local development.
# By default, the artifact resolver accepts only the 50-epoch package named by
# `TARGET_ARTIFACT_SLUG`; set `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1` only for local
# debugging against an older package.
#

# %% _kg_hide-input=true _kg_hide-output=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 40.091722, "end_time": "2026-07-23T00:37:55.980744+00:00", "exception": false, "start_time": "2026-07-23T00:37:15.889022+00:00", "status": "completed"} tags=["hide-input"]
import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]

# The safe path for offline reruns is to use attached wheels.
# Set BIOHUB_ALLOW_PIP_INSTALL=1 only for an interactive internet-enabled run.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def find_artifacts_root() -> Path:
    # Strict path by request: attach this dataset through Kaggle Add Input.
    artifacts = Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")
    if not artifacts.exists():
        raise FileNotFoundError(
            "Required input is missing: " + str(artifacts) + "\n"
            "Attach pilkwang/biohub-tracking-support-pack-50ep-v1 with Add Input."
        )
    if not has_model_artifact(artifacts):
        raise FileNotFoundError(
            "The attached support dataset does not contain the expected repo/weights: " + str(artifacts)
        )
    return artifacts

def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates = [artifacts / "wheels", artifacts]
    return [path for path in candidates if _has_package_file(path)]


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)

# %% [markdown] papermill={"duration": 0.005804, "end_time": "2026-07-23T00:37:55.992525+00:00", "exception": false, "start_time": "2026-07-23T00:37:55.986721+00:00", "status": "completed"}
# ## Predict Candidate Graphs
#
# The inference step writes one `.geff` graph per test video. Keeping graph
# prediction separate from CSV conversion makes the graph repair and diagnostics
# transparent.
#

# %% _kg_hide-input=true _kg_hide-output=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 559.568674, "end_time": "2026-07-23T00:47:15.566717+00:00", "exception": false, "start_time": "2026-07-23T00:37:55.998043+00:00", "status": "completed"} tags=["hide-input"]
# STRICT INPUT GUARD
if not TEST_DIR.exists():
    raise FileNotFoundError(
        "Competition test data is missing at " + str(TEST_DIR) + "\n"
        "Attach Biohub - Cell Tracking During Development with Add Input."
    )

# BIOHUB CUDA HARD GUARD
import torch as _torch_guard
print("torch", _torch_guard.__version__, "cuda", _torch_guard.cuda.is_available(), flush=True)
if not _torch_guard.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stop before inference to avoid CPU timeout.")
print("device=cuda", _torch_guard.cuda.get_device_name(0), flush=True)

# PATCH: expand detector test-time augmentation to spatial D4 transforms.
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
_new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
if _old in _s:
    _ps.write_text(_s.replace(_old, _new))
    print("TTA patch applied (spatial D4 transforms)")
else:
    print("TTA WARNING: block not found - using default 4-way")

def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


test_stems = list_test_stems()
print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--weights",
    WEIGHTS_RELATIVE,
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])

start_time = time.time()
print(" ".join(predict_cmd))
subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
predict_seconds = time.time() - start_time
print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")


# %% [markdown] papermill={"duration": 0.006037, "end_time": "2026-07-23T00:47:15.579346+00:00", "exception": false, "start_time": "2026-07-23T00:47:15.573309+00:00", "status": "completed"}
# ## Build `submission.csv`
#
# Rows are streamed directly to disk with the required schema. This avoids holding
# the full hidden-test submission table in memory.
#
# The standard gap closer intentionally handles only one missing frame. Two-missing-frame repair is handled by the stricter `gap2` pass so that a loose environment override cannot introduce non-consecutive edges.
#

# %% _kg_hide-input=true jupyter={"source_hidden": true} kg_hide_input=true kg_hide_output=true papermill={"duration": 213.454573, "end_time": "2026-07-23T00:50:49.039852+00:00", "exception": false, "start_time": "2026-07-23T00:47:15.585279+00:00", "status": "completed"} tags=["hide-input"]
import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1



def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint



def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print("Could not read DeepCenter manifest:", manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []
    for section in [
        manifest.get("model", {}),
        manifest.get("models", {}).get("full_frame_center", {}) if isinstance(manifest.get("models", {}), dict) else {},
        manifest.get("full_frame_center", {}),
    ]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []
    for section in sections:
        for key in ("weight_path", "path"):
            rel = section.get(key)
            if isinstance(rel, str) and rel:
                candidates.append(root / rel)
        for key in ("last_checkpoint", "best_checkpoint"):
            item = section.get(key)
            if isinstance(item, dict):
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)
    for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
        candidates.append(root / "weights" / "full_frame_center" / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates


def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", DEEPCENTER_CHECKPOINT_DEFAULT).strip()
    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get("BIOHUB_DEEPCENTER_MANIFEST", DEEPCENTER_MANIFEST_DEFAULT).strip()
    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))

    input_root = Path("/kaggle/input")
    preferred_dirs = [
        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    ]
    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / "ARTIFACT_MANIFEST.json"))
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.append(directory / "weights" / "full_frame_center" / name)
            candidates.append(directory / name)
    if input_root.exists():
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.extend(sorted(input_root.glob(f"**/full_frame_center/**/{name}")))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        path = path.expanduser()
        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


try:
    import torch
except Exception as _dc_torch_error:
    torch = None


if torch is not None:
    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None

def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print("DeepCenter add-only repair gate disabled by configuration.")
        return None
    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError("torch is required for DeepCenter add-only repair gate")
        print("DeepCenter add-only repair gate skipped because torch is unavailable.")
        return None
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_errors: list[str] = []
    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue
        try:
            print("Trying DeepCenter add-only gate checkpoint:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("checkpoint has no model_state")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(
                    f"expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}"
                )
            cfg = SimpleNamespace(**checkpoint.get("config", {}))
            model = _DCDeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            print("Loaded DeepCenter add-only gate checkpoint:", checkpoint_path)
            print("DeepCenter checkpoint epoch:", checkpoint.get("epoch"), "best_score:", checkpoint.get("best_score"))
            return {
                "model": model,
                "cfg": cfg,
                "device": device,
                "path": checkpoint_path,
                "torch": torch,
            }
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print("Skipping incompatible DeepCenter checkpoint:", checkpoint_path, "|", type(exc).__name__, exc)
    message = "No usable DeepCenter checkpoint found for add-only repair gate."
    if REQUIRE_DEEPCENTER_VETO:
        checked = "\n".join(str(p) for p in _dc_checkpoint_candidates()[:80])
        errors = "\n".join(load_errors[-20:])
        raise FileNotFoundError(message + "\nChecked:\n" + checked + ("\nLoad errors:\n" + errors if errors else ""))
    print(message)
    return None


def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def deepcenter_heatmap_for_frame(
    dataset: str,
    t: int,
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)
    if cached is not None:
        return cached
    model = detector_bundle["model"]
    cfg = detector_bundle["cfg"]
    device = detector_bundle["device"]
    torch_mod = detector_bundle["torch"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)
    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(device=device, dtype=torch_mod.float32)
        heatmap = torch_mod.sigmoid(model(tensor))[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap


def deepcenter_score_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(dataset, int(t), detector_bundle, frame_cache, heatmap_cache)
    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle["cfg"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = max(0, z - DEEPCENTER_SCORE_WIN_Z), min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1)
    y0, y1 = max(0, y - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1)
    x0, x1 = max(0, x - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1)
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None


def deepcenter_accept_repair_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True
    if detector_bundle is None or dataset is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True

def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached

        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {
                node_id: GAP_DENSITY_REFERENCE_UM
                for node_id in frame_ids
            }
            density_cache[t] = result
            return result

        positions = np.stack(
            [_position_um(nodes_by_id[node_id]) for node_id in frame_ids]
        )
        tree = cKDTree(positions)
        query_k = min(
            len(frame_ids),
            max(2, GAP_DENSITY_NEIGHBORS + 1),
        )
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]

        result: dict[int, float] = {}
        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[
                np.isfinite(neighbour_distances)
            ]
            spacing = (
                float(np.median(neighbour_distances))
                if neighbour_distances.size
                else GAP_DENSITY_REFERENCE_UM
            )
            result[node_id] = spacing

        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros(
                (len(end_ids), len(start_ids)),
                dtype=np.float64,
            )
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (
                            source_spacing.get(
                                end_ids[i],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                            + target_spacing.get(
                                start_ids[j],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                        )
                        step_delta = float(
                            np.clip(
                                GAP_DENSITY_GAIN
                                * (
                                    local_spacing
                                    - GAP_DENSITY_REFERENCE_UM
                                ),
                                -GAP_DENSITY_MAX_STEP_DELTA_UM,
                                GAP_DENSITY_MAX_STEP_DELTA_UM,
                            )
                        )
                        adaptive_threshold[i, j] = (
                            threshold_um + step_delta * (gap + 1)
                        )
                        stats[
                            "gap_density_step_delta_milli_sum"
                        ] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold

            stats["gap_density_candidates_expanded"] += int(
                (adaptive_allowed & ~base_allowed).sum()
            )
            stats["gap_density_candidates_restricted"] += int(
                (base_allowed & ~adaptive_allowed).sum()
            )
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats[
                        "gap_density_selected_outside_base"
                    ] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                requires_center_confirmation = (
                    DEEPCENTER_GAP_VETO and marginal_gap and middle_reused
                )
                if DEEPCENTER_GAP_VETO and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                elif DEEPCENTER_GAP_VETO and not middle_reused:
                    stats["deepcenter_gap_bypassed_synthetic_node"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    dataset,
                    mid_t,
                    node_point(middle),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "gap",
                    DEEPCENTER_GAP_THRESHOLD,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(
                SHORT_TRACK_RESCUE_MAX_NODES_ABS,
                max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue
                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_synthetic_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )

    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id]

    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)

    return nodes_by_id, edges, stats


DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
print(f"Found {len(geffs)} prediction graphs")
if len(geffs) != len(test_stems):
    found = {path.stem for path in geffs}
    missing = sorted(set(test_stems) - found)
    raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

stats_rows: list[dict[str, object]] = []
seen_datasets: set[str] = set()
row_id = 0
total_nodes = 0
total_edges = 0

with SUBMISSION_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for geff_path in geffs:
        dataset = geff_path.stem
        seen_datasets.add(dataset)
        graph = graph_from_geff(geff_path)

        nodes_by_id: dict[int, dict[str, object]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }

        raw_edges: list[dict[str, object]] = []
        for row in graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        for node_id in sorted(nodes_by_id):
            node = nodes_by_id[node_id]
            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node["node_id"]),
                "t": int(node["t"]),
                "z": max(0, int(round(float(node["z"])))),
                "y": max(0, int(round(float(node["y"])))),
                "x": max(0, int(round(float(node["x"])))),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        division_sources: dict[int, int] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            target_id = int(edge["target_id"])
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise AssertionError(f"{dataset}: dangling edge after filtering")
            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            })
            row_id += 1
            division_sources[source_id] = division_sources.get(source_id, 0) + 1

        node_count = len(nodes_by_id)
        edge_count = len(edges)
        total_nodes += node_count
        total_edges += edge_count
        stats_rows.append({
            "dataset": dataset,
            "raw_nodes": raw_node_count,
            "nodes": node_count,
            "raw_edges": filter_stats["raw_edges"],
            "edges": edge_count,
            "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
            "edge_to_node_ratio": edge_count / max(node_count, 1),
            "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
            **filter_stats,
        })

expected_datasets = set(test_stems)
missing_datasets = sorted(expected_datasets - seen_datasets)
extra_datasets = sorted(seen_datasets - expected_datasets)
if missing_datasets or extra_datasets:
    raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
assert total_nodes > 0, "No node rows produced"

header = SUBMISSION_PATH.open().readline().strip().split(",")
assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
stats["predict_minutes_total"] = predict_seconds / 60.0
stats["experiment_tag"] = EXPERIMENT_TAG
stats.to_csv(RUN_STATS_PATH, index=False)

print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
print(f"Wrote {RUN_STATS_PATH}")
display(stats.describe(include="all"))
display(pd.read_csv(SUBMISSION_PATH, nrows=8))


# %% [markdown] papermill={"duration": 0.007011, "end_time": "2026-07-23T00:50:49.053797+00:00", "exception": false, "start_time": "2026-07-23T00:50:49.046786+00:00", "status": "completed"}
# ## Promotion rule
#
# - Legitimate clean leaderboard baseline: `0.908`.
# - This notebook changes only the adaptive short-track rescue branch.
# - Promote only when fixed-8 LocalCV improves, node recall or edge TP increases, and FP growth remains controlled.
# - Reject the candidate when rescue adds many rows without a positive paired CV delta.
#

# %% papermill={"duration": 0.497297, "end_time": "2026-07-23T00:50:49.558091+00:00", "exception": false, "start_time": "2026-07-23T00:50:49.060794+00:00", "status": "completed"}
# ＊＊＊＊＊ GPT FEEDBACK START / END ＊＊＊＊＊
import hashlib
from pathlib import Path

_feedback_path = Path(SUBMISSION_PATH)
_feedback_sha = hashlib.sha256(_feedback_path.read_bytes()).hexdigest()
_feedback_rows = int(total_nodes + total_edges)
_feedback_divisions = int(stats["division_like_sources"].sum()) if "division_like_sources" in stats else 0
_feedback_activation = int(
    stats["gap_density_candidates_expanded"].sum()
    + stats["gap_density_candidates_restricted"].sum()
)
_feedback_cols = [
    "dataset", "nodes", "edges", "division_like_sources",
    "gap_candidates", "gap_pairs_selected",
    "gap_density_candidates_expanded", "gap_density_selected_outside_base",
]

# Clean-submission audit. This checks only biological/format validity and does not alter predictions.
_audit_df = pd.read_csv(_feedback_path)
_node_df = _audit_df[_audit_df["row_type"] == "node"].copy()
_edge_df = _audit_df[_audit_df["row_type"] == "edge"].copy()
_clean_audit_rows = []
for _dataset in test_stems:
    _zarr_meta = json.loads((TEST_DIR / f"{_dataset}.zarr" / "0" / "zarr.json").read_text())
    _shape = tuple(int(v) for v in _zarr_meta["shape"])
    _n = _node_df[_node_df["dataset"] == _dataset]
    _e = _edge_df[_edge_df["dataset"] == _dataset]
    _node_times = dict(zip(_n["node_id"].astype(int), _n["t"].astype(int)))
    _valid_ids = set(_node_times)
    _dangling = int(sum((int(s) not in _valid_ids) or (int(t) not in _valid_ids) for s, t in zip(_e["source_id"], _e["target_id"])))
    _nonconsecutive = int(sum(_node_times.get(int(t), -999) != _node_times.get(int(s), -999) + 1 for s, t in zip(_e["source_id"], _e["target_id"])))
    _outdegree = _e.groupby("source_id").size() if len(_e) else pd.Series(dtype=int)
    _max_outdegree = int(_outdegree.max()) if len(_outdegree) else 0
    _bad_time = int(((_n["t"] < 0) | (_n["t"] >= _shape[0])).sum())
    _bad_z = int(((_n["z"] < 0) | (_n["z"] >= _shape[1])).sum())
    _bad_y = int(((_n["y"] < 0) | (_n["y"] >= _shape[2])).sum())
    _bad_x = int(((_n["x"] < 0) | (_n["x"] >= _shape[3])).sum())
    _clean_audit_rows.append({
        "dataset": _dataset,
        "node_time_violations": _bad_time,
        "coordinate_violations": _bad_z + _bad_y + _bad_x,
        "dangling_edges": _dangling,
        "nonconsecutive_edges": _nonconsecutive,
        "max_outdegree": _max_outdegree,
    })
_clean_audit = pd.DataFrame(_clean_audit_rows)
if int(_clean_audit[["node_time_violations", "coordinate_violations", "dangling_edges", "nonconsecutive_edges"]].to_numpy().sum()) != 0:
    raise AssertionError("Clean-submission audit failed:\n" + _clean_audit.to_string(index=False))
if int(_clean_audit["max_outdegree"].max()) > 2:
    raise AssertionError("Unexpected hub-like source detected:\n" + _clean_audit.to_string(index=False))

_REF_124_ROWS, _REF_124_NODES, _REF_124_EDGES, _REF_124_DIVISIONS = 236185, 120210, 115975, 308

print("")
print("＊＊＊＊＊ GPT FEEDBACK START ＊＊＊＊＊")
print("Notebook: biohub_132_clean_short_track_rescue_lightcv_nohack")
print("Public title: 🧬 Biohub 132 | Clean Short-Track Rescue + Light Local CV | No Hack")
print("Legitimate clean leaderboard baseline after metric fix: 0.908")
print("Strategy: rescue only high-confidence five-node components removed by the minimum-length-six filter")
print("Detection threshold: 0.96875")
print("ILP appearance cost: 0.0")
print("ILP disappearance cost: 1.575")
print("Motion relink tight / relaxed gate: 6.0 / 9.5")
print("Adaptive short-track rescue: True")
print("Rescue minimum length: 5")
print("Rescue minimum mean edge probability: 0.90")
print("Rescue maximum mean edge distance: 2.75 µm")
print("Rescue maximum node fraction / absolute cap: 0.006 / 60")
print("Automatic input discovery: False")
print("Artificial hub/fork nodes added: False")
print("Negative-time nodes added: False")
print("Out-of-volume nodes added: False")
print("Cross-clip graph edges added: False")
print("Hidden-test labels used: False")
print("Model inference passes: 1")
print(f"Prediction minutes: {predict_seconds / 60.0:.2f}")
print(f"Rows: {_feedback_rows}")
print(f"Node rows: {int(total_nodes)}")
print(f"Edge rows: {int(total_edges)}")
print(f"Division-like sources: {_feedback_divisions}")
print(f"SHA256: {_feedback_sha}")
print(f"Short-track rescue triggered datasets: {int((stats['short_track_rescue_triggered'] > 0).sum()) if 'short_track_rescue_triggered' in stats else 0}")
print(f"Short-track components rescued: {int(stats['short_track_rescue_components'].sum()) if 'short_track_rescue_components' in stats else 0}")
print(f"Short-track nodes rescued: {int(stats['short_track_rescue_nodes'].sum()) if 'short_track_rescue_nodes' in stats else 0}")
print("Difference versus the historical Notebook-124 clean graph reference:")
print(f"- Rows delta: {_feedback_rows - _REF_124_ROWS:+d}")
print(f"- Node delta: {int(total_nodes) - _REF_124_NODES:+d}")
print(f"- Edge delta: {int(total_edges) - _REF_124_EDGES:+d}")
print(f"- Division-like source delta: {_feedback_divisions - _REF_124_DIVISIONS:+d}")
print("Per-dataset key statistics:")
print(stats[_feedback_cols + ['short_track_rescue_triggered', 'short_track_rescue_components', 'short_track_rescue_nodes']].to_string(index=False))
print("Clean-submission audit:")
print(_clean_audit.to_string(index=False))
print("Submit recommendation: PROMOTE ONLY IF FIXED-8 LITE CV IMPROVES WITH CONTROLLED FP")
print("＊＊＊＊＊ GPT FEEDBACK END ＊＊＊＊＊")


# %% papermill={"duration": 0.017448, "end_time": "2026-07-23T00:50:49.583398+00:00", "exception": false, "start_time": "2026-07-23T00:50:49.565950+00:00", "status": "completed"}
# FINAL KAGGLE SUBMISSION GUARD
from pathlib import Path
import pandas as pd
_submit_file = Path("/kaggle/working/submission.csv")
if not _submit_file.exists():
    raise FileNotFoundError("submission.csv was not created at /kaggle/working/submission.csv")
_submit_check = pd.read_csv(_submit_file, nrows=5)
_expected = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
if list(_submit_check.columns) != _expected:
    raise ValueError(f"Unexpected submission columns: {list(_submit_check.columns)}")
print("SUBMISSION_READY: /kaggle/working/submission.csv")
print("Kaggle output filename: submission.csv")


# %% [markdown] papermill={"duration": 0.007121, "end_time": "2026-07-23T00:50:49.597816+00:00", "exception": false, "start_time": "2026-07-23T00:50:49.590695+00:00", "status": "completed"}
# ## Fixed-8 Official-Spec Lite CV
#
# The hidden-test `submission.csv` is written and protected before diagnostics. The fixed public-train split then measures whether rescued five-node components improve node recall and edge TP without adding excessive FP.
#
# Inspect `short_track_rescue_components`, `short_track_rescue_nodes`, edge TP/FP/FN, node recall, and the paired score delta together.
#

# %% papermill={"duration": 1281.78293, "end_time": "2026-07-23T01:12:11.388019+00:00", "exception": false, "start_time": "2026-07-23T00:50:49.605089+00:00", "status": "completed"}

# -----------------------------------------------------------------------------
# Guarded Lite-CV phase. submission.csv already exists and has passed the final
# Kaggle schema guard above. This phase never overwrites that file.
# -----------------------------------------------------------------------------
import traceback

_SUBMISSION_FILE_SAFE = Path('/kaggle/working/submission.csv')
_SUBMISSION_SHA_BEFORE_CV = hashlib.sha256(_SUBMISSION_FILE_SAFE.read_bytes()).hexdigest()
_SUBMISSION_ROWS_BEFORE_CV = sum(1 for _ in _SUBMISSION_FILE_SAFE.open('r', encoding='utf-8')) - 1

CV_STATUS_PATH = WORKING_DIR / '127_lite_cv_status.json'
CV_FIXED_DATASETS = [
    '44b6_0113de3b', '44b6_0b24845f', '44b6_341df25f', '44b6_e57ff5c6',
    '6bba_05b6850b', '6bba_05db0fb1', '6bba_969618f6', '6bba_fc83837d',
]

_original_test_dir = TEST_DIR
_original_submission_path = SUBMISSION_PATH
_original_run_stats_path = RUN_STATS_PATH
_original_experiment_tag = EXPERIMENT_TAG

cv_status = {
    'status': 'not_started',
    'datasets': CV_FIXED_DATASETS,
    'submission_sha_before_cv': _SUBMISSION_SHA_BEFORE_CV,
    'submission_rows_before_cv': _SUBMISSION_ROWS_BEFORE_CV,
}

try:
    TRAIN_DIR = COMP_DIR / 'train'
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f'CV train directory not found: {TRAIN_DIR}')

    TEST_DIR = TRAIN_DIR
    SUBMISSION_PATH = WORKING_DIR / '127_official_spec_smoke_cv_predictions.csv'
    RUN_STATS_PATH = WORKING_DIR / '127_official_spec_smoke_cv_run_stats.csv'
    EXPERIMENT_TAG = '127_combined_offline_smoke_cv'
    CV_DATASETS_OVERRIDE = list(CV_FIXED_DATASETS)
    CV_ALLOW_MIXED_FALLBACK = False
    OFFICIAL_MAX_MATCH_UM = 7.0
    OFFICIAL_GEFF_DIR = WORKING_DIR / '127_official_spec_smoke_cv_geffs'
    OFFICIAL_METRIC_ROWS_PATH = WORKING_DIR / '127_official_spec_smoke_metric_per_dataset.csv'
    OFFICIAL_METRIC_SUMMARY_PATH = WORKING_DIR / '127_official_spec_smoke_metric_summary.json'
    OFFICIAL_METRIC_STDOUT_PATH = WORKING_DIR / '127_official_spec_smoke_metric_notes.txt'

    # Remove only temporary model prediction graphs. The already-written submission.csv
    # remains untouched in /kaggle/working.
    shutil.rmtree(REPO_DIR / 'predictions', ignore_errors=True)


    from dataclasses import dataclass
    from collections import defaultdict
    import hashlib
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    OFFICIAL_SPEC_REPOSITORY = "royerlab/kaggle-cell-tracking-competition"
    OFFICIAL_SPEC_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
    OFFICIAL_SPEC_DOCUMENT = "metrics.md"
    OFFICIAL_SPEC_MAX_DISTANCE_UM = 7.0
    OFFICIAL_SPEC_NODE_PENALTY_A = 0.1
    OFFICIAL_SPEC_DIVISION_WEIGHT = 0.1
    OFFICIAL_SPEC_EXACT_SOURCE_COPY = False

    @dataclass(frozen=True)
    class SpecResult:
        edge_tp: int
        edge_fp: int
        edge_fn: int
        division_tp: int
        division_fp: int
        division_fn: int
        num_pred_nodes: int
        num_gt_nodes: int
        matched_gt_nodes: int


    def _normalise_nodes(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["node_id", "t", "z", "y", "x"]
        out = df.loc[:, cols].copy()
        out["node_id"] = out["node_id"].astype(np.int64)
        out["t"] = out["t"].astype(np.int64)
        for c in ("z", "y", "x"):
            out[c] = out[c].astype(np.float64)
        return out.drop_duplicates("node_id", keep="first").reset_index(drop=True)


    def _normalise_edges(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["source_id", "target_id"])
        out = df.loc[:, ["source_id", "target_id"]].copy()
        out["source_id"] = out["source_id"].astype(np.int64)
        out["target_id"] = out["target_id"].astype(np.int64)
        out = out[out.source_id != out.target_id].drop_duplicates().reset_index(drop=True)
        valid = set(nodes.node_id.astype(int))
        out = out[out.source_id.isin(valid) & out.target_id.isin(valid)].copy()
        times = nodes.set_index("node_id")["t"].to_dict()
        out = out[
            out.apply(lambda r: int(times[int(r.target_id)]) == int(times[int(r.source_id)]) + 1, axis=1)
        ].copy()
        # Patched host scorer accepts at most two outgoing branches. Keep a deterministic
        # two-edge subset only for malformed diagnostic inputs; clean Notebook 124 is already <=2.
        out = out.sort_values(["source_id", "target_id"]).groupby("source_id", as_index=False).head(2)
        return out.reset_index(drop=True)


    def _audit_prediction_nodes(nodes: pd.DataFrame) -> dict[str, int]:
        arr = nodes[["z", "y", "x"]].to_numpy(float)
        return {
            "negative_time_nodes": int((nodes.t < 0).sum()),
            "nonfinite_coordinate_nodes": int((~np.isfinite(arr).all(axis=1)).sum()),
            # Sentinel exploit coordinates are many orders outside the microscopy volume.
            "sentinel_coordinate_nodes": int((np.abs(arr) >= 1000.0).any(axis=1).sum()),
        }


    def _node_match(
        pred_nodes: pd.DataFrame,
        gt_nodes: pd.DataFrame,
        scale=(1.625, 0.40625, 0.40625),
        max_distance_um: float = 7.0,
    ) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
        """Same-time optimal bipartite matching under a physical-distance gate."""
        p2g: dict[int, int] = {}
        g2p: dict[int, int] = {}
        distances: dict[int, float] = {}
        scale_arr = np.asarray(scale, dtype=np.float64)
        common_times = sorted(set(pred_nodes.t.astype(int)) & set(gt_nodes.t.astype(int)))
        for t in common_times:
            p = pred_nodes[pred_nodes.t == t]
            g = gt_nodes[gt_nodes.t == t]
            if p.empty or g.empty:
                continue
            pc = p[["z", "y", "x"]].to_numpy(float) * scale_arr
            gc = g[["z", "y", "x"]].to_numpy(float) * scale_arr
            dist = np.sqrt(((gc[:, None, :] - pc[None, :, :]) ** 2).sum(axis=2))
            big = max_distance_um * 1000.0 + 1.0
            cost = np.where(dist <= max_distance_um, dist, big)
            rr, cc = linear_sum_assignment(cost)
            gids = g.node_id.astype(int).to_numpy()
            pids = p.node_id.astype(int).to_numpy()
            for r, c in zip(rr, cc):
                d = float(dist[int(r), int(c)])
                if d > max_distance_um:
                    continue
                gid, pid = int(gids[int(r)]), int(pids[int(c)])
                p2g[pid] = gid
                g2p[gid] = pid
                distances[pid] = d
        return p2g, g2p, distances


    def _adjacency(edges: pd.DataFrame):
        succ, pred = defaultdict(list), defaultdict(list)
        for r in edges.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            succ[s].append(t); pred[t].append(s)
        return succ, pred


    def _components(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[int, int]:
        adj = defaultdict(set)
        for r in edges.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            adj[s].add(t); adj[t].add(s)
        comp, cid = {}, 0
        for n in nodes.node_id.astype(int):
            if n in comp:
                continue
            stack=[n]; comp[n]=cid
            while stack:
                x=stack.pop()
                for y in adj.get(x, ()):
                    if y not in comp:
                        comp[y]=cid; stack.append(y)
            cid += 1
        return comp


    def _edge_counts(pred_edges, gt_edges, p2g):
        gt_set = {(int(r.source_id), int(r.target_id)) for r in gt_edges.itertuples(index=False)}
        gt_succ, gt_pred = _adjacency(gt_edges)
        tp_gt = set()
        fp = 0
        for r in pred_edges.itertuples(index=False):
            ps, pt = int(r.source_id), int(r.target_id)
            gs, gt = p2g.get(ps), p2g.get(pt)
            if gs is not None and gt is not None and (gs, gt) in gt_set and (gs, gt) not in tp_gt:
                tp_gt.add((gs, gt))
                continue
            evaluable = (gs is not None and len(gt_succ.get(gs, ())) > 0) or (
                gt is not None and len(gt_pred.get(gt, ())) > 0
            )
            if evaluable:
                fp += 1
        tp = len(tp_gt)
        fn = max(0, len(gt_set) - tp)
        return tp, fp, fn


    def _kuhn_maximum_matching(valid_pairs: dict[int, set[int]]) -> dict[int, int]:
        """Map right(pred fork) -> left(GT division), maximum cardinality."""
        match_r: dict[int, int] = {}
        def dfs(left: int, seen: set[int]) -> bool:
            for right in sorted(valid_pairs.get(left, ())):
                if right in seen:
                    continue
                seen.add(right)
                if right not in match_r or dfs(match_r[right], seen):
                    match_r[right] = left
                    return True
            return False
        for left in sorted(valid_pairs):
            dfs(left, set())
        return match_r


    def _division_counts(pred_nodes, pred_edges, gt_nodes, gt_edges, global_p2g, scale, max_distance_um):
        psucc, ppred = _adjacency(pred_edges)
        gsucc, gpred = _adjacency(gt_edges)
        gt_components = _components(gt_nodes, gt_edges)
        pred_forks = {n for n, ch in psucc.items() if len(ch) >= 2}
        gt_divisions = {n for n, ch in gsucc.items() if len(ch) == 2}
        pred_by_id = pred_nodes.set_index("node_id")
        gt_by_id = gt_nodes.set_index("node_id")

        valid_pairs: dict[int, set[int]] = defaultdict(set)
        local_candidate_forks: set[int] = set()

        for gd in sorted(gt_divisions):
            daughters = list(gsucc[gd])[:2]
            parent_side = {gd, *gpred.get(gd, [])}
            daughter_sets = []
            for d in daughters:
                daughter_sets.append({d, *gsucc.get(d, [])})
            local_gt_ids = sorted(parent_side | set().union(*daughter_sets))
            local_gt = gt_nodes[gt_nodes.node_id.isin(local_gt_ids)]
            if local_gt.empty:
                continue
            # Independent local matching, as specified for division windows.
            time_min, time_max = int(local_gt.t.min()), int(local_gt.t.max())
            local_pred = pred_nodes[(pred_nodes.t >= time_min) & (pred_nodes.t <= time_max)]
            lp2g, lg2p, _ = _node_match(local_pred, local_gt, scale, max_distance_um)
            parent_pred = {lg2p[g] for g in parent_side if g in lg2p}
            candidates = {p for p in parent_pred if p in pred_forks}
            for p in parent_pred:
                candidates.update(c for c in psucc.get(p, []) if c in pred_forks)
            local_candidate_forks.update(candidates)

            for fork in candidates:
                children = list(psucc.get(fork, []))
                if len(children) < 2:
                    continue
                branches = [{c, *psucc.get(c, [])} for c in children]
                # Parent anchor must be the fork or its immediate predecessor.
                parent_ok = fork in parent_pred or any(p in parent_pred for p in ppred.get(fork, []))
                if not parent_ok:
                    continue
                # Reject merged direct children and merged fallback grandchildren.
                if any(len(ppred.get(c, [])) != 1 for c in children):
                    continue
                evidence = [[False] * len(branches) for _ in range(2)]
                for di, dset in enumerate(daughter_sets):
                    for bi, bset in enumerate(branches):
                        evidence[di][bi] = any(lp2g.get(p) in dset for p in bset)
                # Two daughter lineages must map to two distinct predicted branches.
                branch_pair_ok = any(
                    evidence[0][b0] and evidence[1][b1]
                    for b0 in range(len(branches)) for b1 in range(len(branches)) if b0 != b1
                )
                if branch_pair_ok:
                    valid_pairs[gd].add(fork)

        matched = _kuhn_maximum_matching(valid_pairs)
        tp_forks = set(matched)
        tp = len(tp_forks)
        fn = max(0, len(gt_divisions) - tp)

        # Conservative FP evidence from the published patched specification.
        fp_forks: set[int] = set()
        for fork in pred_forks - tp_forks:
            g = global_p2g.get(fork)
            if g is not None and len(gsucc.get(g, [])) > 0:
                fp_forks.add(fork); continue
            if fork in local_candidate_forks:
                fp_forks.add(fork); continue
            children = list(psucc.get(fork, []))
            if any(len(ppred.get(c, [])) != 1 for c in children):
                fp_forks.add(fork); continue
            branch_components = []
            for c in children:
                direct_g = global_p2g.get(c)
                if direct_g is not None:
                    branch_components.append(gt_components.get(direct_g)); continue
                comps = {gt_components.get(global_p2g[gch]) for gch in psucc.get(c, []) if gch in global_p2g}
                comps.discard(None)
                if len(comps) == 1:
                    branch_components.append(next(iter(comps)))
            branch_components = [c for c in branch_components if c is not None]
            if len(set(branch_components)) >= 2:
                fp_forks.add(fork)
        return tp, len(fp_forks), fn


    def official_spec_evaluate(pred_nodes, pred_edges, gt_nodes, gt_edges, scale=(1.625,0.40625,0.40625), max_distance_um=7.0):
        pred_nodes = _normalise_nodes(pred_nodes)
        gt_nodes = _normalise_nodes(gt_nodes)
        audit = _audit_prediction_nodes(pred_nodes)
        if any(audit.values()):
            raise ValueError(f"Prediction failed clean-node audit: {audit}")
        pred_edges = _normalise_edges(pred_edges, pred_nodes)
        gt_edges = _normalise_edges(gt_edges, gt_nodes)
        p2g, g2p, _ = _node_match(pred_nodes, gt_nodes, scale, max_distance_um)
        etp, efp, efn = _edge_counts(pred_edges, gt_edges, p2g)
        dtp, dfp, dfn = _division_counts(
            pred_nodes, pred_edges, gt_nodes, gt_edges, p2g, scale, max_distance_um
        )
        return SpecResult(etp, efp, efn, dtp, dfp, dfn, len(pred_nodes), len(gt_nodes), len(g2p))


    def _jaccard(tp, fp, fn):
        den = int(tp) + int(fp) + int(fn)
        return float(tp) / den if den else 1.0


    def official_spec_per_sample(r: SpecResult, estimated_total_nodes: float):
        edge_j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        if np.isfinite(estimated_total_nodes) and estimated_total_nodes > 0:
            node_ratio = r.num_pred_nodes / float(estimated_total_nodes)
            factor = 1.0 - OFFICIAL_SPEC_NODE_PENALTY_A * (
                (r.num_pred_nodes - float(estimated_total_nodes)) / float(estimated_total_nodes)
            )
            adjusted = max(0.0, edge_j * factor)
        else:
            node_ratio, adjusted = float("nan"), edge_j
        return {
            "edge_jaccard": edge_j,
            "adj_edge_jaccard": adjusted,
            "division_jaccard": _jaccard(r.division_tp, r.division_fp, r.division_fn),
            "node_recall": (r.matched_gt_nodes / r.num_gt_nodes) if r.num_gt_nodes else 1.0,
            "total_node_ratio": node_ratio,
            "edge_weight": r.edge_tp + r.edge_fp + r.edge_fn,
        }


    def official_spec_summarise(rows: list[dict]) -> dict:
        w = np.array([float(r.get("edge_weight", 0)) for r in rows], dtype=float)
        adj = np.array([float(r["adj_edge_jaccard"]) for r in rows], dtype=float)
        edge_adj = float(np.average(adj, weights=w)) if w.sum() > 0 else float(adj.mean())
        etp=sum(int(r["edge_tp"]) for r in rows); efp=sum(int(r["edge_fp"]) for r in rows); efn=sum(int(r["edge_fn"]) for r in rows)
        dtp=sum(int(r["division_tp"]) for r in rows); dfp=sum(int(r["division_fp"]) for r in rows); dfn=sum(int(r["division_fn"]) for r in rows)
        matched=sum(int(r["matched_gt_nodes"]) for r in rows); ngt=sum(int(r["num_gt_nodes"]) for r in rows)
        div_j=_jaccard(dtp,dfp,dfn)
        return {
            "score": edge_adj + OFFICIAL_SPEC_DIVISION_WEIGHT * div_j,
            "adj_edge_jaccard": edge_adj,
            "edge_jaccard_micro": _jaccard(etp,efp,efn),
            "division_jaccard": div_j,
            "edge_tp": etp, "edge_fp": efp, "edge_fn": efn,
            "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
            "node_recall": matched/ngt if ngt else 1.0,
        }

    # ------------------------- pre-GPU self-tests -------------------------
    def _df_nodes(rows): return pd.DataFrame(rows, columns=["node_id","t","z","y","x"])
    def _df_edges(rows): return pd.DataFrame(rows, columns=["source_id","target_id"])
    _gt_n=_df_nodes([(1,0,0,0,0),(2,1,0,1,0),(3,1,0,-1,0)])
    _gt_e=_df_edges([(1,2),(1,3)])
    _pr_n=_gt_n.copy(); _pr_e=_gt_e.copy()
    _r=official_spec_evaluate(_pr_n,_pr_e,_gt_n,_gt_e)
    assert (_r.edge_tp,_r.edge_fp,_r.edge_fn)==(2,0,0), _r
    assert (_r.division_tp,_r.division_fn)==(1,0), _r
    # An unmatched sentinel fork has no local GT evidence and must never become a TP.
    _hn=_df_nodes([(10,-1000,-10000,-10000,-10000),(11,-999,-10000,-10000,-10000),(12,-999,-10001,-10000,-10000)])
    try:
        official_spec_evaluate(_hn,_df_edges([(10,11),(10,12)]),_gt_n,_gt_e)
        raise AssertionError("sentinel exploit audit did not fire")
    except ValueError:
        pass
    assert abs(_jaccard(2,1,1)-0.5) < 1e-12

    _SPEC_FINGERPRINT = hashlib.sha256(
        (OFFICIAL_SPEC_REPOSITORY + OFFICIAL_SPEC_COMMIT + OFFICIAL_SPEC_DOCUMENT + "offline-spec-v1").encode()
    ).hexdigest()
    print("OFFLINE OFFICIAL-SPEC PREFLIGHT PASSED — GPU inference may start.")
    print("Repository provenance:", OFFICIAL_SPEC_REPOSITORY)
    print("Pinned specification commit:", OFFICIAL_SPEC_COMMIT)
    print("Exact official source copy:", OFFICIAL_SPEC_EXACT_SOURCE_COPY)
    print("Offline evaluator fingerprint:", _SPEC_FINGERPRINT)

    # STRICT TRAIN INPUT GUARD
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            "Competition train data is missing at " + str(TRAIN_DIR) + "\n"
            "Attach Biohub - Cell Tracking During Development with Add Input."
        )

    # BIOHUB CUDA HARD GUARD
    import torch as _torch_guard
    print("torch", _torch_guard.__version__, "cuda", _torch_guard.cuda.is_available(), flush=True)
    if not _torch_guard.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop before inference to avoid CPU timeout.")
    print("device=cuda", _torch_guard.cuda.get_device_name(0), flush=True)

    # PATCH: expand detector test-time augmentation to spatial D4 transforms.
    _ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
    _s = _ps.read_text()
    _old = """        if cfg.det_tta:
                tta_flips = [(-1,), (-2,), (-2, -1)]
                for dims in tta_flips:
                    imgs_flip = imgs.flip(dims)
                    _, det_flip = model.encode(imgs_flip)
                    for f in range(W):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                    del imgs_flip, det_flip
                for f in range(W):
                    det_logits[f] = det_logits[f] / 4"""
    _new = """        if cfg.det_tta:
                _nv = 1
                for dims in [(-1,), (-2,), (-2, -1)]:
                    imgs_flip = imgs.flip(dims)
                    _, det_flip = model.encode(imgs_flip)
                    for f in range(W):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                    del imgs_flip, det_flip
                    _nv += 1
                for _k in (1, 3):
                    imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                    _, det_rot = model.encode(imgs_rot)
                    for f in range(W):
                        det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                    del imgs_rot, det_rot
                    _nv += 1
                imgs_t = imgs.transpose(-1, -2)
                _, det_t = model.encode(imgs_t)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
                del imgs_t, det_t
                _nv += 1
                imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
                _, det_at = model.encode(imgs_at)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
                del imgs_at, det_at
                _nv += 1
                for f in range(W):
                    det_logits[f] = det_logits[f] / _nv"""
    if _old in _s:
        _ps.write_text(_s.replace(_old, _new))
        print("TTA patch applied (spatial D4 transforms)")
    else:
        print("TTA WARNING: block not found - using default 4-way")



    def paired_train_stems() -> list[str]:
        zarr_names = {p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr")}
        geff_names = {p.stem for p in TRAIN_DIR.glob("*.geff")}
        stems = sorted(zarr_names & geff_names)
        if not stems:
            raise FileNotFoundError(f"No paired train .zarr/.geff datasets found in {TRAIN_DIR}")
        return stems


    def _extract_split0_holdout(obj) -> list[str]:
        entries = obj
        if isinstance(obj, dict):
            for key in ("splits", "folds", "data"):
                if isinstance(obj.get(key), list):
                    entries = obj[key]
                    break
            else:
                entries = [obj]
        if not isinstance(entries, list):
            return []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            split_id = entry.get("split", entry.get("fold", idx))
            try:
                split_id = int(split_id)
            except Exception:
                continue
            if split_id != 0:
                continue
            for key in ("test", "val", "validation", "valid"):
                vals = entry.get(key)
                if isinstance(vals, list) and vals:
                    return [str(v) for v in vals]
        return []


    def resolve_cv_stems(all_stems: list[str]) -> tuple[list[str], str, str]:
        available = set(all_stems)
        if CV_DATASETS_OVERRIDE:
            missing = sorted(set(CV_DATASETS_OVERRIDE) - available)
            if missing:
                raise ValueError(f"BIOHUB_CV_DATASETS contains unavailable datasets: {missing}")
            return sorted(CV_DATASETS_OVERRIDE), "manual_override", "manual override"

        candidates = sorted(
            p for p in REPO_DIR.rglob("*.json")
            if "split" in p.name.lower() and "kaggle_test_splits_50ep" not in p.name
        )
        for path in candidates:
            try:
                names = _extract_split0_holdout(json.loads(path.read_text()))
            except Exception:
                continue
            names = sorted(set(names) & available)
            if names:
                return names, "official_split0_holdout", str(path.relative_to(REPO_DIR))

        if not CV_ALLOW_MIXED_FALLBACK:
            raise RuntimeError(
                "No split-0 holdout JSON could be resolved. Set BIOHUB_CV_DATASETS explicitly "
                "or enable BIOHUB_CV_ALLOW_MIXED_FALLBACK."
            )
        return all_stems, "mixed_in_sample", "all paired train datasets; absolute score is optimistic"


    all_train_stems = paired_train_stems()
    test_stems, CV_SPLIT_KIND, CV_SPLIT_SOURCE = resolve_cv_stems(all_train_stems)
    print(f"Paired train datasets: {len(all_train_stems)}")
    print(f"CV split kind: {CV_SPLIT_KIND}")
    print(f"CV split source: {CV_SPLIT_SOURCE}")
    print(f"CV datasets ({len(test_stems)}): {test_stems}")

    if len(test_stems) != 8:
        raise RuntimeError(f"Smoke CV must use exactly 8 fixed datasets, got {len(test_stems)}")
    print("SMOKE MODE: one T4 GPU is used during inference; expected prediction time is roughly 16–30 minutes.")
    print("SMOKE MODE: offline official-spec evaluation usually adds roughly 2–8 minutes.")

    splits_path = REPO_DIR / "official_spec_smoke_cv_split0.json"
    splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

    predict_cmd = [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir", str(TRAIN_DIR),
        "--splits", str(splits_path.name),
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        predict_cmd.append("--use-ilp")
    if SLICE:
        predict_cmd.extend(["--slice", SLICE])

    start_time = time.time()
    print(" ".join(predict_cmd))
    subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
    predict_seconds = time.time() - start_time
    print(f"CV prediction completed in {predict_seconds / 60:.2f} minutes")

    geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
    print(f"Found {len(geffs)} prediction graphs")
    if len(geffs) != len(test_stems):
        found = {path.stem for path in geffs}
        missing = sorted(set(test_stems) - found)
        raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with SUBMISSION_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for geff_path in geffs:
            dataset = geff_path.stem
            seen_datasets.add(dataset)
            graph = graph_from_geff(geff_path)

            nodes_by_id: dict[int, dict[str, object]] = {}
            for row in graph.node_attrs().iter_rows(named=True):
                node_id = int(row["node_id"])
                nodes_by_id[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }

            raw_edges: list[dict[str, object]] = []
            for row in graph.edge_attrs().iter_rows(named=True):
                edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
                raw_edges.append({
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_prob": None if edge_prob is None else float(edge_prob),
                })

            raw_node_count = len(nodes_by_id)
            nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            division_sources: dict[int, int] = {}
            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source_id,
                    "target_id": target_id,
                })
                row_id += 1
                division_sources[source_id] = division_sources.get(source_id, 0) + 1

            node_count = len(nodes_by_id)
            edge_count = len(edges)
            total_nodes += node_count
            total_edges += edge_count
            stats_rows.append({
                "dataset": dataset,
                "raw_nodes": raw_node_count,
                "nodes": node_count,
                "raw_edges": filter_stats["raw_edges"],
                "edges": edge_count,
                "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
                "edge_to_node_ratio": edge_count / max(node_count, 1),
                "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
                **filter_stats,
            })

    expected_datasets = set(test_stems)
    missing_datasets = sorted(expected_datasets - seen_datasets)
    extra_datasets = sorted(seen_datasets - expected_datasets)
    if missing_datasets or extra_datasets:
        raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"

    header = SUBMISSION_PATH.open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

    stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
    stats["predict_minutes_total"] = predict_seconds / 60.0
    stats["experiment_tag"] = EXPERIMENT_TAG
    stats.to_csv(RUN_STATS_PATH, index=False)

    print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
    print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
    print(f"Wrote {RUN_STATS_PATH}")
    display(stats.describe(include="all"))
    display(pd.read_csv(SUBMISSION_PATH, nrows=8))

    from geff import GeffMetadata


    def _graph_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        graph = graph_from_geff(path)
        n = pd.DataFrame(graph.node_attrs().to_dicts())
        e = pd.DataFrame(graph.edge_attrs().to_dicts())
        if e.empty:
            e = pd.DataFrame(columns=["source_id", "target_id"])
        return _normalise_nodes(n), e[["source_id", "target_id"]].copy()


    def _estimated_total_nodes(gt_path: Path) -> float:
        try:
            metadata = GeffMetadata.read(gt_path)
            value = (metadata.extra or {}).get("estimated_number_of_nodes")
            return float(value) if value is not None else float("nan")
        except Exception as exc:
            print("Metadata warning", gt_path.name, type(exc).__name__, str(exc)[:120])
            return float("nan")

    pred_csv = pd.read_csv(SUBMISSION_PATH)
    metric_rows = []
    notes = [
        f"repository={OFFICIAL_SPEC_REPOSITORY}",
        f"commit={OFFICIAL_SPEC_COMMIT}",
        f"document={OFFICIAL_SPEC_DOCUMENT}",
        f"exact_official_source_copy={OFFICIAL_SPEC_EXACT_SOURCE_COPY}",
        f"fingerprint={_SPEC_FINGERPRINT}",
    ]

    for dataset in test_stems:
        part = pred_csv[pred_csv.dataset == dataset]
        pred_nodes = part[part.row_type == "node"][["node_id","t","z","y","x"]].copy()
        pred_edges = part[part.row_type == "edge"][["source_id","target_id"]].copy()
        gt_path = TRAIN_DIR / f"{dataset}.geff"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_nodes, gt_edges = _graph_tables(gt_path)
        er = official_spec_evaluate(
            pred_nodes, pred_edges, gt_nodes, gt_edges,
            scale=VOXEL_SCALE_UM,
            max_distance_um=OFFICIAL_SPEC_MAX_DISTANCE_UM,
        )
        n_total = _estimated_total_nodes(gt_path)
        per = official_spec_per_sample(er, n_total)
        row = {
            "dataset": dataset,
            **er.__dict__,
            "estimated_total_nodes": n_total,
            **per,
        }
        metric_rows.append(row)
        line = (
            f"{dataset}: edge TP/FP/FN={er.edge_tp}/{er.edge_fp}/{er.edge_fn} "
            f"div TP/FP/FN={er.division_tp}/{er.division_fp}/{er.division_fn} "
            f"adj_edge={per['adj_edge_jaccard']:.6f} node_recall={per['node_recall']:.6f}"
        )
        notes.append(line); print(line)

    official_metric_df = pd.DataFrame(metric_rows)
    official_metric_df.to_csv(OFFICIAL_METRIC_ROWS_PATH, index=False)
    official_summary = official_spec_summarise(metric_rows)
    official_summary.update({
        "metric_label": "offline_official_spec_v1",
        "exact_official_source_copy": OFFICIAL_SPEC_EXACT_SOURCE_COPY,
        "official_repository": OFFICIAL_SPEC_REPOSITORY,
        "official_spec_commit": OFFICIAL_SPEC_COMMIT,
        "official_spec_document": OFFICIAL_SPEC_DOCUMENT,
        "offline_evaluator_fingerprint": _SPEC_FINGERPRINT,
        "cv_split_kind": CV_SPLIT_KIND,
        "cv_split_source": CV_SPLIT_SOURCE,
        "datasets": test_stems,
        "max_match_um": OFFICIAL_SPEC_MAX_DISTANCE_UM,
    })
    OFFICIAL_METRIC_SUMMARY_PATH.write_text(json.dumps(official_summary, indent=2, default=str))
    OFFICIAL_METRIC_STDOUT_PATH.write_text("\n".join(notes) + "\n\n" + json.dumps(official_summary, indent=2, default=str))

    print("Wrote", OFFICIAL_METRIC_ROWS_PATH)
    print("Wrote", OFFICIAL_METRIC_SUMMARY_PATH)
    print("Wrote", OFFICIAL_METRIC_STDOUT_PATH)
    print("Offline official-spec summary:")
    print(json.dumps(official_summary, indent=2, default=str))
    cv_status.update({
        'status': 'completed',
        'score': float(official_summary.get('score', float('nan'))),
        'adj_edge_jaccard': float(official_summary.get('adj_edge_jaccard', float('nan'))),
        'edge_jaccard_micro': float(official_summary.get('edge_jaccard_micro', float('nan'))),
        'division_jaccard': float(official_summary.get('division_jaccard', float('nan'))),
        'edge_tp': int(official_summary.get('edge_tp', 0)),
        'edge_fp': int(official_summary.get('edge_fp', 0)),
        'edge_fn': int(official_summary.get('edge_fn', 0)),
        'node_recall': float(official_summary.get('node_recall', float('nan'))),
        'prediction_minutes': float(predict_seconds / 60.0),
        'reference_124_score': 0.879219745066878,
        'delta_vs_124': float(official_summary.get('score', float('nan'))) - 0.879219745066878,
    })
except Exception as exc:
    cv_status.update({
        'status': 'failed_but_submission_preserved',
        'error_type': type(exc).__name__,
        'error': str(exc),
        'traceback_tail': traceback.format_exc().splitlines()[-12:],
    })
    print('LITE_CV_WARNING:', type(exc).__name__, str(exc))
    print('The clean submission.csv remains valid and will still be available for submission.')
finally:
    TEST_DIR = _original_test_dir
    SUBMISSION_PATH = _original_submission_path
    RUN_STATS_PATH = _original_run_stats_path
    EXPERIMENT_TAG = _original_experiment_tag

    if not _SUBMISSION_FILE_SAFE.exists():
        raise FileNotFoundError('submission.csv disappeared during CV diagnostics')
    _submission_sha_after_cv = hashlib.sha256(_SUBMISSION_FILE_SAFE.read_bytes()).hexdigest()
    _submission_rows_after_cv = sum(1 for _ in _SUBMISSION_FILE_SAFE.open('r', encoding='utf-8')) - 1
    cv_status['submission_sha_after_cv'] = _submission_sha_after_cv
    cv_status['submission_rows_after_cv'] = _submission_rows_after_cv
    cv_status['submission_preserved_exactly'] = (
        _submission_sha_after_cv == _SUBMISSION_SHA_BEFORE_CV
        and _submission_rows_after_cv == _SUBMISSION_ROWS_BEFORE_CV
    )
    CV_STATUS_PATH.write_text(json.dumps(cv_status, indent=2, default=str))

if not cv_status['submission_preserved_exactly']:
    raise AssertionError('CV diagnostics altered submission.csv')

print('')
print('＊＊＊＊＊ COMBINED RUN SUMMARY ＊＊＊＊＊')
print('submission.csv ready:', _SUBMISSION_FILE_SAFE)
print('submission rows:', cv_status['submission_rows_after_cv'])
print('submission SHA256:', cv_status['submission_sha_after_cv'])
print('Lite CV status:', cv_status['status'])
if cv_status['status'] == 'completed':
    print('Lite CV official-spec score:', cv_status['score'])
    print('Notebook 124 fixed-8 reference:', cv_status['reference_124_score'])
    print('Delta vs Notebook 124:', f"{cv_status['delta_vs_124']:+.9f}")
    print('Lite CV edge TP/FP/FN:', cv_status['edge_tp'], cv_status['edge_fp'], cv_status['edge_fn'])
    print('Lite CV node recall:', cv_status['node_recall'])
print('CV status file:', CV_STATUS_PATH)
print('SUBMISSION_READY: /kaggle/working/submission.csv')
print('＊＊＊＊＊ END COMBINED RUN SUMMARY ＊＊＊＊＊')


# %% [markdown] papermill={"duration": 0.008755, "end_time": "2026-07-23T01:12:11.406005+00:00", "exception": false, "start_time": "2026-07-23T01:12:11.397250+00:00", "status": "completed"}
# ## Submission output
#
# After the Version completes, inspect the final Lite CV summary. Submit `submission.csv` only when the conservative rescue is activated and produces a positive paired LocalCV delta over the legitimate `0.908` clean baseline.
#
