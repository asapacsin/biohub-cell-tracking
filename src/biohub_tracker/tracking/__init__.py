"""Tracking: greedy nearest-neighbour baseline and optional Hungarian linker."""

from biohub_tracker.tracking.data_types import Detection, TrackObservation
from biohub_tracker.tracking.division import DivisionConfig, apply_divisions
from biohub_tracker.tracking.linker import TrackingConfig, link_consecutive_nodes
from biohub_tracker.tracking.nearest_neighbor import link_consecutive_frames
from biohub_tracker.tracking.tracker import track_video_detections

__all__ = [
    "Detection",
    "DivisionConfig",
    "TrackObservation",
    "TrackingConfig",
    "apply_divisions",
    "link_consecutive_frames",
    "link_consecutive_nodes",
    "track_video_detections",
]
