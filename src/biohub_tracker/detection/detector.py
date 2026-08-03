from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.detection.blob import BlobDetectionConfig, detect_frame
from biohub_tracker.tracking.data_types import Detection


def detect_frame_as_detections(
    frame_zyx: NDArray[np.generic],
    *,
    video_id: str,
    frame_index: int,
    voxel_spacing_zyx: tuple[float, float, float],
    config: BlobDetectionConfig,
) -> list[Detection]:
    """Wrap the existing blob detector into baseline ``Detection`` objects."""
    candidates = detect_frame(
        frame_zyx,
        dataset=video_id,
        t=frame_index,
        voxel_spacing_zyx=voxel_spacing_zyx,
        config=config,
    )
    return [
        Detection(
            video_id=video_id,
            frame_index=frame_index,
            detection_id=index,
            centroid_z=candidate.z,
            centroid_y=candidate.y,
            centroid_x=candidate.x,
            confidence=candidate.score,
            mask=None,
            bbox=None,
        )
        for index, candidate in enumerate(candidates)
    ]
