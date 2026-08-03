from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Detection:
    """Per-frame detection. ``detection_id`` restarts at 0 each frame."""

    video_id: str
    frame_index: int
    detection_id: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    confidence: float | None
    mask: NDArray[np.generic] | None = None
    bbox: tuple[int, int, int, int, int, int] | None = None

    @property
    def centroid_zyx(self) -> tuple[float, float, float]:
        return self.centroid_z, self.centroid_y, self.centroid_x


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Tracked observation. ``cell_id`` persists across frames.

    ``parent_id`` is set for division children (competition lineage via two outgoing edges).
    ``link_status`` values: initial | matched | new | division_child.
    """

    video_id: str
    frame_index: int
    detection_id: int
    cell_id: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    parent_id: int | None
    link_distance: float | None = None
    link_status: str = "initial"
    node_id: int | None = None
    division_score: float | None = None

    @property
    def centroid_zyx(self) -> tuple[float, float, float]:
        return self.centroid_z, self.centroid_y, self.centroid_x
