from __future__ import annotations

import numpy as np

from biohub_tracker.detection import BlobDetectionConfig, detect_frame


def _place_gaussian_blob(
    frame: np.ndarray, center_zyx: tuple[int, int, int], amplitude: float = 1.0
) -> None:
    zz, yy, xx = np.indices(frame.shape)
    z0, y0, x0 = center_zyx
    blob = np.exp(-(((zz - z0) / 1.2) ** 2 + ((yy - y0) / 2.5) ** 2 + ((xx - x0) / 2.5) ** 2))
    frame += (amplitude * blob).astype(frame.dtype)


def test_blob_detector_finds_bright_spots() -> None:
    frame = np.zeros((8, 32, 32), dtype=np.float32)
    _place_gaussian_blob(frame, (2, 10, 10), amplitude=1000)
    _place_gaussian_blob(frame, (5, 20, 22), amplitude=900)
    frame_u16 = np.clip(frame, 0, 65535).astype(np.uint16)
    config = BlobDetectionConfig(
        lower_percentile=0.0,
        upper_percentile=100.0,
        gaussian_sigma_um=0.5,
        threshold=0.15,
        minimum_separation_um=2.0,
    )
    detections = detect_frame(
        frame_u16,
        dataset="tiny",
        t=0,
        voxel_spacing_zyx=(1.625, 0.40625, 0.40625),
        config=config,
    )
    centers = {(int(item.z), int(item.y), int(item.x)) for item in detections}
    assert any(abs(z - 2) <= 1 and abs(y - 10) <= 2 and abs(x - 10) <= 2 for z, y, x in centers)
    assert any(abs(z - 5) <= 1 and abs(y - 20) <= 2 and abs(x - 22) <= 2 for z, y, x in centers)


def test_blob_detector_returns_empty_on_blank_frame() -> None:
    frame = np.full((4, 16, 16), 100, dtype=np.uint16)
    detections = detect_frame(
        frame,
        dataset="tiny",
        t=1,
        voxel_spacing_zyx=(1.625, 0.40625, 0.40625),
        config=BlobDetectionConfig(threshold=0.9),
    )
    assert detections == []
