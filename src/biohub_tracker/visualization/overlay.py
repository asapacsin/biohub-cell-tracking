from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from biohub_tracker.tracking.data_types import TrackObservation


def save_tracking_overlay(
    frame_zyx: NDArray[np.generic],
    observations: list[TrackObservation],
    previous_by_cell: dict[int, TrackObservation],
    output_path: str | Path,
) -> Path:
    """Save a 2D overlay PNG with cell IDs and matched-link lines.

    Visualization method: maximum-intensity projection over Z.
    """
    from PIL import Image, ImageDraw

    mip = np.max(np.asarray(frame_zyx, dtype=np.float32), axis=0)
    if mip.max() > mip.min():
        norm = (mip - mip.min()) / (mip.max() - mip.min())
    else:
        norm = np.zeros_like(mip)
    rgb = np.stack([norm, norm, norm], axis=-1)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for obs in observations:
        y, x = float(obs.centroid_y), float(obs.centroid_x)
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0))
        draw.text((x + 4, y - 4), str(obs.cell_id), fill=(255, 220, 0))
        prev = previous_by_cell.get(obs.cell_id)
        if prev is not None and obs.link_status == "matched":
            draw.line(
                (float(prev.centroid_x), float(prev.centroid_y), x, y),
                fill=(0, 180, 255),
                width=1,
            )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
