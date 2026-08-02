from __future__ import annotations

from pathlib import Path

from biohub_tracker.config import ProjectConfig
from biohub_tracker.models import PredictionGraph


def run_prediction_pipeline(
    competition_root: str | Path,
    config: ProjectConfig,
) -> list[PredictionGraph]:
    del competition_root, config
    raise NotImplementedError(
        "Prediction is intentionally unavailable in Milestones 0-1. "
        "Inspect official data successfully before implementing detection and tracking."
    )

