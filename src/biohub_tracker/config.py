from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    node_id_start: int = 1
    sort_rows: bool = True
    strict_validation: bool = True


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    raw: dict[str, Any]
    submission: SubmissionConfig


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    submission_raw = raw.get("submission", {})
    if not isinstance(submission_raw, dict):
        raise ValueError("submission configuration must be a mapping")
    submission = SubmissionConfig(
        node_id_start=int(submission_raw.get("node_id_start", 1)),
        sort_rows=bool(submission_raw.get("sort_rows", True)),
        strict_validation=bool(submission_raw.get("strict_validation", True)),
    )
    if submission.node_id_start < 0:
        raise ValueError("submission.node_id_start must be non-negative")
    return ProjectConfig(raw=raw, submission=submission)
