from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from biohub_tracker.annotation_reader import discover_annotation_files, inspect_table
from biohub_tracker.models import CompetitionLayout
from biohub_tracker.zarr_reader import VolumeDatasetReader, discover_zarr_stores


def discover_competition_layout(root: str | Path) -> CompetitionLayout:
    competition_root = Path(root)
    sample_candidates = (
        sorted(competition_root.rglob("sample_submission.csv")) if competition_root.exists() else []
    )
    if len(sample_candidates) > 1:
        raise ValueError(f"Multiple sample_submission.csv files found: {sample_candidates}")
    test_stores = discover_zarr_stores(competition_root, "test")
    train_stores = discover_zarr_stores(competition_root, "train")
    annotations = discover_annotation_files(competition_root)
    excluded = {path.resolve() for path in sample_candidates + annotations}
    other_files = tuple(
        str(path)
        for path in sorted(competition_root.rglob("*"))
        if path.is_file()
        and path.resolve() not in excluded
        and not any(parent.suffix.lower() == ".zarr" for parent in path.parents)
    ) if competition_root.exists() else ()
    return CompetitionLayout(
        root=str(competition_root),
        sample_submission=str(sample_candidates[0]) if sample_candidates else None,
        test_stores=tuple(str(path) for path in test_stores),
        train_stores=tuple(str(path) for path in train_stores),
        annotation_files=tuple(str(path) for path in annotations),
        other_files=other_files,
    )


def inspect_competition(root: str | Path) -> dict[str, Any]:
    layout = discover_competition_layout(root)
    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout": asdict(layout),
        "missing_authoritative_inputs": [],
        "sample_submission": None,
        "test_datasets": [],
        "training_annotations": [],
        "errors": [],
    }
    missing = report["missing_authoritative_inputs"]
    if layout.sample_submission is None:
        missing.append("sample_submission.csv")
    if not layout.test_stores:
        missing.append("test/*.zarr")
    if not layout.train_stores:
        missing.append("training image .zarr stores")
    if not layout.annotation_files:
        missing.append("training tracking annotations")
    if layout.sample_submission:
        try:
            report["sample_submission"] = asdict(inspect_table(layout.sample_submission))
        except Exception as exc:  # inspection should report every independent failure
            report["errors"].append(f"sample submission: {exc}")
    reader = VolumeDatasetReader(root)
    for dataset in reader.dataset_names():
        try:
            report["test_datasets"].append(asdict(reader.metadata(dataset)))
        except Exception as exc:
            report["errors"].append(f"test dataset {dataset}: {exc}")
    for annotation in layout.annotation_files:
        try:
            report["training_annotations"].append(asdict(inspect_table(annotation)))
        except Exception as exc:
            report["errors"].append(f"annotation {annotation}: {exc}")
    return report


def write_inspection_report(report: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def format_inspection_report(report: dict[str, Any]) -> str:
    layout = report["layout"]
    lines = [
        f"Competition root: {layout['root']}",
        f"Sample submission: {layout['sample_submission'] or 'NOT FOUND'}",
        f"Test Zarr stores: {len(layout['test_stores'])}",
        f"Training Zarr stores: {len(layout['train_stores'])}",
        f"Annotation files: {len(layout['annotation_files'])}",
    ]
    if report["missing_authoritative_inputs"]:
        lines.append("Missing: " + ", ".join(report["missing_authoritative_inputs"]))
    for dataset in report["test_datasets"]:
        lines.extend(
            [
                f"Dataset {dataset['name']}:",
                f"  array={dataset['array_path']} shape={tuple(dataset['shape'])}",
                f"  axes={tuple(dataset['axes'])} dtype={dataset['dtype']}",
                f"  time_points={dataset['time_points']} channels={dataset['channel_count']}",
                f"  voxel_spacing_zyx={tuple(dataset['voxel_spacing_zyx'])}",
                f"  multiscale_levels={tuple(dataset['multiscale_levels'])}",
            ]
        )
    if report["sample_submission"]:
        sample = report["sample_submission"]
        lines.append(f"Sample columns: {tuple(sample['columns'])}")
        lines.append(f"Sample dtypes: {sample['dtypes']}")
        lines.append(f"Sample rows: {sample['sample_rows']}")
    if report["errors"]:
        lines.append("Inspection errors:")
        lines.extend(f"  - {message}" for message in report["errors"])
    return "\n".join(lines)

