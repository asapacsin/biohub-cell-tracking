from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_tracker.models import FileTableInspection

TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}


def discover_annotation_files(root: str | Path) -> list[Path]:
    competition_root = Path(root)
    candidates: list[Path] = []
    for path in competition_root.rglob("*") if competition_root.exists() else []:
        if not path.is_file() or path.name.lower() == "sample_submission.csv":
            continue
        if any(parent.suffix.lower() == ".zarr" for parent in path.parents):
            continue
        lowered = "/".join(part.lower() for part in path.parts)
        if path.suffix.lower() in TABLE_SUFFIXES and any(
            marker in lowered for marker in ("annot", "track", "label", "train")
        ):
            candidates.append(path)
    return sorted(candidates)


def inspect_table(path: str | Path, sample_rows: int = 5) -> FileTableInspection:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(source)
    elif suffix == ".tsv":
        table = pd.read_csv(source, sep="\t")
    elif suffix == ".parquet":
        table = pd.read_parquet(source)
    elif suffix in {".json", ".jsonl"}:
        table = pd.read_json(source, lines=suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported table format: {source}")
    rows: list[dict[str, Any]] = json.loads(
        table.head(sample_rows).to_json(orient="records", date_format="iso")
    )
    return FileTableInspection(
        path=str(source),
        format=suffix.removeprefix("."),
        columns=tuple(str(column) for column in table.columns),
        dtypes={str(column): str(dtype) for column, dtype in table.dtypes.items()},
        row_count=len(table),
        sample_rows=tuple(rows),
    )
