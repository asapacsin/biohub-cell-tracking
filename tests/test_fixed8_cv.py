from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    PER_DATASET_COLUMNS,
    aggregate_metric_rows,
    evaluate_tables,
    find_fixed8_prediction_geffs,
    validate_fixed8_inputs,
    write_metric_outputs,
)

EXPECTED_FIXED8 = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_05b6850b",
    "6bba_05db0fb1",
    "6bba_969618f6",
    "6bba_fc83837d",
]


def _make_fixed8_inputs(root: Path) -> None:
    root.mkdir()
    for dataset in FIXED8_DATASETS:
        (root / f"{dataset}.zarr").mkdir()
        (root / f"{dataset}.geff").mkdir()


def _perfect_row(dataset: str) -> dict[str, object]:
    nodes = pd.DataFrame(
        [
            {"node_id": 1, "t": 0, "z": 1, "y": 1, "x": 1},
            {"node_id": 2, "t": 1, "z": 1, "y": 2, "x": 1},
        ]
    )
    edges = pd.DataFrame([{"source_id": 1, "target_id": 2}])
    return evaluate_tables(dataset, nodes, edges, nodes, edges, estimated_total_nodes=2)


def test_fixed8_dataset_list_is_exact() -> None:
    assert FIXED8_DATASETS == EXPECTED_FIXED8


def test_only_fixed8_inputs_are_selected_and_extras_are_ignored(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _make_fixed8_inputs(train)
    (train / "unrelated_extra.zarr").mkdir()
    (train / "unrelated_extra.geff").mkdir()

    selected = validate_fixed8_inputs(train)

    assert [dataset for dataset, _, _ in selected] == EXPECTED_FIXED8
    assert all(
        "unrelated_extra" not in str(path)
        for _, zarr, geff in selected
        for path in (zarr, geff)
    )


def test_missing_required_zarr_is_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _make_fixed8_inputs(train)
    missing = train / f"{FIXED8_DATASETS[3]}.zarr"
    missing.rmdir()

    with pytest.raises(FileNotFoundError, match=r"missing required \.zarr stores") as exc:
        validate_fixed8_inputs(train)
    assert str(missing) in str(exc.value)


def test_missing_required_geff_is_rejected(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _make_fixed8_inputs(train)
    missing = train / f"{FIXED8_DATASETS[5]}.geff"
    missing.rmdir()

    with pytest.raises(FileNotFoundError, match=r"missing required \.geff ground truth") as exc:
        validate_fixed8_inputs(train)
    assert str(missing) in str(exc.value)


def test_prediction_discovery_ignores_unrelated_datasets(tmp_path: Path) -> None:
    prediction_root = tmp_path / "repo" / "predictions"
    for dataset in [*FIXED8_DATASETS, "unrelated_extra"]:
        graph = prediction_root / dataset / "unet_transformer" / "split_0" / f"{dataset}.geff"
        graph.mkdir(parents=True)

    selected = find_fixed8_prediction_geffs(tmp_path / "repo", "unet_transformer")

    assert [path.stem for path in selected] == EXPECTED_FIXED8


def test_metric_aggregation_reuses_official_spec_lite() -> None:
    rows = [_perfect_row(dataset) for dataset in FIXED8_DATASETS]

    summary = aggregate_metric_rows(rows)

    assert summary["datasets"] == EXPECTED_FIXED8
    assert summary["edge_tp"] == 8
    assert summary["edge_fp"] == 0
    assert summary["edge_fn"] == 0
    assert summary["adj_edge_jaccard"] == 1.0
    assert summary["node_recall"] == 1.0


def test_output_schemas(tmp_path: Path) -> None:
    rows = [_perfect_row(dataset) for dataset in FIXED8_DATASETS]
    summary = aggregate_metric_rows(rows)
    summary.update(
        {
            "detection_threshold": 0.96875,
            "checkpoint_path": "/support/weights/checkpoint.pth",
            "runtime_seconds": 1.25,
        }
    )
    manifest = {"schema_version": 1, "datasets": EXPECTED_FIXED8}

    write_metric_outputs(tmp_path, rows, summary, manifest)

    assert list(pd.read_csv(tmp_path / "per_dataset.csv").columns) == PER_DATASET_COLUMNS
    written_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["datasets"] == EXPECTED_FIXED8
    assert written_summary["checkpoint_path"].endswith("checkpoint.pth")
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 1
