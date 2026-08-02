import numpy as np
import pandas as pd
import pytest

from biohub_tracker.fixtures import tiny_expected_graph
from biohub_tracker.submission import SUBMISSION_COLUMNS, build_submission, validate_submission
from biohub_tracker.submission.validator import ValidationError


def test_submission_schema_sentinels_ids_and_dtypes(tiny_metadata) -> None:
    table = build_submission([tiny_expected_graph()])
    assert list(table.columns) == SUBMISSION_COLUMNS
    assert np.array_equal(table["id"], np.arange(len(table), dtype=np.int64))
    for column in ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        assert table[column].dtype == np.dtype("int64")
    nodes = table[table.row_type == "node"]
    edges = table[table.row_type == "edge"]
    assert (nodes[["source_id", "target_id"]] == -1).all().all()
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all()
    assert not table.isna().any().any()
    validate_submission(
        table,
        expected_dataset_names={"tiny"},
        metadata_by_dataset={"tiny": tiny_metadata},
    )


def test_dataset_name_must_omit_zarr() -> None:
    graph = tiny_expected_graph("tiny.zarr")
    table = build_submission([graph])
    with pytest.raises(ValidationError, match="omit"):
        validate_submission(table, expected_dataset_names={"tiny.zarr"})


def test_row_ids_must_be_consecutive() -> None:
    table = build_submission([tiny_expected_graph()])
    table.loc[1, "id"] = 100
    with pytest.raises(ValidationError, match="consecutive"):
        validate_submission(table, expected_dataset_names={"tiny"})


def test_no_nan_values() -> None:
    table = build_submission([tiny_expected_graph()])
    table.loc[0, "dataset"] = None
    with pytest.raises(ValidationError, match="missing"):
        validate_submission(table, expected_dataset_names={"tiny"})


def test_csv_round_trip_preserves_schema(tmp_path, tiny_metadata) -> None:
    table = build_submission([tiny_expected_graph()])
    destination = tmp_path / "submission.csv"
    table.to_csv(destination, index=False)
    loaded = pd.read_csv(destination)
    assert list(loaded.columns) == SUBMISSION_COLUMNS
    assert {column: str(dtype) for column, dtype in loaded.dtypes.items()} == {
        column: str(dtype) for column, dtype in table.dtypes.items()
    }
    validate_submission(
        loaded,
        expected_dataset_names={"tiny"},
        metadata_by_dataset={"tiny": tiny_metadata},
    )


def test_empty_detection_result_has_exact_typed_schema() -> None:
    table = build_submission([])
    assert list(table.columns) == SUBMISSION_COLUMNS
    assert table.empty
    numeric = [column for column in SUBMISSION_COLUMNS if column not in {"dataset", "row_type"}]
    assert all(str(table[column].dtype) == "int64" for column in numeric)
