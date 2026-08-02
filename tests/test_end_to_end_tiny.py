from pathlib import Path

import pandas as pd

from biohub_tracker.fixtures import generate_tiny_competition, tiny_expected_graph
from biohub_tracker.inspection import inspect_competition
from biohub_tracker.submission import build_submission, validate_submission, write_submission


def test_tiny_graph_to_validated_csv(tmp_path: Path, tiny_metadata) -> None:
    graph = tiny_expected_graph()
    table = build_submission([graph])
    validate_submission(
        table,
        expected_dataset_names={"tiny"},
        metadata_by_dataset={"tiny": tiny_metadata},
    )
    path = write_submission(table, tmp_path / "submission.csv")
    reloaded = pd.read_csv(path)
    validate_submission(
        reloaded,
        expected_dataset_names={"tiny"},
        metadata_by_dataset={"tiny": tiny_metadata},
    )
    assert len(reloaded) == 12


def test_tiny_competition_passes_inspection_and_strict_validation(tmp_path: Path) -> None:
    competition_root = generate_tiny_competition(tmp_path / "competition")
    report = inspect_competition(competition_root)

    assert report["missing_authoritative_inputs"] == []
    assert report["errors"] == []
    assert [dataset["name"] for dataset in report["test_datasets"]] == ["tiny"]
    assert [dataset["name"] for dataset in report["training_datasets"]] == ["tiny_training"]
    assert len(report["training_annotations"]) == 1

    submission = pd.read_csv(competition_root / "sample_submission.csv")
    validate_submission(submission, competition_root)
