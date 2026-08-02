from pathlib import Path

import pandas as pd

from biohub_tracker.fixtures import tiny_expected_graph
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

