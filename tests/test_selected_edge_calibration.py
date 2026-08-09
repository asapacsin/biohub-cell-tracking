from __future__ import annotations

import math

import pandas as pd
import pytest

from biohub_pipeline.selected_edge_calibration import (
    correctness_auroc,
    evaluate_calibration,
    prepare_scored_associations,
)


def test_predeclared_budget_rejects_weak_error_capture() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["demo"] * 20,
            "t": list(range(20)),
            "correct": [False, *([True] * 18), False],
            "selected_edge_score": [0.1, 0.2, 0.3, 0.4, 0.5, *list(range(6, 21))],
        }
    )

    metrics, selected = evaluate_calibration(frame, budget_fraction=0.05)

    assert len(selected) == 1
    assert metrics["captured_errors"] == 1
    assert metrics["error_capture"] == 0.5
    assert metrics["correctness_auroc"] < 0.8
    assert metrics["classification"] == "D. HYPOTHESIS REJECTED"


def test_predeclared_budget_supports_strong_confidence_separation() -> None:
    frame = pd.DataFrame(
        {
            "correct": [False] * 5 + [True] * 95,
            "selected_edge_score": list(range(100)),
        }
    )

    metrics, selected = evaluate_calibration(frame)

    assert len(selected) == 5
    assert metrics["captured_errors"] == 5
    assert metrics["error_capture"] == 1.0
    assert metrics["intervention_precision"] == 1.0
    assert metrics["correctness_auroc"] == 1.0
    assert metrics["classification"] == "B. PROMISING - NEEDS INDEPENDENT VALIDATION"


def test_correctness_auroc_uses_average_ranks_for_ties() -> None:
    assert correctness_auroc([False, True, False, True], [0.1, 0.2, 0.2, 0.3]) == 0.875
    assert math.isnan(correctness_auroc([True, True], [0.1, 0.2]))


def test_prepare_scored_associations_parses_strings_and_drops_missing_scores() -> None:
    frame = pd.DataFrame(
        {
            "correct": ["True", "false", "True"],
            "selected_edge_score": [0.8, 0.2, None],
        }
    )

    result = prepare_scored_associations(frame)

    assert result["correct"].tolist() == [False, True]
    assert result["selected_edge_score"].tolist() == [0.2, 0.8]


def test_prepare_scored_associations_rejects_ambiguous_boolean_values() -> None:
    frame = pd.DataFrame({"correct": ["yes"], "selected_edge_score": [0.5]})

    with pytest.raises(ValueError, match="non-boolean"):
        prepare_scored_associations(frame)
