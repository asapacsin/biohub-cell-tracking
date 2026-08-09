from __future__ import annotations

import numpy as np
import pandas as pd

from biohub_pipeline.association_density import (
    add_bins,
    assign_density_bins,
    assign_margin_bins,
    classify_hypothesis,
)


def test_density_bins_use_supplied_quartile_thresholds() -> None:
    bins, thresholds = assign_density_bins([1, 2, 3, 4, np.nan], thresholds=(1, 2, 3))
    assert thresholds == (1, 2, 3)
    assert bins.tolist() == ["low", "medium_low", "medium_high", "high", "unavailable"]


def test_margin_bins_have_stable_boundary_semantics() -> None:
    bins = assign_margin_bins([0.0, 0.009, 0.01, 0.049, 0.05, 0.149, 0.15, np.nan])
    assert bins.tolist() == [
        "<0.01",
        "<0.01",
        "0.01-0.05",
        "0.01-0.05",
        "0.05-0.15",
        "0.05-0.15",
        ">=0.15",
        "unavailable",
    ]


def test_add_bins_is_reproducible() -> None:
    frame = pd.DataFrame({"local_density_pred": range(8), "candidate_margin": [np.nan] * 8})
    first, thresholds = add_bins(frame)
    second, again = add_bins(frame)
    assert thresholds == again
    assert first["density_bin"].tolist() == second["density_bin"].tolist()


def test_classifier_cannot_verify_without_candidate_margins() -> None:
    # High density has errors, low density does not; association errors dominate.
    frame = pd.DataFrame(
        {
            "correct": [True] * 10 + [False] * 10,
            "density_bin": pd.Categorical(["low"] * 10 + ["high"] * 10),
            "candidate_margin": [np.nan] * 20,
            "outcome": ["correct"] * 10 + ["wrong_association"] * 10,
        }
    )
    result = classify_hypothesis(frame)
    assert result["conclusion"] == "B PARTIALLY VERIFIED"
    assert result["density_signal"] is True
    assert result["margin_available"] is False
