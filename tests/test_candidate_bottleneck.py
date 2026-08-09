from __future__ import annotations

import pandas as pd

from biohub_pipeline.candidate_bottleneck import classify, summarize


def _rows(causes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["6bba_demo"] * len(causes),
            "dataset_family": ["6bba"] * len(causes),
            "cause": causes,
            "source_detected": [cause != "detection_miss" for cause in causes],
            "target_detected": [cause != "detection_miss" for cause in causes],
            "candidate_present": [
                cause not in {"detection_miss", "candidate_threshold"} for cause in causes
            ],
            "target_rank": [2 if cause == "scorer_ranking" else 1 for cause in causes],
            "ilp_selected": [
                cause not in {"detection_miss", "candidate_threshold", "ilp_global"}
                for cause in causes
            ],
            "final_correct": [cause == "correct" for cause in causes],
            "local_density_pred": list(range(len(causes))),
            "competitor_margin": [0.1] * len(causes),
        }
    )


def test_predeclared_classifier_verifies_dominant_ranking() -> None:
    frame = _rows(
        ["correct"] * 60
        + ["scorer_ranking"] * 30
        + ["candidate_threshold"] * 5
        + ["ilp_global"] * 5
    )
    summary, _ = summarize(
        frame,
        [
            {
                "dataset": "6bba_demo",
                "detected_node_recall": 1.0,
                "final_node_recall": 1.0,
            }
        ],
    )

    result = classify(summary)

    assert result["classification"] == "C. BOTTLENECK VERIFIED"
    assert result["ranking_share"] == 0.75
    assert "Retrain the edge scorer" in result["recommended_next_action"]


def test_predeclared_classifier_rejects_threshold_dominance() -> None:
    frame = _rows(
        ["correct"] * 60
        + ["scorer_ranking"] * 10
        + ["candidate_threshold"] * 25
        + ["ilp_global"] * 5
    )
    summary, _ = summarize(
        frame,
        [
            {
                "dataset": "6bba_demo",
                "detected_node_recall": 1.0,
                "final_node_recall": 1.0,
            }
        ],
    )

    result = classify(summary)

    assert result["classification"] == "D. HYPOTHESIS REJECTED"
    assert result["mechanism_shares"]["candidate_threshold"] == 0.625
    assert "pre-ILP edge gate" in result["recommended_next_action"]
