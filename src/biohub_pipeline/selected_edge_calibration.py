"""Evaluate whether selected-edge confidence can target association errors."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_BUDGET_FRACTION = 0.05
DEFAULT_MIN_ERROR_CAPTURE = 0.40
DEFAULT_MIN_CORRECTNESS_AUROC = 0.80

TIE_BREAK_COLUMNS = (
    "dataset",
    "t",
    "gt_source_id",
    "gt_target_id",
)


def _as_boolean(series: pd.Series) -> pd.Series:
    """Return a strict boolean series without treating non-empty strings as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        values = sorted(normalized[invalid].unique())
        raise ValueError(f"correctness column contains non-boolean values: {values}")
    return normalized.isin({"true", "1"})


def prepare_scored_associations(
    frame: pd.DataFrame,
    *,
    score_column: str = "selected_edge_score",
    correct_column: str = "correct",
) -> pd.DataFrame:
    """Validate, normalize, and deterministically order scored associations."""
    missing = {score_column, correct_column} - set(frame.columns)
    if missing:
        raise ValueError(f"diagnostic is missing required columns: {sorted(missing)}")

    result = frame.loc[frame[score_column].notna()].copy()
    result[score_column] = pd.to_numeric(result[score_column], errors="raise")
    result = result[np.isfinite(result[score_column])].copy()
    if result.empty:
        raise ValueError("diagnostic contains no finite selected-edge scores")
    result[correct_column] = _as_boolean(result[correct_column])

    tie_breaks = [column for column in TIE_BREAK_COLUMNS if column in result.columns]
    return result.sort_values(
        [score_column, *tie_breaks],
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)


def correctness_auroc(correct: Iterable[bool], confidence: Iterable[float]) -> float:
    """Compute tie-aware AUROC with correctness as the positive class."""
    labels = np.asarray(list(correct), dtype=bool)
    scores = np.asarray(list(confidence), dtype=float)
    if len(labels) != len(scores):
        raise ValueError("correctness and confidence arrays must have equal length")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_calibration(
    frame: pd.DataFrame,
    *,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    min_error_capture: float = DEFAULT_MIN_ERROR_CAPTURE,
    min_correctness_auroc: float = DEFAULT_MIN_CORRECTNESS_AUROC,
    score_column: str = "selected_edge_score",
    correct_column: str = "correct",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a predeclared low-confidence intervention budget.

    Support requires both the minimum error-capture rate at the intervention budget
    and the minimum correctness AUROC. This is deliberately stricter than showing
    enrichment over random selection: the score must identify enough of the remaining
    errors to justify confidence-only calibration as the next production lever.
    """
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if not 0 <= min_error_capture <= 1:
        raise ValueError("min_error_capture must be in [0, 1]")
    if not 0 <= min_correctness_auroc <= 1:
        raise ValueError("min_correctness_auroc must be in [0, 1]")

    scored = prepare_scored_associations(
        frame,
        score_column=score_column,
        correct_column=correct_column,
    )
    budget_count = max(1, math.ceil(len(scored) * budget_fraction))
    selected = scored.iloc[:budget_count].copy()
    selected["selected_for_intervention"] = True

    errors = ~scored[correct_column]
    selected_errors = ~selected[correct_column]
    error_count = int(errors.sum())
    captured_count = int(selected_errors.sum())
    error_capture = captured_count / error_count if error_count else 0.0
    precision = captured_count / budget_count
    prevalence = error_count / len(scored)
    precision_lift = precision / prevalence if prevalence else math.nan
    auroc = correctness_auroc(scored[correct_column], scored[score_column])
    supported = bool(
        error_capture >= min_error_capture
        and np.isfinite(auroc)
        and auroc >= min_correctness_auroc
    )
    classification = (
        "B. PROMISING - NEEDS INDEPENDENT VALIDATION"
        if supported
        else "D. HYPOTHESIS REJECTED"
    )

    metrics: dict[str, Any] = {
        "classification": classification,
        "hypothesis_supported": supported,
        "scored_associations": len(scored),
        "scored_errors": error_count,
        "error_prevalence": prevalence,
        "budget_fraction": budget_fraction,
        "budget_count": budget_count,
        "confidence_threshold": float(selected[score_column].max()),
        "captured_errors": captured_count,
        "error_capture": error_capture,
        "intervention_precision": precision,
        "precision_lift_vs_random": precision_lift,
        "correctness_auroc": auroc,
        "min_error_capture": min_error_capture,
        "min_correctness_auroc": min_correctness_auroc,
        "recommended_next_action": (
            "Keep the production recipe unchanged and target edge-ranking quality rather "
            "than confidence-only calibration."
        ),
    }
    return metrics, selected


def summarize_groups(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    score_column: str = "selected_edge_score",
    correct_column: str = "correct",
) -> pd.DataFrame:
    """Evaluate the same budget independently within requested diagnostic groups."""
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for group, part in frame.groupby(column, observed=True, sort=True):
            metrics, _ = evaluate_calibration(
                part,
                budget_fraction=budget_fraction,
                score_column=score_column,
                correct_column=correct_column,
            )
            rows.append(
                {
                    "group_type": column,
                    "group": str(group),
                    **{
                        key: metrics[key]
                        for key in (
                            "scored_associations",
                            "scored_errors",
                            "error_prevalence",
                            "budget_count",
                            "captured_errors",
                            "error_capture",
                            "intervention_precision",
                            "precision_lift_vs_random",
                            "correctness_auroc",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)
