from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from biohub_tracker.association.candidates import SparseCandidateGraph

FEATURE_NAMES = (
    "is_division",
    "delta_t",
    "distance_um",
    "abs_dz_um",
    "abs_dy_um",
    "abs_dx_um",
    "confidence_mean",
    "appearance_similarity",
    "intensity_log_ratio",
    "volume_log_ratio",
    "temporal_density_log_ratio",
    "daughter_separation_um",
    "midpoint_distance_um",
    "parent_child_distance_mean_um",
)


def candidate_feature_matrix(graph: SparseCandidateGraph) -> NDArray[np.float64]:
    """Return one stable feature row per edge followed by one row per division event."""
    rows: list[list[float]] = []
    for edge in graph.edges:
        rows.append(
            [
                0.0,
                float(edge.delta_t),
                edge.distance_um,
                abs(edge.displacement_zyx_um[0]),
                abs(edge.displacement_zyx_um[1]),
                abs(edge.displacement_zyx_um[2]),
                edge.confidence_mean,
                edge.appearance_similarity or 0.0,
                edge.intensity_log_ratio or 0.0,
                edge.volume_log_ratio or 0.0,
                edge.temporal_density_log_ratio,
                0.0,
                0.0,
                0.0,
            ]
        )
    for division in graph.divisions:
        rows.append(
            [
                1.0,
                1.0,
                division.parent_child_distance_mean_um,
                0.0,
                0.0,
                0.0,
                division.confidence_mean,
                0.0,
                0.0,
                0.0,
                0.0,
                division.daughter_separation_um,
                division.midpoint_distance_um,
                division.parent_child_distance_mean_um,
            ]
        )
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LinearAssociationArtifact:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        lengths = {
            len(self.feature_names),
            len(self.means),
            len(self.scales),
            len(self.coefficients),
        }
        if lengths != {len(FEATURE_NAMES)}:
            raise ValueError("Association artifact has incompatible feature dimensions")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("Association artifact feature order does not match this runtime")

    def predict_logits(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return ((features - means) / scales) @ coefficients + self.intercept

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LinearAssociationArtifact:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(str(value) for value in raw["feature_names"]),
            means=tuple(float(value) for value in raw["means"]),
            scales=tuple(float(value) for value in raw["scales"]),
            coefficients=tuple(float(value) for value in raw["coefficients"]),
            intercept=float(raw["intercept"]),
        )


@dataclass(frozen=True, slots=True)
class LinearAssociationScorer:
    artifact: LinearAssociationArtifact

    @classmethod
    def load(cls, path: str | Path) -> LinearAssociationScorer:
        return cls(LinearAssociationArtifact.load(path))

    def score(self, graph: SparseCandidateGraph) -> SparseCandidateGraph:
        logits = self.artifact.predict_logits(candidate_feature_matrix(graph))
        edge_count = len(graph.edges)
        for edge, score in zip(graph.edges, logits[:edge_count], strict=True):
            edge.score = float(score)
        for division, score in zip(graph.divisions, logits[edge_count:], strict=True):
            division.score = float(score)
        return graph


def fit_linear_association_model(
    features: NDArray[np.float64],
    labels: NDArray[np.int8],
    *,
    l2_regularization: float = 1e-3,
    max_iterations: int = 500,
) -> LinearAssociationArtifact:
    """Fit a class-balanced logistic event scorer and return a portable JSON artifact."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
    if labels.shape != (features.shape[0],) or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must contain one binary value per feature row")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("association training requires both positive and negative candidates")
    if l2_regularization < 0 or max_iterations < 1:
        raise ValueError("regularization must be non-negative and iterations positive")

    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-8] = 1.0
    standardized = (features - means) / scales
    sample_weights = np.where(
        labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives)
    )

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
        logits = standardized @ coefficients + intercept
        losses = np.logaddexp(0.0, logits) - labels * logits
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        residual = sample_weights * (probabilities - labels)
        value = float(np.mean(sample_weights * losses))
        value += 0.5 * l2_regularization * float(coefficients @ coefficients)
        gradient = np.empty_like(parameters)
        gradient[:-1] = standardized.T @ residual / len(labels) + l2_regularization * coefficients
        gradient[-1] = residual.mean()
        return value, gradient

    result = minimize(
        objective,
        np.zeros(features.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"Association optimization failed: {result.message}")
    return LinearAssociationArtifact(
        feature_names=FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in result.x[:-1]),
        intercept=float(result.x[-1]),
    )
