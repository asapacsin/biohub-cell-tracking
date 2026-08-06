"""Read-only validation of the external V106 competition and support artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from biohub_pipeline.config import PipelineConfig

REQUIRED_RUNTIME_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "yaml",
    "torch",
    "tracksdata",
    "zarr",
    "pyscipopt",
    "geff",
    "ilpy",
    "polars",
    "blosc2",
)


def validation_report(
    config: PipelineConfig,
    data_dir: Path | None,
    weights_dir: Path | None,
    support_dir: Path | None,
) -> dict[str, Any]:
    test_stores = (
        [] if data_dir is None or not data_dir.is_dir() else sorted(data_dir.glob("*.zarr"))
    )
    prediction_script = (
        None
        if support_dir is None
        else support_dir / "repo" / str(config.inference["prediction_script"])
    )
    weights_path = (
        None
        if weights_dir is None
        else weights_dir / Path(str(config.inference["weights_relative"]))
    )
    missing_modules = [
        name for name in REQUIRED_RUNTIME_MODULES if importlib.util.find_spec(name) is None
    ]
    return {
        "config_valid": True,
        "data_dir": None if data_dir is None else str(data_dir),
        "data_dir_exists": bool(data_dir and data_dir.is_dir()),
        "test_store_count": len(test_stores),
        "weights_path": None if weights_path is None else str(weights_path),
        "weights_exist": bool(weights_path and weights_path.is_file()),
        "support_dir": None if support_dir is None else str(support_dir),
        "prediction_script": None if prediction_script is None else str(prediction_script),
        "prediction_script_exists": bool(prediction_script and prediction_script.is_file()),
        "missing_runtime_modules": missing_modules,
        "ready_for_full_inference": bool(
            test_stores
            and weights_path
            and weights_path.is_file()
            and prediction_script
            and prediction_script.is_file()
        ),
    }


def require_full_inputs(report: dict[str, Any]) -> None:
    errors: list[str] = []
    if not report["data_dir_exists"]:
        errors.append("data directory is missing")
    elif report["test_store_count"] == 0:
        errors.append("data directory contains no .zarr test stores")
    if not report["weights_exist"]:
        errors.append(f"required weights are missing: {report['weights_path']}")
    if not report["prediction_script_exists"]:
        errors.append(
            f"support-artifact prediction script is missing: {report['prediction_script']}"
        )
    if report["missing_runtime_modules"]:
        errors.append("missing runtime modules: " + ", ".join(report["missing_runtime_modules"]))
    if errors:
        raise FileNotFoundError("V106 runtime inputs are incomplete:\n- " + "\n- ".join(errors))
