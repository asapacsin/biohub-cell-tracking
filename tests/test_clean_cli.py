from __future__ import annotations

import json

import pytest

from biohub_pipeline.run import main


def test_dry_run_needs_no_real_data_or_weights(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", "configs/clean_v106.yaml", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_valid"] is True
    assert report["model_loaded"] is False
    assert report["inference_started"] is False
    assert report["ready_for_full_inference"] is False


def test_normal_run_has_clear_missing_input_error() -> None:
    with pytest.raises(FileNotFoundError, match="runtime inputs are incomplete"):
        main(["--config", "configs/clean_v106.yaml"])
