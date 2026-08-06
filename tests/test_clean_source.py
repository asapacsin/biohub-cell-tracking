from __future__ import annotations

import hashlib
from pathlib import Path


def test_selected_notebook_snapshot_is_exact() -> None:
    path = Path("upstream_clean_v106/clean-approach-lightweight-local-cv-no-hack.ipynb")
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5adc99aef3b61f2d8c5da5253eb1df13262986e8879bf6f630b5c1b5fa345d9d"
    )


def test_vendored_source_and_license_are_present() -> None:
    assert Path("upstream_clean_v106/notebook_source.py").is_file()
    assert "Apache License" in Path("LICENSE").read_text(encoding="utf-8")
