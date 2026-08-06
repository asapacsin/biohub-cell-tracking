"""Regenerate exact notebook-owned graph/evaluation functions from the preserved V106 source."""

from __future__ import annotations

import ast
import hashlib
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream_clean_v106" / "notebook_source.py"
EXPECTED_SHA256 = "718372e34e63eabe92148662deb7caac8e84c875accb7d6423e53f283286d352"

POST_HEADER = '''"""Notebook-owned graph repair code vendored from clean V106.\n\nSPDX-License-Identifier: Apache-2.0\n"""\nfrom __future__ import annotations\n\nimport json\nimport math\nimport os\nfrom pathlib import Path\n\ntry:\n    import blosc2\nexcept ImportError:\n    blosc2 = None\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.spatial import cKDTree\n\nTEST_DIR = Path(".")\n'''

EVAL_NAMES = [
    "SpecResult",
    "_normalise_nodes",
    "_normalise_edges",
    "_audit_prediction_nodes",
    "_node_match",
    "_adjacency",
    "_components",
    "_edge_counts",
    "_kuhn_maximum_matching",
    "_division_counts",
    "official_spec_evaluate",
    "_jaccard",
    "official_spec_per_sample",
    "official_spec_summarise",
    "_df_nodes",
    "_df_edges",
]


def _node_sources(text: str, names: list[str]) -> str:
    future = "from __future__ import annotations"
    parse_text = text.replace(future, " " * len(future))
    tree = ast.parse(parse_text)
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        name = getattr(node, "name", None)
        if name in names and name not in found:
            source = textwrap.dedent(ast.get_source_segment(text, node) or "")
            if name == "SpecResult":
                source = "@dataclass(frozen=True)\n" + source
            found[name] = source
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError(f"missing upstream definitions: {missing}")
    return "\n\n\n".join(found[name] for name in names) + "\n"


def main() -> None:
    text = UPSTREAM.read_text(encoding="utf-8")
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"upstream script fingerprint changed: {actual}")

    lines = text.splitlines()
    block = "\n".join(lines[919:2371]).rstrip() + "\n"
    if not block.startswith("def edge_distance_um") or "def filter_output_graph" not in block:
        raise RuntimeError("upstream postprocessing boundaries changed")

    defaults = """\nVOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)\n"""
    config_source = (ROOT / "configs" / "clean_v106.yaml").read_text(encoding="utf-8")
    defaults += """\n# Runtime globals are initialized from the authoritative YAML by configure().\ndef configure(settings: dict[str, object], test_dir: str | Path | None = None) -> None:\n    global TEST_DIR, VOXEL_SCALE_UM\n    if test_dir is not None:\n        TEST_DIR = Path(test_dir)\n    mapping = {key: key.upper() for key in settings if key != "voxel_scale_um"}\n    for key, global_name in mapping.items():\n        globals()[global_name] = settings[key]\n    VOXEL_SCALE_UM = tuple(float(v) for v in settings["voxel_scale_um"])\n\n"""
    del config_source
    (ROOT / "src" / "biohub_pipeline" / "postprocessing.py").write_text(
        POST_HEADER + defaults + block, encoding="utf-8"
    )

    eval_header = '''"""Official-spec-lite utilities copied from the V106 local-CV cell.\n\nSPDX-License-Identifier: Apache-2.0\n"""\nfrom __future__ import annotations\n\nfrom collections import defaultdict\nfrom dataclasses import dataclass\nfrom math import isfinite\n\nimport numpy as np\nimport pandas as pd\nfrom scipy.optimize import linear_sum_assignment\nfrom scipy.spatial import cKDTree\n\nOFFICIAL_SPEC_NODE_PENALTY_A = 0.1\nOFFICIAL_SPEC_DIVISION_WEIGHT = 0.1\n\n'''
    eval_source = _node_sources(text, EVAL_NAMES)
    (ROOT / "src" / "biohub_pipeline" / "evaluation.py").write_text(
        eval_header + eval_source, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
