#!/usr/bin/env bash
# Postprocess-only ablation of stages blamed for postprocessing_removed losses.
# Control vs motion_relink OFF (primary) and single_parent OFF (secondary check).
set -euo pipefail

SRC="${1:-/home/mc46451/biohub-cell-tracking}"
LOG_DIR="${SRC}/logs/slurm"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/ppstage_ablation_det0_96875.log"

FIXED8=(
  44b6_0113de3b 44b6_0b24845f 44b6_341df25f 44b6_e57ff5c6
  6bba_05b6850b 6bba_05db0fb1 6bba_969618f6 6bba_fc83837d
)
HOLDOUT8=(
  44b6_0c582fdc 44b6_0db75fae 44b6_12dfb391 44b6_144b256d
  6bba_062c8d37 6bba_07477033 6bba_07e24132 6bba_085bf656
)

cd "${SRC}"
STAGE_INPUTS=()
for d in "${FIXED8[@]}" "${HOLDOUT8[@]}"; do
  STAGE_INPUTS+=("data/competition/train/${d}.zarr" "data/competition/train/${d}.geff")
done

echo "Starting ppstage ablation at $(date -Iseconds)" | tee "${LOG}"

tar \
  --exclude='.venv' --exclude='.git' --exclude='**/__pycache__' \
  --exclude='outputs' --exclude='logs' \
  --exclude='data/support/repo/predictions' --exclude='data/tmp_seed314159' \
  -cf - \
  README.md LICENSE pyproject.toml configs src scripts \
  data/support \
  "${STAGE_INPUTS[@]}" \
| srun -p gpu_batch -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 0-01:30:00 \
  --chdir=/tmp \
  bash -lc "
set -euo pipefail
DEST=/tmp/\${USER}/biohub-ppstage-ablation
OUT_NFS=\${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875
FIXED_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_v1/raw_geff
HOLD_RAW=\${HOME}/biohub-outputs/holdout8/det0_96875_safeon/raw_geff
if [[ ! -d \"\$HOLD_RAW\" ]]; then
  HOLD_RAW=\${HOME}/biohub-outputs/experiments/candidate_edge_bottleneck_holdout8/raw_geff
fi

rm -rf \"\$DEST\"
mkdir -p \"\$DEST\"
tar xf - -C \"\$DEST\"
cd \"\$DEST\"

source \"\${HOME}/miniconda3/etc/profile.d/conda.sh\"
conda activate biohub
python -m pip install -U pip >/dev/null
python -m pip install --no-deps data/support/wheels/*.whl >/dev/null 2>&1 || true
python -m pip install --find-links=data/support/wheels --prefer-binary \
  tracksdata zarr \"geff>=1.1.3.1.1\" \"geff-spec<1.2\" \"ilpy>=0.5.1\" \
  pyscipopt polars blosc2 dask imagecodecs pyarrow \"rustworkx>=0.17.1\" \
  \"sqlalchemy>=2\" \"scikit-image>=0.24\" \"numcodecs>=0.13,<0.16\" donfig >/dev/null
export PYTHONPATH=\"\${DEST}/src\${PYTHONPATH:+:\$PYTHONPATH}\"

test -d \"\$FIXED_RAW\"
test -d \"\$HOLD_RAW\"
rm -rf \"\$OUT_NFS\"
mkdir -p \"\$OUT_NFS\"

export FIXED_RAW HOLD_RAW OUT_NFS
python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import tracksdata as td

from biohub_pipeline import postprocessing
from biohub_pipeline.config import load_config
from biohub_pipeline.evaluation import official_spec_summarise
from biohub_pipeline.fixed8_cv import (
    FIXED8_DATASETS,
    evaluate_postprocessed_predictions,
    evaluate_tables,
    write_metric_outputs,
    _estimated_total_nodes,
    _read_graph_tables,
)
from biohub_pipeline.submission import graph_rows, validate_submission_file, write_rows

FIXED_RAW = Path(os.environ['FIXED_RAW'])
HOLD_RAW = Path(os.environ['HOLD_RAW'])
OUT_NFS = Path(os.environ['OUT_NFS'])
DATA_DIR = Path('data/competition/train')
CONTROL_SCORE = 0.8847464271589631
HOLDOUT_CONTROL_SCORE = 0.9590130516216544

HOLDOUT8 = [
    '44b6_0c582fdc', '44b6_0db75fae', '44b6_12dfb391', '44b6_144b256d',
    '6bba_062c8d37', '6bba_07477033', '6bba_07e24132', '6bba_085bf656',
]

VARIANTS = [
    'ppstage_control_det0_96875',
    'ppstage_motion_relink_off_det0_96875',
    'ppstage_single_parent_off_det0_96875',
    'ppstage_motion_and_parent_off_det0_96875',
]


def evaluate_custom(pred_csv: Path, datasets: list[str], config):
    predictions = pd.read_csv(pred_csv)
    found = sorted(predictions['dataset'].unique().tolist())
    if found != sorted(datasets):
        raise RuntimeError(f'datasets {found} != {sorted(datasets)}')
    scale = tuple(float(v) for v in config.postprocessing['voxel_scale_um'])
    max_distance_um = float(config.local_cv['max_match_um'])
    rows = []
    for dataset in datasets:
        part = predictions[predictions['dataset'] == dataset]
        pred_nodes = part[part['row_type'] == 'node'].loc[:, ['node_id', 't', 'z', 'y', 'x']]
        pred_edges = part[part['row_type'] == 'edge'].loc[:, ['source_id', 'target_id']]
        gt_path = DATA_DIR / f'{dataset}.geff'
        gt_nodes, gt_edges = _read_graph_tables(gt_path)
        rows.append(
            evaluate_tables(
                dataset, pred_nodes, pred_edges, gt_nodes, gt_edges,
                _estimated_total_nodes(gt_path), scale=scale, max_distance_um=max_distance_um,
            )
        )
    return rows, {**official_spec_summarise(rows), 'datasets': list(datasets)}


def convert_and_score(raw_dir: Path, datasets: list[str], config_path: Path, out: Path, label: str):
    config = load_config(config_path)
    assert abs(float(config.inference['detection_threshold']) - 0.96875) < 1e-12
    assert float(config.inference['ensemble_alpha']) == 0.5
    assert config.postprocessing['output_safe_divisions'] is True
    assert config.postprocessing['output_gap2_recovery'] is False
    assert config.postprocessing['use_deepcenter_veto'] is False
    assert config.postprocessing['output_filter_short_tracks'] is True

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / 'config').mkdir()
    shutil.copy2(config_path, out / 'config' / config_path.name)

    postprocessing.configure(config.postprocessing, DATA_DIR)
    pred_csv = out / 'predictions' / 'postprocessed_submission.csv'
    all_rows = []
    stats_by_ds = {}
    for dataset in datasets:
        geff_path = raw_dir / f'{dataset}.geff'
        assert geff_path.exists(), geff_path
        loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = loaded[0] if isinstance(loaded, tuple) else loaded
        nodes = {
            int(r['node_id']): {
                'node_id': int(r['node_id']), 't': int(r['t']),
                'z': float(r['z']), 'y': float(r['y']), 'x': float(r['x']),
            }
            for r in graph.node_attrs().iter_rows(named=True)
        }
        edges = []
        for r in graph.edge_attrs().iter_rows(named=True):
            p = r.get('edge_prob') if hasattr(r, 'get') else None
            edges.append({
                'source_id': int(r['source_id']),
                'target_id': int(r['target_id']),
                'edge_prob': None if p is None else float(p),
            })
        nodes, edges, stats = postprocessing.filter_output_graph(nodes, edges, dataset=dataset)
        stats_by_ds[dataset] = {
            k: int(stats.get(k, 0))
            for k in (
                'motion_relink_edges', 'motion_relink_tight_edges', 'motion_relink_relaxed_edges',
                'motion_relink_replaced_raw_edges', 'motion_relink_fallback_raw',
                'dropped_multi_parent_edges', 'short_track_nodes_removed', 'short_track_edges_removed',
            )
        }
        all_rows.extend(graph_rows(dataset, nodes, edges, len(all_rows)))

    write_rows(all_rows, pred_csv)
    validate_submission_file(pred_csv)
    (out / 'predictions' / 'stage_stats.json').write_text(
        json.dumps(stats_by_ds, indent=2, sort_keys=True) + '\n'
    )

    if datasets == list(FIXED8_DATASETS):
        rows, aggregate = evaluate_postprocessed_predictions(pred_csv, DATA_DIR, config)
        delta = float(aggregate['score']) - CONTROL_SCORE
    else:
        rows, aggregate = evaluate_custom(pred_csv, datasets, config)
        delta = float(aggregate['score']) - HOLDOUT_CONTROL_SCORE

    summary = {
        **aggregate,
        'label': label,
        'detection_threshold': 0.96875,
        'ensemble_alpha': 0.5,
        'output_motion_relink': bool(config.postprocessing['output_motion_relink']),
        'output_single_parent_repair': bool(config.postprocessing['output_single_parent_repair']),
        'output_filter_short_tracks': bool(config.postprocessing['output_filter_short_tracks']),
        'delta_vs_control': delta,
        'motion_relink_replaced_raw_edges_total': int(
            sum(int(v.get('motion_relink_replaced_raw_edges', 0)) for v in stats_by_ds.values())
        ),
        'dropped_multi_parent_edges_total': int(
            sum(int(v.get('dropped_multi_parent_edges', 0)) for v in stats_by_ds.values())
        ),
    }
    manifest = {
        'schema_version': 1,
        'experiment': label,
        'raw_geff_source': str(raw_dir),
        'config': str(config_path),
        'datasets': list(datasets),
        'note': 'Postprocess-only stage ablation; no re-inference',
    }
    write_metric_outputs(out, rows, summary, manifest)
    (out / 'DONE').write_text('ok\\n')
    return summary, rows


comparison = {
    'control_score_fixed8': CONTROL_SCORE,
    'control_score_holdout8': HOLDOUT_CONTROL_SCORE,
    'fixed8_raw_geff': str(FIXED_RAW),
    'holdout_raw_geff': str(HOLD_RAW),
    'hypothesis': 'Disabling motion_relink recovers harmful postprocessing_removed edges without net FP increase',
    'variants': {},
}

started = time.perf_counter()
for name in VARIANTS:
    cfg = Path(f'configs/experiments/{name}.yaml')
    print('VARIANT', name, flush=True)
    f_out = OUT_NFS / 'fixed8' / name
    f_sum, f_rows = convert_and_score(FIXED_RAW, list(FIXED8_DATASETS), cfg, f_out, f'fixed8_{name}')
    h_out = OUT_NFS / 'holdout8' / name
    h_sum, h_rows = convert_and_score(HOLD_RAW, HOLDOUT8, cfg, h_out, f'holdout8_{name}')
    comparison['variants'][name] = {
        'fixed8': {
            'score': float(f_sum['score']),
            'delta_vs_control': float(f_sum['delta_vs_control']),
            'edge_tp_fp_fn': [f_sum['edge_tp'], f_sum['edge_fp'], f_sum['edge_fn']],
            'div_tp_fp_fn': [f_sum['division_tp'], f_sum['division_fp'], f_sum['division_fn']],
            'motion_relink': f_sum['output_motion_relink'],
            'single_parent': f_sum['output_single_parent_repair'],
            'motion_relink_replaced_raw_edges_total': f_sum['motion_relink_replaced_raw_edges_total'],
            'dropped_multi_parent_edges_total': f_sum['dropped_multi_parent_edges_total'],
            'per_dataset': [
                {
                    'dataset': r['dataset'],
                    'adj_edge_jaccard': float(r['adj_edge_jaccard']),
                    'edge_tp': int(r['edge_tp']),
                    'edge_fp': int(r['edge_fp']),
                    'edge_fn': int(r['edge_fn']),
                }
                for r in f_rows
            ],
        },
        'holdout8': {
            'score': float(h_sum['score']),
            'delta_vs_control': float(h_sum['delta_vs_control']),
            'edge_tp_fp_fn': [h_sum['edge_tp'], h_sum['edge_fp'], h_sum['edge_fn']],
            'div_tp_fp_fn': [h_sum['division_tp'], h_sum['division_fp'], h_sum['division_fn']],
            'motion_relink': h_sum['output_motion_relink'],
            'single_parent': h_sum['output_single_parent_repair'],
            'motion_relink_replaced_raw_edges_total': h_sum['motion_relink_replaced_raw_edges_total'],
            'dropped_multi_parent_edges_total': h_sum['dropped_multi_parent_edges_total'],
            'per_dataset': [
                {
                    'dataset': r['dataset'],
                    'adj_edge_jaccard': float(r['adj_edge_jaccard']),
                    'edge_tp': int(r['edge_tp']),
                    'edge_fp': int(r['edge_fp']),
                    'edge_fn': int(r['edge_fn']),
                }
                for r in h_rows
            ],
        },
    }
    print(
        '  fixed8', f_sum['score'], 'delta', f_sum['delta_vs_control'],
        'holdout', h_sum['score'], 'delta', h_sum['delta_vs_control'],
        flush=True,
    )

ctrl = comparison['variants']['ppstage_control_det0_96875']
assert abs(ctrl['fixed8']['score'] - CONTROL_SCORE) < 1e-9, ctrl['fixed8']['score']
assert abs(ctrl['holdout8']['score'] - HOLDOUT_CONTROL_SCORE) < 1e-9, ctrl['holdout8']['score']

ranked = sorted(
    (
        (
            v['fixed8']['delta_vs_control'] + v['holdout8']['delta_vs_control'],
            name,
            v,
        )
        for name, v in comparison['variants'].items()
        if name != 'ppstage_control_det0_96875'
    ),
    reverse=True,
)
best_name, best = ranked[0][1], ranked[0][2]
promote = (
    best['fixed8']['delta_vs_control'] >= 0.001
    or (
        best['fixed8']['delta_vs_control'] > 0
        and best['holdout8']['delta_vs_control'] > 0
        and best['fixed8']['delta_vs_control'] + best['holdout8']['delta_vs_control'] >= 0.001
    )
) and best['holdout8']['delta_vs_control'] >= -0.002

comparison['best_variant'] = best_name
comparison['promote_best'] = bool(promote)
comparison['runtime_seconds'] = time.perf_counter() - started
comparison['decision'] = (
    f'PROMOTE {best_name}'
    if comparison['promote_best']
    else f'KEEP CONTROL; best exploratory={best_name} but evidence insufficient/risky'
)
(OUT_NFS / 'comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
(OUT_NFS / 'DONE').write_text('ok\n')
print('COMPARISON', json.dumps(comparison, indent=2, sort_keys=True))
print('DECISION', comparison['decision'])
PY
" 2>&1 | tee -a "${LOG}"

mkdir -p "${SRC}/outputs/experiments/ppstage_ablation_det0_96875"
srun -p gpu_batch -N1 -n1 --gres=gpu:1 -t 00:08:00 --mem=4G --chdir=/tmp bash -lc '
  cd ${HOME}/biohub-outputs/experiments/ppstage_ablation_det0_96875 && \
  tar cf - DONE comparison.json \
    fixed8/ppstage_control_det0_96875/summary.json \
    fixed8/ppstage_control_det0_96875/per_dataset.csv \
    fixed8/ppstage_motion_relink_off_det0_96875/summary.json \
    fixed8/ppstage_motion_relink_off_det0_96875/per_dataset.csv \
    fixed8/ppstage_single_parent_off_det0_96875/summary.json \
    fixed8/ppstage_single_parent_off_det0_96875/per_dataset.csv \
    fixed8/ppstage_motion_and_parent_off_det0_96875/summary.json \
    fixed8/ppstage_motion_and_parent_off_det0_96875/per_dataset.csv \
    holdout8/ppstage_control_det0_96875/summary.json \
    holdout8/ppstage_control_det0_96875/per_dataset.csv \
    holdout8/ppstage_motion_relink_off_det0_96875/summary.json \
    holdout8/ppstage_motion_relink_off_det0_96875/per_dataset.csv \
    holdout8/ppstage_single_parent_off_det0_96875/summary.json \
    holdout8/ppstage_single_parent_off_det0_96875/per_dataset.csv \
    holdout8/ppstage_motion_and_parent_off_det0_96875/summary.json \
    holdout8/ppstage_motion_and_parent_off_det0_96875/per_dataset.csv
' | tar xf - -C "${SRC}/outputs/experiments/ppstage_ablation_det0_96875"
echo "LOGIN_MIRROR_OK $(date -Iseconds)" | tee -a "${LOG}"
cat "${SRC}/outputs/experiments/ppstage_ablation_det0_96875/comparison.json"
