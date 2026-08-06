# Upstream source record

- Author: Yusuke Togashi
- Title: `Clean Approach + Lightweight Local CV | No Hack`
- URL: <https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack>
- Previous slug: `yusuketogashi/biohub-another-approach` (redirects to the selected notebook)
- Selected version: 106 of 106
- Reported V106 public score: 0.908
- Reported best score: 0.908 at V96
- Runtime reported by Kaggle: 35m 16s on GPU T4 x2
- Licence: Apache-2.0
- Notebook SHA-256: `5adc99aef3b61f2d8c5da5253eb1df13262986e8879bf6f630b5c1b5fa345d9d`
- Jupytext source byte SHA-256: `2af5f0977ccbb22e326e5badce360d6b16f0be3b26edf022350de1852a69f5d3`

## Selection reason

V96 and V106 share the highest reported accessible clean score, 0.908. V106 is the newest
complete accessible version, explicitly rejects artificial metric exploitation, contains the
fixed-8 local-CV workflow, and documents the support artifact and dependencies. The old slug now
redirects to this source. API-only historical pulls were inaccessible, so no unavailable version
was reconstructed.

## Required inputs

1. Kaggle competition `biohub-cell-tracking-during-development` (`test/*.zarr`; `train/*.zarr`
   and `train/*.geff` only for fixed-8 CV).
2. Dataset `pilkwang/biohub-tracking-support-pack-50ep-v1`, containing `repo/`, `weights/`, and
   optional offline wheels.
3. Weight `weights/unet_transformer/split_0/edge_predictor_best.pth`.
4. The hidden cover-image dataset is presentation-only and not required by inference.

The notebook expects `tracksdata`, Zarr 3, PySCIPOpt, GEFF/geff-spec, ilpy, Polars, Blosc2, Dask,
imagecodecs, scikit-image, PyArrow, rustworkx, SQLAlchemy, numcodecs, donfig, and their documented
transitive dependencies. The support artifact is the preferred offline source.

The selected notebook itself does not contain the learned model implementation or weights; it
materializes both from the support dataset. They must not be approximated.

