# Legacy Layout Routes Archive

Archived on 2026-04-30.

This folder contains standalone script entrypoints for superseded layout experiments:

- absolute layout AR
- relative layout AR
- retrieved layout baselines
- retrieved residual layout baselines
- layout-frame MLP diagnostics

The current active route is the parent-first coarse scene pipeline in the top-level `scripts/` folder:

- `tokenize_manual_coarse_scene_dataset.py`
- `train_manual_coarse_scene_ar.py`
- `sample_manual_coarse_scene_ar.py`
- `attach_coarse_scene_true_shape_to_samples.py`
- `evaluate_manual_coarse_scene_relation_spatial.py`

Only script entrypoints were archived. Shared modules in `partition_gen/` and regression tests in `tests/` were left in place because the active coarse scene code still reuses common split IO and geometry helper functions.
