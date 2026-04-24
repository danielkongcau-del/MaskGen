# Data

This document describes the data directories currently used by the project.

## Active Data

### `data/remote`

Original source data. This directory is large and is not required for every experiment, but it is useful if `data/remote_256` needs to be regenerated.

Current size is about `2.9 GB`.

### `data/remote_256`

The working `256x256` version of the dataset.

Important subdirectories:

- `train/masks_id`, `val/masks_id`, `test/masks_id`: masks stored as class ids.
- `train/masks`, `val/masks`, `test/masks`: masks stored with original grayscale values.
- `meta/class_map.json`: mapping between grayscale values and class ids.

This is the main mask source for all current scripts.

### `data/remote_256_partition`

Partition graph dataset built from `data/remote_256`.

Each JSON describes a full mask as a planar partition:

- `vertices`: global polygon vertices
- `faces`: connected semantic regions
- `adjacency`: face adjacency relations
- per-face attributes such as `label`, `area`, `bbox`, `outer`, and `holes`

This is the structural source used by the current mainline.

### `data/remote_256_geometry_approx_debug`

Debug output from the current geometry approximator.

Each file represents one selected face after:

```text
old base primitive decomposition -> union all base primitives -> approximated polygon
```

This directory currently contains selected examples, not a full dataset.

### `data/remote_256_convex_partition_from_approx_debug`

Debug output from:

```text
geometry approximator output -> constrained triangulation -> greedy convex merge
```

This directory currently contains selected examples, not a full dataset.

## Removed Or Historical Data

Earlier experiment outputs have been cleaned from the workspace. These included:

- `remote_256_boundary`
- `remote_256_dual`
- `remote_256_geometry*`
- old primitive debug datasets
- old convex partition debug datasets

Those branches are documented in [experiments.md](experiments.md), but their derived data is not needed for the current mainline.

## Rebuild Order

If rebuilding from scratch, use this order:

1. `data/remote` -> `data/remote_256`
2. `data/remote_256` -> `data/remote_256_partition`
3. selected `partition` face -> `geometry_approx`
4. `geometry_approx` -> convex partition

See [commands.md](commands.md) for concrete commands.

## Git Guidance

Most data directories should not be committed. The repo currently treats `data/` as generated/local data. For GitHub, prefer committing code and documentation, not the local datasets.
