# Commands

Commands assume PowerShell from the repository root.

The expected conda environment is:

```powershell
conda run -n lmf ...
```

## Prepare `256x256` Data

```powershell
conda run -n lmf python scripts/prepare_remote_256.py
```

Output:

```text
data/remote_256
```

## Build Partition Graph Dataset

```powershell
conda run -n lmf python scripts/build_partition_graph_dataset.py `
  --input-root data/remote_256 `
  --output-root data/remote_256_partition
```

Output:

```text
data/remote_256_partition
```

## Build Geometry Approximation For One Face

Example:

```powershell
conda run -n lmf python scripts/build_geometry_approx_single.py `
  --split val `
  --stem 83 `
  --face-id 16 `
  --output data/remote_256_geometry_approx_debug/val/graphs/83_face16.json
```

Output:

```text
data/remote_256_geometry_approx_debug/val/graphs/83_face16.json
```

## Visualize Geometry Approximation

```powershell
conda run -n lmf python scripts/visualize_geometry_approx.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_geometry_approx.png
```

## Build Convex Partition From Geometry Approximation

```powershell
conda run -n lmf python scripts/build_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json
```

Output:

```text
data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json
```

## Build Bridged Convex Partition From Geometry Approximation

This branch records bridge candidates for holes and then uses an optimal backend when available. If CGAL is not available, it falls back to the current CDT + greedy convex merge and marks `optimal=false`.

```powershell
conda run -n lmf python scripts/build_bridged_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json
```

## Visualize Bridged Convex Partition

```powershell
conda run -n lmf python scripts/visualize_bridged_convex_partition.py `
  --partition-json data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_bridged_convex_partition.png
```

## Visualize Convex Partition

```powershell
conda run -n lmf python scripts/visualize_convex_partition.py `
  --partition-json data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_convex_partition_from_approx.png
```

## Raw Polygon Convex Partition

This bypasses the geometry approximator and runs directly on a face from `data/remote_256_partition`.

```powershell
conda run -n lmf python scripts/build_convex_partition_single.py `
  --partition-root data/remote_256_partition `
  --split val `
  --stem 83 `
  --face-id 16 `
  --output data/remote_256_convex_partition_debug/val/graphs/83_face16.json
```

This is useful for comparison, but not the current preferred path.

## Historical Commands

Training and sampling commands for topology, geometry, boundary, and pair-boundary models still exist under `scripts/`.

They are documented in [python_scripts.md](python_scripts.md), but they are not the current recommended workflow.
