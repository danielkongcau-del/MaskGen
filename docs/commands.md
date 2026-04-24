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

This branch records bridge candidates for holes and then uses CGAL when available. Simple polygons run directly through `CGAL::optimal_convex_partition_2`; polygons with holes use `epsilon_slit_snap_v1` to cut holes open before CGAL. If CGAL is not available, it falls back to the current CDT + greedy convex merge and marks `optimal=false`.

Build the CGAL CLI backend first if you want true optimal partitioning for simple polygons without holes:

```powershell
cmd /c 'call "D:\Microsoft Visual Studio\VC\Auxiliary\Build\vcvars64.bat" && cmake -Wno-dev -S tools -B build\cgal_tools -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake && cmake --build build\cgal_tools --config Release'
```

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

## Benchmark Convex Splitters

Compare the CDT + greedy baseline against the bridged CGAL splitter on geometry approximation JSON files.

```powershell
conda run -n lmf python scripts/benchmark_convex_splitters.py `
  --approx-root data/remote_256_geometry_approx_debug `
  --split val `
  --output outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --backend auto `
  --cut-slit-scales 1e-7 1e-6 1e-5
```

Summarize the JSONL benchmark:

```powershell
conda run -n lmf python scripts/summarize_convex_splitter_benchmark.py `
  --input outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output outputs/benchmarks/convex_splitter_benchmark_val.md
```

Export visual checks for low-IoU, fallback, rejected-bridge, or non-improving cases:

```powershell
conda run -n lmf python scripts/export_convex_splitter_failures.py `
  --benchmark-jsonl outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output-dir outputs/visualizations/convex_splitter_failures
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
