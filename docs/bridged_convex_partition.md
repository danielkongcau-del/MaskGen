# Bridged Convex Partition

This document describes the experimental splitter branch that starts after the fixed geometry approximator.

## Scope

Input is always:

```text
geometry approximator output -> approx_geometry
```

The geometry approximator is treated as fixed. This branch does not change `partition_gen/geometry_approximator.py`.

## Goal

The goal is to replace the current CDT-first splitter with a branch closer to the mathematical target:

```text
polygon with holes
  -> choose bridges
  -> cut holes open into a simple boundary walk
  -> convex partition backend
  -> convex pieces
```

The long-term target is an optimal convex partition backend, preferably CGAL's `optimal_convex_partition_2`.

## Current v1

The current implementation is a Python framework plus a CGAL CLI backend for simple polygons without holes.

Supported:

- input from geometry approximator JSON
- bridge candidate generation from polygon vertices
- `outer_star_v1` bridge set search for holes
- bridge metadata in output JSON
- boundary-walk metadata for visualization
- validation of convex pieces
- CGAL `optimal_convex_partition_2` backend for simple polygons with `hole_count == 0`
- CGAL bridge-cut backend for polygons with holes using `epsilon_slit_snap_v1`
- fallback to current CDT + greedy convex merge

Not yet implemented:

- general bridge tree cut-open traversal
- globally optimal bridge-tree selection for polygons with holes

## Backend Behavior

If `backend=auto` and a CGAL CLI named `optimal_convex_partition_cli` is not found, the module uses:

```text
backend = fallback_cdt_greedy
optimal = false
```

Fallback output must not be interpreted as optimal.

If `backend=cgal` is explicitly requested and the CLI is not available, the script fails with a clear error.

For polygons with holes, v1 uses `outer_star_v1` bridges, opens them with a very narrow slit, runs CGAL on the resulting simple polygon, then snaps slit-boundary vertices back to the original bridge centerline. The result is validated against the original polygon-with-holes.

After snapping, the implementation runs a local post-processing pass:

```text
snap slit vertices back
  -> cluster near-duplicate slit vertices
  -> remove near-duplicate / near-collinear vertices
  -> greedily merge adjacent pieces when their union is still strictly convex
```

This pass is only intended to remove numerical artifacts introduced by the temporary slit. It is not a semantic merge rule and does not change the fixed geometry approximator.

The hole path records:

```text
backend = cgal_bridge_cut
bridge_cut_mode = epsilon_slit_snap_v1
simple_polygon_optimal = true
optimal_scope = selected_bridge_cut_simple_polygon
global_optimal = false
```

This means CGAL is optimal for the selected cut-open simple polygon. It does not yet claim global optimality over all possible bridge trees.

For no-hole polygons, the CGAL backend records:

```text
backend = cgal
simple_polygon_optimal = true
optimal_scope = simple_polygon
global_optimal = true
```

Fallback results record:

```text
backend = fallback_cdt_greedy
simple_polygon_optimal = false
optimal_scope = fallback_cdt_greedy
global_optimal = false
```

## Build CGAL Backend

On Windows, use MSVC plus vcpkg:

```powershell
cmd /c 'call "D:\Microsoft Visual Studio\VC\Auxiliary\Build\vcvars64.bat" && cmake -Wno-dev -S tools -B build\cgal_tools -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake && cmake --build build\cgal_tools --config Release'
```

The expected executable is:

```text
build/cgal_tools/Release/optimal_convex_partition_cli.exe
```

`partition_gen/bridged_convex_partition.py` searches this location automatically before falling back.

## Files

```text
partition_gen/bridged_convex_partition.py
scripts/build_bridged_convex_partition_from_approx_single.py
scripts/visualize_bridged_convex_partition.py
tools/optimal_convex_partition_cli.cpp
tools/CMakeLists.txt
```

## Example

```powershell
conda run -n lmf python scripts/build_bridged_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json

conda run -n lmf python scripts/visualize_bridged_convex_partition.py `
  --partition-json data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_bridged_convex_partition.png
```

## Benchmarking

Use the benchmark tools to compare this splitter against the CDT + greedy baseline over geometry approximation JSON files:

```powershell
conda run -n lmf python scripts/benchmark_convex_splitters.py `
  --approx-root data/remote_256_geometry_approx_debug `
  --split val `
  --output outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --backend auto `
  --cut-slit-scales 1e-7 1e-6 1e-5

conda run -n lmf python scripts/summarize_convex_splitter_benchmark.py `
  --input outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output outputs/benchmarks/convex_splitter_benchmark_val.md
```

To inspect failures or non-improvements:

```powershell
conda run -n lmf python scripts/export_convex_splitter_failures.py `
  --benchmark-jsonl outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output-dir outputs/visualizations/convex_splitter_failures
```

## Output Fields

Important payload fields:

- `method`: should be `bridged_optimal_convex_partition`
- `bridge_policy`: currently `outer_star_v1`
- `backend_info.backend`
- `backend_info.optimal`
- `bridge_candidates`
- `selected_bridge_set`
- `simple_polygon_vertex_count`
- `simple_polygon_boundary_walk`
- `backend_info.bridge_cut_mode`
- `backend_info.optimal_scope`
- `backend_info.post_snap_cleanup_eps`
- `backend_info.post_snap_merge_count`
- `primitives`
- `validation`

Each primitive follows the same style as `convex_partition.py`:

```json
{
  "id": 0,
  "type": "triangle",
  "outer": [[0, 0], [1, 0], [0, 1]],
  "holes": [],
  "vertex_count": 3,
  "area": 0.5,
  "centroid": [0.333, 0.333]
}
```

## Validation

The validation block reports:

- `is_valid`
- `iou`
- `covered_area`
- `original_area`
- `overlap_area`
- `all_convex`
- `piece_count`

For fallback results, `validation.iou` should still be close to `1`, but `backend_info.optimal` remains `false`.

For `cgal_bridge_cut`, `validation.iou` is computed against the original polygon with holes after snapping slit edges back to the bridge centerline.

`--cut-slit-scale` controls only the numerical width of the temporary slit used to make the polygon strictly simple for CGAL. It is not a semantic geometry threshold; the slit edges are snapped back before validation/output.
