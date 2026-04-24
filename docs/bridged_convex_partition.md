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

The current implementation is a Python framework plus fallback backend.

Supported:

- input from geometry approximator JSON
- bridge candidate generation from polygon vertices
- `outer_star_v1` bridge set search for holes
- bridge metadata in output JSON
- boundary-walk metadata for visualization
- validation of convex pieces
- fallback to current CDT + greedy convex merge

Not yet implemented:

- general bridge tree cut-open traversal
- CGAL CLI integration
- true optimal convex partition on polygons with holes

## Backend Behavior

If `backend=auto` and a CGAL CLI named `optimal_convex_partition_cli` is not found, the module uses:

```text
backend = fallback_cdt_greedy
optimal = false
```

Fallback output must not be interpreted as optimal.

If `backend=cgal` is explicitly requested and the CLI is not available, the script fails with a clear error.

## Files

```text
partition_gen/bridged_convex_partition.py
scripts/build_bridged_convex_partition_from_approx_single.py
scripts/visualize_bridged_convex_partition.py
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
