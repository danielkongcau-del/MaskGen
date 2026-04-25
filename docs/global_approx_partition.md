# Global Approx Partition

This document describes the full-image shared-arc approximation branch based on face-level geometry approximations.

## Scope

This branch runs before the interpreter/generator and after the raw partition graph.

It does not modify the fixed face-level geometry approximator. It uses that approximator as the source of low-complexity boundary candidates, then reconciles adjacent faces into one shared planar map.

## Goal

The target representation is:

```text
partition graph small edges
  -> maximal boundary chains / arcs
  -> face-level geometry approximations
  -> owner boundary transfer under global planar validation
  -> face rings reconstructed from shared arcs
  -> per-face convex partition remains possible
```

The key unit is a maximal boundary chain:

```text
same incident face pair
continuous through non-junction vertices
stops at junctions / border endpoints / incident-pair changes
```

This is more natural than simplifying every small partition edge independently, because meaningless pixel-level boundary complexity is removed by the already accepted geometry approximator.

## Invariants

The output should satisfy:

- all face rings close and form valid polygons
- every shared boundary arc is stored once
- adjacent faces reference the same arc in opposite directions
- polygon union has no gap or overlap
- face adjacency matches the original partition graph
- junction count is reported and should not grow unexpectedly
- each reconstructed face can still be sent to the convex partition stage

## Reconciliation Strategy

Version 1 uses a conservative owner-transfer strategy:

```text
build exact maximal arcs as a valid base map
run geometry approximator for every face
for each shared arc, choose one owner face
extract the corresponding simplified boundary segment from the owner approximation
tentatively replace the raw arc with the owner segment
accept the replacement only if full-image validation still passes
otherwise keep the raw exact arc
```

This means adjacent faces always reference the same accepted arc. A low-complexity boundary is used only when it does not break closure, adjacency, gap/overlap checks, or downstream convex partitionability.

## Files

```text
partition_gen/global_approx_partition.py
scripts/build_global_approx_partition_single.py
scripts/visualize_global_approx_partition.py
tests/test_global_approx_partition.py
```

## Example

```powershell
conda run -n lmf python scripts/build_global_approx_partition_single.py `
  --partition-graph data/remote_256_partition/val/graphs/83.json `
  --output outputs/visualizations/global_approx_val83.json `
  --face-simplify-tolerance 1.5

conda run -n lmf python scripts/visualize_global_approx_partition.py `
  --global-json outputs/visualizations/global_approx_val83.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/global_approx_val83.png
```

## Output Fields

Top-level fields:

- `format`: `global_owner_approx_partition_v1`
- `source_partition_graph`
- `source_mask`
- `size`
- `arcs`
- `faces`
- `face_approximations`
- `reconciliation`
- `validation`

Each arc stores:

- `id`
- `incident_faces`
- `source_vertex_ids`
- `source_edge_ids`
- `original_points`
- `points`
- `original_vertex_count`
- `vertex_count`
- `method`
- `owner_face_id`
- `owner_distance`
- `vertex_reduction`
- `simplified`

Each face stores:

- `id`
- `label`
- `bbox`
- `outer_arc_refs`
- `hole_arc_refs`
- `outer`
- `holes`
- `approx_area`
- `is_valid`

## Validation

The validation block reports:

- `is_valid`
- `all_faces_valid`
- `all_face_rings_closed`
- `original_union_area`
- `approx_union_area`
- `overlap_area`
- `union_iou`
- `original_adjacency_count`
- `approx_adjacency_count`
- `missing_adjacency`
- `extra_adjacency`
- `arc_count`
- `shared_arc_count`
- `junction_count`
- `face_count`

The `reconciliation` block reports the owner-transfer policy, accepted owner arcs, and rejected owner arcs.
