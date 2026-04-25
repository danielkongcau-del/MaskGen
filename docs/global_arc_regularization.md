# Global Arc Regularization

This module is the final geometry cleanup layer before interpretation.

It runs after full-image shared-boundary approximation:

```text
partition graph
  -> geometry approximator
  -> global shared-arc approximation
  -> global arc regularization
  -> interpreter
```

The goal is to remove raster staircase artifacts from already shared arcs without changing the planar topology.

## Scope

The first version only regularizes arcs. It does not delete or merge semantic faces.

This distinction matters:

- Staircase artifacts are boundary geometry noise.
- Tiny sliver faces may be semantic regions or representation artifacts.

Sliver cleanup therefore needs a separate policy and is not part of this module yet.

## Method

For each shared arc:

1. Keep the two arc endpoints fixed.
2. Check whether all intermediate vertices are close to the straight segment between the endpoints.
3. If the maximum bidirectional Hausdorff distance is below `max_distance`, replace the arc with the straight segment.
4. Rebuild all face rings from the shared arcs.
5. Accept the replacement only if full-image validation still passes.

The current implementation also supports local subsegment smoothing. If only one portion of a longer arc is a staircase artifact, the regularizer can replace that subsegment with a straight segment while preserving larger semantic turns elsewhere on the same arc.

For residual staircases that cross several arc boundaries or junctions, an optional face-chain pass can smooth a near-linear chain along a face boundary. It projects the chain's points onto a straight line, updates all shared arcs using those moved junctions, then accepts the change only if the full-image map remains valid. This pass runs on both outer rings and hole rings, because many road/building staircases appear as internal boundaries of a large face rather than as one simple shared arc.

An experimental strip-face pass can also snap an entire thin, elongated face to its minimum rotated rectangle. This is intentionally conservative: it only helps when a semantic region is close to a single strip. Complex road/void regions should still be handled by face-chain smoothing or a future local strip-segment pass, not by forcing the whole face into one rectangle.

The validation requires:

- all face rings remain valid;
- shared boundaries are still stored once;
- neighboring faces still reference the same arc in opposite directions;
- union IoU remains close to 1;
- no gap / overlap is introduced;
- original adjacency is preserved.

## Conservative Defaults

Default mode is line-only smoothing:

```text
allow_polyline_smoothing = false
enable_subsegment_smoothing = true
```

This avoids turning meaningful multi-segment semantic boundaries into over-smoothed polylines. Multi-segment RDP smoothing exists as an explicit option for experiments, but it is not the default.

For more aggressive cleanup where small semantic boundary shifts are acceptable, increase the distance and search span:

```text
max_distance = 2.0
max_subsegment_span = 96
max_candidates_per_arc = 96
enable_face_chain_smoothing = true
face_chain_max_distance = 2.0
face_chain_max_span = 96
```

For cases where the goal is to remove visibly meaningless staircases and a small semantic loss is acceptable, a stronger preset is:

```text
max_distance = 2.5
max_subsegment_span = 128
max_candidates_per_arc = 128
enable_face_chain_smoothing = true
face_chain_max_distance = 3.5
face_chain_max_span = 128
max_face_chain_candidates = 1024
enable_strip_face_smoothing = true
strip_min_aspect_ratio = 3.0
strip_max_width = 18.0
```

This mode is slower because every accepted candidate is checked by rebuilding and validating the whole planar map.

## Commands

Build a regularized global approximation:

```powershell
conda run -n lmf python scripts/build_regularized_global_approx_single.py `
  --global-json outputs/visualizations/global_approx_val37_risk.json `
  --partition-graph data/remote_256_partition/val/graphs/37.json `
  --output outputs/visualizations/global_approx_val37_regularized.json `
  --simplify-tolerance 1.25 `
  --max-distance 1.25
```

Aggressive cleanup:

```powershell
conda run -n lmf python scripts/build_regularized_global_approx_single.py `
  --global-json outputs/visualizations/global_approx_val37_risk.json `
  --partition-graph data/remote_256_partition/val/graphs/37.json `
  --output outputs/visualizations/global_approx_val37_regularized_aggressive.json `
  --simplify-tolerance 1.25 `
  --max-distance 2.0 `
  --max-subsegment-span 96 `
  --max-candidates-per-arc 96 `
  --enable-face-chain-smoothing `
  --face-chain-max-distance 2.0 `
  --face-chain-max-span 96
```

Aggressive cleanup with controlled semantic loss:

```powershell
conda run -n lmf python scripts/build_regularized_global_approx_single.py `
  --global-json outputs/visualizations/global_approx_val37_risk.json `
  --partition-graph data/remote_256_partition/val/graphs/37.json `
  --output outputs/visualizations/global_approx_val37_regularized_semantic_loss.json `
  --simplify-tolerance 1.5 `
  --max-distance 2.5 `
  --max-subsegment-span 128 `
  --max-candidates-per-arc 128 `
  --enable-face-chain-smoothing `
  --face-chain-max-distance 3.5 `
  --face-chain-max-span 128 `
  --max-face-chain-candidates 1024 `
  --enable-strip-face-smoothing `
  --strip-min-aspect-ratio 3.0 `
  --strip-max-width 18.0
```

Visualize before / after:

```powershell
conda run -n lmf python scripts/visualize_global_arc_regularization.py `
  --before-json outputs/visualizations/global_approx_val37_risk.json `
  --after-json outputs/visualizations/global_approx_val37_regularized.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 37 `
  --output outputs/visualizations/global_approx_val37_regularized.png
```

## Current Limitations

- It does not remove sliver faces.
- It does not reason about semantic class priority.
- The face-chain pass can move junction coordinates, but it preserves the graph's arc/reference structure and requires full-image validation.
- It only straightens full arcs, subsegments, or face-boundary chains whose existing shape is already close to a straight segment.
- The strip-face pass only handles whole faces that are close to a single narrow rectangle. It is not a general road-network simplifier.
