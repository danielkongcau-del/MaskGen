# MaskGen

MaskGen is an experimental project for building a learnable structural representation of remote-sensing semantic masks.

The current direction is not direct pixel-level mask generation. The active line converts a `256x256` mask into polygonal structure, approximates each selected connected region with a simpler polygon, then decomposes that polygon into convex primitives that can later be modeled by a generator.

## Current Main Pipeline

```text
data/remote
  -> data/remote_256
  -> data/remote_256_partition
  -> geometry approximator
  -> constrained triangulation
  -> greedy convex merge
  -> convex primitives
```

The current working prototype focuses on selected connected regions, not the full-image generator yet.

Main idea:

- `partition graph`: extracts connected semantic regions (`faces`) from each mask.
- `geometry approximator`: uses the old base primitive decomposition only as a geometry approximation tool, then unions all base pieces back into one simplified polygon.
- `convex partition`: runs constrained Delaunay triangulation on the approximated polygon, then greedily merges adjacent pieces only when the union is still convex.

## Repository Layout

```text
data/
  remote/                                      original 1024x1024 data, optional but useful for regeneration
  remote_256/                                  256x256 masks and images
  remote_256_partition/                        partition graph dataset
  remote_256_geometry_approx_debug/            selected geometry approximator outputs
  remote_256_convex_partition_from_approx_debug/ selected convex partition outputs

partition_gen/
  geometry_approximator.py                     current geometry approximation layer
  convex_partition.py                          current CDT + greedy convex merge layer
  primitive_decomposition.py                   old base primitive decomposition used by the approximator
  dual_graph.py                                partition/dual-graph geometry helpers

scripts/
  prepare_remote_256.py                        build 256x256 masks
  build_partition_graph_dataset.py             build partition graphs
  build_geometry_approx_single.py              run geometry approximator for one face
  build_convex_partition_from_approx_single.py run convex partition from an approximator output
  visualize_geometry_approx.py                 visualize geometry approximation
  visualize_convex_partition.py                visualize CDT + convex merge output

docs/
  data.md
  pipeline.md
  commands.md
  experiments.md
  python_scripts.md
```

## Minimal Example

Run one region through the current mainline:

```powershell
conda run -n lmf python scripts/build_geometry_approx_single.py `
  --split val --stem 83 --face-id 16 `
  --output data/remote_256_geometry_approx_debug/val/graphs/83_face16.json

conda run -n lmf python scripts/build_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json

conda run -n lmf python scripts/visualize_convex_partition.py `
  --partition-json data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 --split val --stem 83 `
  --output outputs/visualizations/face83_16_convex_partition_from_approx.png
```

See [docs/commands.md](docs/commands.md) for more commands.

## Current Status

The active workflow is a representation-building prototype. It has been validated on selected examples such as:

- `val/83 face16`
- `val/10 face26`

It is not yet a complete unconditional mask generator. The next major step is to batch this workflow over more regions and then define the explanation selector and generator target.

## Notes

Several earlier model-based branches remain in code for reference, including topology AR generation, geometry decoding, global boundary prediction, and pair-boundary prediction. They are documented in [docs/experiments.md](docs/experiments.md), but they are not the current main path.
