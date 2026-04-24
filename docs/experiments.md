# Experiments

This document records important branches that were tried before the current mainline.

## Direct Raster Generation

Early discussion considered direct unconditional generation of `256x256` masks.

Reason it is not the current path:

- raw mask pixels hide topology and region relationships
- structural losses are weak constraints
- generated masks are likely to be locally plausible but globally incoherent

## Topology AR Generator

Files still present:

- `partition_gen/ar_dataset.py`
- `partition_gen/topology_training.py`
- `partition_gen/models/topology_transformer.py`
- `scripts/train_topology.py`
- `scripts/sample_topology.py`

This branch generated face-level dual graph topology with an autoregressive transformer.

Reason it is not current mainline:

- topology generation itself was feasible
- downstream geometry recovery was the bottleneck
- the representation was not yet good enough for high-quality mask reconstruction

## Geometry Decoder

Files still present:

- `partition_gen/geometry_dataset.py`
- `partition_gen/geometry_training.py`
- `partition_gen/models/geometry_decoder.py`
- `scripts/train_geometry.py`
- `scripts/sample_geometry.py`
- `scripts/eval_geometry.py`

This branch predicted face polygons from true or generated dual graphs.

Reason it is not current mainline:

- per-face polygon prediction treated neighboring faces too independently
- shared boundaries were not enforced strongly enough
- generated geometry often drifted or simplified into low-complexity shapes

## Global Boundary Prediction

Files still present:

- `partition_gen/boundary_dataset.py`
- `partition_gen/boundary_training.py`
- `partition_gen/models/boundary_predictor.py`
- `scripts/train_boundary.py`
- `scripts/sample_boundary.py`
- `scripts/eval_boundary.py`

This branch predicted a shared boundary mask conditioned on a graph.

Reason it is not current mainline:

- global binary boundary supervision was too coarse
- the model did not know which face pair each boundary segment belonged to
- rendering from predicted boundaries was unstable on complex samples

## Pair Boundary Prediction

Files still present:

- `partition_gen/pair_boundary_training.py`
- `partition_gen/pair_render.py`
- `partition_gen/models/pair_boundary_predictor.py`
- `scripts/train_boundary_pairs.py`
- `scripts/sample_boundary_pairs.py`
- `scripts/eval_boundary_pairs.py`

This branch predicted boundary masks for each face pair.

Reason it is not current mainline:

- pair-level supervision improved identity information
- overlapping pair predictions created ambiguity
- final render quality was still limited by boundary prediction stability

## Primitive And Composite Groups

Files still present:

- `partition_gen/primitive_decomposition.py`
- `scripts/build_primitive_dataset.py`
- `scripts/visualize_face_primitives.py`
- `scripts/visualize_composite_patch_scores.py`

This branch decomposed each face into triangles/quads, then compressed or grouped them.

Useful piece retained:

- the `base` primitive decomposition is now used by the geometry approximator

Reason it is not the final representation:

- treating the base pieces as real structure led back to unstable internal boundaries
- composite grouping could over-merge or produce representations that were not generator-friendly

## Raw Polygon CDT

Files:

- `partition_gen/convex_partition.py`
- `scripts/build_convex_partition_single.py`
- `scripts/visualize_convex_partition.py`

This branch directly ran constrained triangulation and greedy convex merge on raw partition polygons.

Reason it is not enough alone:

- raw pixel polygon boundaries are too jagged
- constrained triangulation produced hundreds of small triangles on complex faces

This led to the current approach:

```text
base geometry approximation -> CDT -> greedy convex merge
```

## Current Lesson

The main issue was not only model architecture. The representation itself needed to become lower entropy and easier to generate.

The current line keeps the useful pieces:

- partition graph extraction
- base primitive geometry approximation
- constrained triangulation
- convex primitive merging

It discards or pauses model branches until the representation is stable.
