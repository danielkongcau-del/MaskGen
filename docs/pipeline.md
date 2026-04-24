# Pipeline

This document describes the current main workflow.

## Goal

The project is trying to build a structural representation that is easier for a future generator to learn than raw pixels.

The current target is not:

```text
noise -> 256x256 mask pixels
```

The current target is closer to:

```text
mask -> connected regions -> approximated polygons -> convex primitives -> future generator target
```

## Step 1: Build `256x256` Masks

Script:

```text
scripts/prepare_remote_256.py
```

Input:

```text
data/remote
```

Output:

```text
data/remote_256
```

This converts original masks to `256x256` while preserving semantic label ids.

## Step 2: Build Partition Graphs

Script:

```text
scripts/build_partition_graph_dataset.py
```

Input:

```text
data/remote_256
```

Output:

```text
data/remote_256_partition
```

This step converts each mask into a planar partition graph. Each connected semantic region is represented as a `face`.

This is the structural source for the current pipeline.

## Step 3: Geometry Approximator

Scripts and modules:

```text
partition_gen/geometry_approximator.py
scripts/build_geometry_approx_single.py
scripts/visualize_geometry_approx.py
```

Input:

```text
data/remote_256_partition/<split>/graphs/<sample>.json
```

Output:

```text
data/remote_256_geometry_approx_debug/<split>/graphs/<sample>_face<id>.json
```

The geometry approximator uses the old `base` primitive decomposition only as a geometry approximation tool.

Important point:

```text
base primitives are not kept as internal structure
```

Instead, all base pieces are unconditionally unioned back into one approximated polygon. The result is a simplified connected geometry that can be passed to constrained triangulation.

## Step 4: Constrained Triangulation

Module:

```text
partition_gen/convex_partition.py
```

The approximated polygon is triangulated using Shapely's constrained Delaunay triangulation.

The triangulation respects the polygon boundary and produces triangle primitives. Since every triangle is convex, this is always a valid initial convex partition.

## Step 5: Greedy Convex Merge

Module:

```text
partition_gen/convex_partition.py
```

After triangulation, adjacent pieces are repeatedly merged.

A merge is accepted only when:

- `A union B` is a single `Polygon`
- the result has no holes
- the result is convex, checked by comparing polygon area with convex hull area

The current greedy priority is based on shared boundary length, with merged area as a secondary ordering signal.

Output:

```text
data/remote_256_convex_partition_from_approx_debug/<split>/graphs/<sample>_face<id>.json
```

## Step 6: Visualization

Script:

```text
scripts/visualize_convex_partition.py
```

The visualization shows:

- source face
- constrained triangulation
- final greedy convex merge result

## Current Example Results

Using the geometry approximator before convex partition:

- `val/83 face16`: `34` triangles -> `15` convex primitives
- `val/10 face26`: `28` triangles -> `12` convex primitives

For comparison, raw pixel-level polygons produced hundreds of triangles on the same examples.

## What This Pipeline Does Not Yet Do

The current pipeline does not yet:

- run across all faces as a full dataset
- select between semantic explanation templates
- train a final generator
- render a full generated scene from generated structural programs

Those are follow-up stages.
