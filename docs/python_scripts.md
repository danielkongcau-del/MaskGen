# Python Scripts

This document lists the Python entrypoints and important modules currently present in the repository.

## Current Mainline Scripts

### `scripts/prepare_remote_256.py`

Builds the `256x256` dataset from the original remote-sensing data.

### `scripts/build_partition_graph_dataset.py`

Converts `256x256` masks into partition graph JSON files.

### `scripts/build_geometry_approx_single.py`

Runs the geometry approximator for one selected face.

It uses the old base primitive decomposition as a geometry approximation tool, unions all base pieces back together, and writes a simplified polygon payload.

### `scripts/visualize_geometry_approx.py`

Visualizes one geometry approximator output.

### `scripts/build_convex_partition_from_approx_single.py`

Runs constrained triangulation and greedy convex merge on one geometry approximator output.

This is the stable CDT + greedy baseline for convex primitive extraction.

### `scripts/build_bridged_convex_partition_from_approx_single.py`

Builds the experimental bridged convex partition from one geometry approximator output.

It records bridge candidates and selected bridges for holes. If no CGAL optimal backend is available, it falls back to the current CDT + greedy convex merge and marks the result as non-optimal.

### `scripts/build_convex_partition_single.py`

Runs constrained triangulation and greedy convex merge directly on a face from `data/remote_256_partition`.

Useful for comparison against the approximator path.

### `scripts/visualize_convex_partition.py`

Visualizes one convex partition result, including source face, triangulation, and final convex primitives.

### `scripts/visualize_bridged_convex_partition.py`

Visualizes one bridged convex partition result, including selected bridges, boundary walk metadata, and final convex pieces.

## Supporting Or Experimental Dataset Scripts

### `scripts/build_cdt_partition_dataset.py`

Builds a simplified partition dataset intended for CDT experiments.

This is not the current preferred path, but remains useful for comparison.

### `scripts/build_dual_graph_dataset.py`

Builds face-level dual graphs from partition graph JSON files.

Used by earlier topology and boundary experiments.

### `scripts/inspect_dual_graph_stats.py`

Computes statistics for dual graph datasets and fits quantizers for sparse autoregressive experiments.

### `scripts/build_geometry_dataset.py`

Builds simplified polygon targets for the earlier geometry decoder branch.

### `scripts/build_primitive_dataset.py`

Builds old primitive decomposition outputs for faces.

The current mainline still uses the underlying base decomposition logic, but this dataset-building script is historical.

### `scripts/build_shared_boundary_dataset.py`

Builds shared-boundary raster and segment targets for boundary prediction experiments.

## Historical Training And Sampling Scripts

### `scripts/train_topology.py`

Trains the sparse autoregressive dual-graph topology model.

### `scripts/sample_topology.py`

Samples dual-graph topologies from a trained topology model.

### `scripts/smoke_test_topology_model.py`

Runs a forward-pass smoke test for the topology model.

### `scripts/train_geometry.py`

Trains the graph-conditioned face geometry decoder.

### `scripts/sample_geometry.py`

Predicts simplified face polygons from dual graphs.

### `scripts/eval_geometry.py`

Evaluates geometry decoder outputs.

### `scripts/smoke_test_geometry_model.py`

Runs a forward-pass smoke test for the geometry decoder.

### `scripts/train_boundary.py`

Trains the graph-conditioned global boundary predictor.

### `scripts/sample_boundary.py`

Predicts shared boundaries from dual graphs and renders masks.

### `scripts/eval_boundary.py`

Evaluates global boundary prediction.

### `scripts/train_boundary_pairs.py`

Trains the pair-level boundary predictor.

### `scripts/sample_boundary_pairs.py`

Predicts pair-level boundary masks and renders outputs.

### `scripts/eval_boundary_pairs.py`

Evaluates pair-level boundary prediction.

### `scripts/render_geometry_masks.py`

Renders predicted geometry JSON files into `256x256` mask PNGs.

## Historical Visualization Scripts

### `scripts/visualize_face_primitives.py`

Visualizes old primitive decompositions for one face.

### `scripts/visualize_composite_patch_scores.py`

Visualizes patch acceptance scores from the old composite primitive branch.

### `scripts/visualize_face_and_repair.py`

Visualizes face partitions and raster repair behavior.

### `scripts/visualize_pair_boundary.py`

Visualizes pair-boundary prediction output.

### `scripts/visualize_pair_features.py`

Visualizes geometric features used by the pair-boundary model.

## Current Mainline Modules

### `partition_gen/geometry_approximator.py`

Implements the geometry approximator used before CDT.

### `partition_gen/convex_partition.py`

Implements constrained triangulation and greedy convex merging.

### `partition_gen/bridged_convex_partition.py`

Implements the experimental bridged convex partition framework. It keeps the geometry approximator fixed, enumerates bridge candidates for holes, dispatches simple no-hole polygons to the CGAL CLI, and uses `epsilon_slit_snap_v1` to run CGAL on cut-open polygons with holes when available. If CGAL is unavailable, it uses the explicit non-optimal fallback.

### `tools/optimal_convex_partition_cli.cpp`

Standalone C++ CLI built with CGAL. It reads a JSON file with an `outer` simple polygon boundary, runs `CGAL::optimal_convex_partition_2`, and writes convex pieces as JSON. It is currently only for simple polygons without holes.

### `partition_gen/primitive_decomposition.py`

Implements the old primitive decomposition logic. The current geometry approximator uses its base decomposition and union operation.

### `partition_gen/dual_graph.py`

Provides JSON loading and polygon helpers for partition and dual graph structures.

### `partition_gen/cdt_partition.py`

Provides polygon simplification helpers for CDT-oriented experiments.

## Historical Modules

### `partition_gen/ar_dataset.py`

Dataset and batching logic for sparse autoregressive dual-graph generation.

### `partition_gen/topology_training.py`

Loss computation for the topology model.

### `partition_gen/geometry_dataset.py`

Dataset and batching logic for the geometry decoder branch.

### `partition_gen/geometry_training.py`

Loss computation for the geometry decoder branch.

### `partition_gen/geometry_render.py`

Renders predicted face geometry into raster masks.

### `partition_gen/boundary_dataset.py`

Dataset logic for global and pair-level boundary experiments.

### `partition_gen/boundary_training.py`

Loss computation for global boundary prediction.

### `partition_gen/pair_boundary_training.py`

Loss computation for pair-level boundary prediction.

### `partition_gen/joint_render.py`

Renders masks from boundary predictions using joint fill logic.

### `partition_gen/pair_render.py`

Renders masks from pair-level boundary masks.

## Historical Model Modules

### `partition_gen/models/topology_transformer.py`

Autoregressive transformer for dual-graph topology generation.

### `partition_gen/models/geometry_decoder.py`

Graph-conditioned decoder for face polygon geometry.

### `partition_gen/models/boundary_predictor.py`

Graph-conditioned global boundary predictor.

### `partition_gen/models/pair_boundary_predictor.py`

Graph-conditioned pair-level boundary predictor.
