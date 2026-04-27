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

### `scripts/build_global_approx_partition_single.py`

Builds a full-image shared-arc approximation from one partition graph JSON.

It merges small partition edges into maximal boundary chains by incident face pair, runs the face-level geometry approximator, transfers owner-face simplified boundary segments into shared arcs when full-image validation allows it, and reconstructs every face ring from shared arc references.

### `scripts/visualize_global_approx_partition.py`

Visualizes one full-image shared-arc approximation.

It displays the original mask, shared arcs, reconstructed approximate faces, and an overlay for topology/planarity inspection.

### `scripts/build_regularized_global_approx_single.py`

Runs the global arc regularizer on one full-image shared-arc approximation JSON.

The default mode straightens only near-linear staircase-like arcs, keeps arc endpoints fixed, and accepts a replacement only if full-image validation still passes.

It also supports subsegment smoothing inside longer arcs, so local staircase artifacts can be removed without flattening the whole boundary.

Optional face-chain smoothing handles staircases that cross arc boundaries or junctions. It now applies to both outer rings and hole rings, which is important for white/road-like regions whose jagged edges appear as internal boundaries of a larger face.

Optional strip-face smoothing is experimental. It snaps a whole thin elongated face to a rotated rectangle and should only be used for high-confidence strip-like regions; complex road networks still need face-chain or future local strip-segment regularization.

### `scripts/visualize_global_arc_regularization.py`

Visualizes one before/after arc regularization pair.

It displays the original mask, pre-regularization arcs, post-regularization arcs, and highlighted changed arcs.

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

### `scripts/benchmark_convex_splitters.py`

Runs a face-level benchmark comparing the stable CDT + greedy baseline against the bridged convex partition splitter.

It writes one JSONL row per face and `cut_slit_scale`, including piece counts, validation IoU, bridge-search counts, backend metadata, and runtime.

### `scripts/summarize_convex_splitter_benchmark.py`

Summarizes a convex splitter benchmark JSONL file into Markdown or JSON.

The summary reports success/fallback rates, piece-count reductions, grouped statistics, worst IoU cases, and samples where the bridged splitter does not improve over the baseline.

### `scripts/export_convex_splitter_failures.py`

Rebuilds and visualizes benchmark rows that need inspection.

It exports cases with low IoU, fallback backend usage, rejected bridge sets, or no piece-count improvement.

### `scripts/build_explanation_evidence_single.py`

Builds one `maskgen_explanation_evidence_v1` file from a global approximation JSON.

It packages faces, arcs, adjacency, face features, arc features, and per-face convex atoms for the explainer.

### `scripts/build_explanation_single.py`

Builds one initial `maskgen_explanation_v1` and nested generator `parse_graph` target from an evidence JSON or global approximation JSON.

The current explainer uses binary label-pair relation scoring plus image-level label-role consistency before emitting support, divider, insert, and residual nodes.

### `scripts/visualize_explanation.py`

Visualizes one explanation JSON with source mask, evidence arcs, evidence faces, and final role-colored parse graph overlay.

### `scripts/analyze_label_pair_relations_single.py`

Runs the standalone binary label-pair relation analyzer on one evidence JSON.

It constructs two-class subscenes for adjacent label pairs, scores support-insert, support-divider, adjacent-support, and residual candidates, then writes diagnostics for each pair.

### `scripts/visualize_label_pair_relations.py`

Visualizes the standalone label-pair relation analysis, including source mask, preferred role by label, and selected pairwise decisions.

### `scripts/build_weak_explanation_single.py`

Builds one weak `maskgen_explanation_v1` using the `weak_convex_face_atoms_v1` profile.

It packs semantic faces, label groups, convex atom nodes, atom membership relations, and face adjacency without assigning support/divider/insert roles.

### `scripts/visualize_weak_explanation.py`

Visualizes one weak explanation with source mask, shared arcs, semantic faces, and convex atom overlays.

### `scripts/render_weak_explanation.py`

Renders and validates one weak explanation parse graph.

It unions each face's `convex_atom` nodes back into rendered face polygons, writes a rendered partition JSON, optionally writes a mask PNG, and reports per-face atom IoU plus full-image gap/overlap metrics.

### `scripts/benchmark_weak_explainer.py`

Runs a batch benchmark for the weak explainer.

It can start from partition graphs, global approximation JSONs, or evidence JSONs, then builds weak explanations, renders them, and writes one JSONL row per sample with validity, IoU, gap/overlap, atom count, relation count, and runtime metrics.

### `scripts/build_generator_targets.py`

Builds `maskgen_generator_target_v1` parse graph targets in batch.

It runs the current pipeline through global approximation, explanation evidence, weak explanation, and weak render validation, then writes one sanitized generator target JSON per sample plus a manifest. The default convex backend is CGAL, and each manifest row records actual backend usage, render validity, convex failures, residual count, and whether the sample is training-usable.

### `scripts/tokenize_generator_targets.py`

Tokenizes generator target parse graphs into fixed-grammar token sequences for autoregressive training.

It reads `data/remote_256_generator_targets_*<split>/manifest.jsonl`, keeps training-usable rows by default, writes `sequences.jsonl`, writes a fixed vocabulary, and records token length statistics. The tokenizer uses a structured weak-parse-graph grammar rather than raw JSON text.

### `scripts/train_manual_topology_ar.py`

Trains the topology-only manual parse-graph AR Transformer on `manual_parse_graph_topology_v1` token sequences.

It can optionally run sampling-based topology validation during eval intervals with `--topology-eval-samples`. When enabled, it writes `topology_eval_iter_<iter>.json`, logs compact topology metrics to `train_log.jsonl`, and saves `ckpt_best_topology_valid.pt` whenever unconstrained sampled semantic-valid rate improves. Grammar `valid_rate` checks token syntax; `semantic_valid_rate` additionally checks role attributes, insert-group children, relation endpoint roles, duplicate/self relations, and unused `REL_BLOCK_OTHER`.

It also computes topology structure targets from the training token set, or from explicit `--topology-target-*`
overrides, and saves `ckpt_best_topology_structure.pt` when the sampled valid rate plus structural-distribution
score improves.

### `scripts/evaluate_manual_topology_ar.py`

Samples a manual topology AR checkpoint and writes validity plus structural-distribution metrics.

It supports both unconstrained and constrained sampling. The constrained sampler enforces the token grammar plus topology semantics by default: insert-group children reserve future `ROLE_INSERT` nodes, relation endpoints are role-filtered, and duplicate/self relation pairs are masked. Optional `--count-prior-token-root` or `--count-prior-json` plus `--count-prior-weight` blends training-set count priors into `node_count`, insert-group `child_count`, `REL_BLOCK_DIVIDES` count, and `REL_BLOCK_ADJACENT_TO` count logits without changing the allowed token set. The summary includes grammar valid rate, semantic-valid rate, EOS count, node-count stats, role/label histograms, role-label histogram, relation means per valid sample, and invalid failure reasons.

### `scripts/summarize_weak_explainer_benchmark.py`

Summarizes a weak explainer benchmark JSONL into Markdown or JSON.

It reports success rates, render-valid rates, atom/face statistics, code-length proxy statistics, worst IoU samples, largest gap/overlap samples, and high atom-per-face samples.

### `scripts/export_weak_explainer_failures.py`

Exports weak explainer benchmark rows that need inspection.

It rebuilds selected samples, writes evidence/weak/render/validation JSONs, renders mask PNGs, and generates weak explanation visualizations for invalid renders, low-IoU cases, or gap/overlap cases.

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

### `partition_gen/global_approx_partition.py`

Implements the full-image shared-arc approximation branch.

It extracts maximal boundary chains, builds owner-face boundary candidates from the existing geometry approximator, greedily accepts only globally valid boundary transfers, reconstructs all face polygons from shared arc references, and reports adjacency/gap/overlap validation metrics.

### `partition_gen/global_arc_regularizer.py`

Implements topology-safe smoothing for full-image shared arcs.

It regularizes staircase-like arcs after global approximation, rebuilds all face rings, and accepts only changes that preserve full-image validity.

The default path searches straightenable full arcs and straightenable subsegments. Optional face-chain smoothing, strip-face smoothing, and polyline smoothing are exposed for more aggressive cleanup experiments.

### `partition_gen/explanation_evidence.py`

Builds the evidence layer consumed by the explainer.

It computes face/arc features, adjacency records, convex atoms, validation, and statistics while keeping global approximation and convex splitting as separate upstream modules.

### `partition_gen/pairwise_relation_explainer.py`

Implements the standalone binary label-pair relation analyzer.

It removes unrelated labels conceptually, tests candidate explanations for each adjacent label pair, uses convex partition code length as geometry evidence, and records why a pair is treated as support-insert, support-divider, adjacent-support, or residual.

### `partition_gen/explainer.py`

Implements the initial explanation builder.

It combines face-local role candidates, binary label-pair relation evidence, and image-level label-role consistency to produce `maskgen_explanation_v1` and the nested `maskgen_generator_target_v1` parse graph.

### `partition_gen/weak_explainer.py`

Implements the weak explanation builder.

It uses evidence faces and existing convex partitions to produce a structural parse graph with `label_group`, `semantic_face`, and `convex_atom` nodes. This path avoids strong semantic role decisions and is intended as the safer first generator target.

### `partition_gen/weak_parse_graph_renderer.py`

Renders and validates the weak parse graph profile.

It converts local convex atom coordinates back to world-space polygons, unions atoms per semantic face, compares the result against evidence face geometry, and can rasterize the rendered faces to a mask for approximate pixel-level inspection.

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
