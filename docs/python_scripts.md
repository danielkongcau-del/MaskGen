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

It supports both unconstrained and constrained sampling. The constrained sampler enforces the token grammar plus topology semantics by default: insert-group children reserve future `ROLE_INSERT` nodes, relation endpoints are role-filtered, and duplicate/self relation pairs are masked. Optional `--count-prior-token-root` or `--count-prior-json` plus `--count-prior-weight` blends training-set count priors into `node_count`, insert-group `child_count`, `REL_BLOCK_DIVIDES` count, and `REL_BLOCK_ADJACENT_TO` count logits without changing the allowed token set. Experimental `--complexity-level` applies a single monotonic high-count tilt to those same count decisions for diagnostics. The summary includes grammar valid rate, semantic-valid rate, EOS count, node-count stats, role/label histograms, role-label histogram, relation means per valid sample, and invalid failure reasons.

### `scripts/train_manual_geometry_ar.py`

Trains the per-node manual geometry AR Transformer.

It reuses the same AR Transformer and manual split token dataloader as topology training. Use the default `--sequence-kind geometry` for v0 role/label/model-conditioned geometry from `geometry_sequences.jsonl`, `--sequence-kind conditioned_geometry` for topology-conditioned v1 rows, `--sequence-kind oracle_frame_geometry` for oracle-frame local shape rows, or `--sequence-kind layout` for absolute layout rows. Optional `--geometry-eval-samples` runs constrained forced-prefix sampling during training and validates extracted payloads.

### `scripts/evaluate_manual_geometry_ar.py`

Samples a manual geometry AR checkpoint and writes validity metrics for generated `MANUAL_GEOMETRY_V1` sequences.

By default it uses constrained sampling with forced prefixes from a geometry token root, then validates decode/re-encode roundtrips, EOS rate, polygon counts, hole counts, atom counts, point counts, and role/label/model histograms. If no token root is available, it can sample from an explicit `--prefix-role`, `--prefix-label`, and `--prefix-geometry-model`. With `--sequence-kind conditioned_geometry` or `oracle_frame_geometry`, it samples topology-conditioned prefixes and evaluates extracted `MANUAL_GEOMETRY_V1` payloads. With `--sequence-kind layout`, it samples/evaluates `MANUAL_LAYOUT_V1` rows.

### `scripts/tokenize_manual_conditioned_geometry_dataset.py`

Builds topology-conditioned geometry token sequences from a topology/geometry split target dataset.

Each row contains a full `MANUAL_TOPOLOGY_V1` context, a `TARGET_NODE` index, and that node's `MANUAL_GEOMETRY_V1` payload. The row includes `loss_start_index` so training can mask condition-token loss and optimize only the generated geometry suffix.

### `scripts/tokenize_manual_oracle_frame_geometry_dataset.py`

Builds oracle-frame local geometry token sequences.

Each row contains topology context, target node index, and a forced geometry prefix through true `FRAME`. The row's `loss_start_index` points at `POLYS` or `ATOMS`, so the model learns only local shape and does not predict frame.

### `scripts/tokenize_manual_layout_dataset.py`

Builds topology-conditioned absolute layout token sequences.

Each row contains topology context followed by one `MANUAL_LAYOUT_V1` target for all renderable `geometry_ref` nodes in topology node order. The target contains absolute `FRAME` bins only.

### `scripts/tokenize_manual_relative_layout_dataset.py`

Builds topology-conditioned relative layout token sequences.

Each row contains topology context followed by one `MANUAL_REL_LAYOUT_V1` target. Root support nodes use absolute frame tokens; insert objects and dividers use deterministic relation-derived anchors and relative frame tokens. `adjacent_to` remains context only and is not used as a default anchor.

### `scripts/attach_generated_geometry_to_topology_samples.py`

Attaches geometry generated by a manual geometry AR checkpoint to generated topology samples.

It decodes semantic-valid `MANUAL_TOPOLOGY_V1` rows, finds renderable nodes with `geometry_ref`, samples a `MANUAL_GEOMETRY_V1` sequence for each node using forced `(role, label, geometry_model)` prefixes, decodes valid geometry payloads, and writes full `parse_graph` JSONs plus a manifest and summary. This is the learned-geometry counterpart to placeholder geometry attachment.

### `scripts/attach_conditioned_generated_geometry_to_topology_samples.py`

Attaches topology-conditioned generated geometry to generated topology samples.

It decodes semantic-valid `MANUAL_TOPOLOGY_V1` rows and, for each renderable node with `geometry_ref`, builds a full topology-conditioned prefix containing the decoded topology and target node index before constrained geometry sampling. This is the v1 learned-geometry attachment path.

### `scripts/attach_oracle_frame_geometry_to_split_targets.py`

Attaches generated local geometry to true split topology while preserving true frames.

It uses an oracle-frame geometry checkpoint to sample only the local `POLYS`/`ATOMS` suffix, writes the true split `frame` back into decoded geometry targets, and reconstructs full parse graphs for spatial audit and visualization.

### `scripts/train_manual_layout_ar.py`

Trains the absolute layout AR generator.

This is a wrapper around `train_manual_geometry_ar.py` with `--sequence-kind layout` and default output root `outputs/manual_layout_ar`.

### `scripts/evaluate_manual_layout_ar.py`

Evaluates an absolute layout AR checkpoint.

This is a wrapper around `evaluate_manual_geometry_ar.py` with `--sequence-kind layout`, reporting layout validity and frame MAE when source token rows include targets.

### `scripts/attach_layout_ar_to_split_targets.py`

Attaches layout-AR predicted frames to true split topology using true local shapes.

Use this to isolate layout quality without local shape noise before testing generated topology.

### `scripts/attach_layout_ar_to_topology_samples.py`

Attaches layout-AR predicted frames plus placeholder local shapes to generated topology samples.

It decodes generated topology rows, samples constrained `MANUAL_LAYOUT_V1`, attaches predicted frames, and retrieves local shapes by `(role, label, geometry_model)` from a split dataset.

### `scripts/attach_retrieved_layout_to_split_targets.py`

Attaches nearest-neighbor retrieved train layouts to split targets using true local shapes.

It builds a retrieval library from `--library-split-root`, scores topology signatures against each query topology from `--split-root`, copies frames from the nearest train layout by `role` / `label` / `geometry_model` order, and falls back to train-set median frames when a target node has no match. Use this as a layout retrieval baseline before training another layout generator.

### `scripts/attach_retrieved_layout_oracle_frame_geometry_to_split_targets.py`

Attaches generated local geometry to split topology using retrieved train-layout frames.

It first retrieves a nearest-neighbor train layout for each query topology, then uses those retrieved frames as the forced `FRAME` prefix for the oracle-frame geometry checkpoint. The geometry model samples only local `POLYS`/`ATOMS`, so this tests the proposed pipeline of generated topology, retrieved layout, and generated local shape without asking the geometry model to invent absolute frame placement.

### `scripts/attach_retrieved_layout_oracle_frame_geometry_to_topology_samples.py`

Attaches retrieved-layout frames and generated local geometry to generated topology samples.

It decodes semantic-valid `MANUAL_TOPOLOGY_V1` sample rows, retrieves a nearest-neighbor train layout for each generated topology, uses the retrieved frames as oracle-frame geometry prefixes, and samples only the local shape suffix from the geometry checkpoint. This is the end-to-end diagnostic path for generated topology plus retrieval layout plus generated local geometry.

### `scripts/train_manual_layout_residual.py`

Trains a residual frame predictor on top of nearest-neighbor retrieved layouts.

It builds one supervised example per renderable geometry node: retrieve a similar train layout, map its frame to the query node, then learn normalized `(dx, dy, dlog_scale, dtheta)` from the retrieved frame to the true frame. By default training excludes same-stem retrieval so the model learns to correct a neighbor layout instead of memorizing the source row.

### `scripts/evaluate_manual_layout_residual.py`

Evaluates a retrieved-layout residual checkpoint.

The report compares raw retrieval baseline MAE against residual-corrected MAE for origin, scale, and orientation, with mapping-mode and retrieval-score diagnostics. This is the main check for whether the "retrieve a real layout, then adjust it" path is actually improving layout quality.

Residual-decoded scale is clamped during evaluation and attach. The clamp uses both the tokenizer's legal scale range and a geometry-aware local-bbox side limit, so large local shapes cannot produce oversized final bboxes even when the raw residual predicts a legal but unsafe scale. The report includes `scale_out_of_range_count`, `geometry_scale_clamped_count`, raw/clamped scale stats, and raw/clamped bbox-huge counts so scale explosions are visible instead of being hidden by high visible-polygon rates.

### `scripts/attach_retrieved_residual_layout_to_split_targets.py`

Attaches residual-corrected retrieved layouts to split targets using true local shapes.

Use this to isolate residual frame quality before reintroducing generated local geometry. It writes full parse-graph targets with predicted corrected `frame` values and true `geometry`/`atoms` payloads from the split dataset, so the output can be audited with `audit_manual_parse_graph_targets.py` and `audit_manual_parse_graph_spatial.py`.

### `scripts/attach_retrieved_residual_layout_oracle_frame_geometry_to_split_targets.py`

Attaches generated local geometry to split topology using residual-corrected retrieved-layout frames.

It retrieves the nearest train layout, predicts a residual frame correction per renderable node, uses the refined frame as the forced oracle-frame geometry prefix, then computes the generated local bbox and clamps the final frame by that bbox before writing the full parse graph. This is the split-target diagnostic for the full "retrieve layout, adjust frame, generate local shape, clamp by generated bbox" chain.

### `scripts/attach_retrieved_residual_layout_oracle_frame_geometry_to_topology_samples.py`

Attaches residual-corrected retrieved-layout frames and generated local geometry to generated topology samples.

It decodes semantic-valid generated topology rows, retrieves a similar train layout, applies the residual frame predictor, samples local `POLYS`/`ATOMS` from the oracle-frame geometry checkpoint, clamps the final frame with the generated local bbox, and writes full parse-graph targets plus per-node frame/refinement/clamp diagnostics.

### `scripts/train_manual_relative_layout_ar.py`

Trains the relative layout AR generator.

This is a wrapper around `train_manual_geometry_ar.py` with `--sequence-kind relative_layout`. It uses deterministic anchors during constrained sampling so the model predicts only absolute/global or relative frame values.

### `scripts/evaluate_manual_relative_layout_ar.py`

Evaluates a relative layout AR checkpoint.

It samples constrained `MANUAL_REL_LAYOUT_V1` sequences, reconstructs absolute frames from relative rows, and reports layout validity plus frame MAE against source token rows. The summary also includes numeric diagnostics for sampled frames: scale stats, origin outside ratio, unit-bbox visible ratio, and `dx` / `dy` / `log_scale_ratio` distributions.

Pass `--safe-relative-layout` to mask relative `dx` / `dy` / `log_scale_ratio` token ranges during sampling for diagnostic runs.

### `scripts/attach_relative_layout_ar_to_split_targets.py`

Attaches relative-layout-AR predicted frames to true split topology using true local shapes.

Use this to isolate relative layout quality before introducing generated topology or placeholder shape noise. Pass `--safe-relative-layout` to use the same safe token masks and clamp decoded frames before attaching; `summary.json` records the safety config and clamp counts.

### `scripts/attach_relative_layout_ar_to_topology_samples.py`

Attaches relative-layout-AR predicted frames plus placeholder local shapes to generated topology samples.

It decodes generated topology rows, samples constrained `MANUAL_REL_LAYOUT_V1`, reconstructs absolute frames from relative anchors, and retrieves local shapes by `(role, label, geometry_model)` from a split dataset. It also supports `--safe-relative-layout` for the diagnostic safe sampler path.

### `scripts/train_manual_layout_frame.py`

Trains the v2a topology-conditioned layout/frame predictor.

It reads topology/geometry split targets, builds one supervised example per renderable geometry node, and predicts quantized `FRAME` fields (`origin_x`, `origin_y`, `scale`, `orientation`) with four classification heads from engineered topology features.

### `scripts/evaluate_manual_layout_frame.py`

Evaluates a layout/frame predictor checkpoint on a split target root.

It reports CE loss, per-head bin accuracy, prediction/target bin histograms, origin/scale/orientation MAE, and role-wise MAE metrics for diagnosing whether layout prediction is improving before local shape generation is reintroduced. Pass `--baseline-train-split-root` to include a `(role, label)` mean-frame baseline in the same output.

### `scripts/overfit_manual_layout_frame.py`

Runs a small-sample overfit diagnostic for the layout/frame MLP.

It trains and evaluates on the same first `--max-examples` examples from a split root. This is intended to catch broken labels, loss wiring, or dequantization before interpreting full validation performance. A healthy model should drive train-set origin MAE down on a small subset.

### `scripts/train_manual_layout_frame_regression.py`

Trains the v2b continuous regression layout/frame baseline.

It uses the same engineered topology features as `train_manual_layout_frame.py`, but predicts normalized `(origin_x, origin_y, scale, sin(theta), cos(theta))` with SmoothL1 loss. This is diagnostic only; keep it only if validation origin MAE clearly beats the `(role, label)` mean baseline.

### `scripts/attach_layout_frame_to_split_targets.py`

Attaches predicted frames to real split topology targets while preserving true local geometry shapes.

Use this to isolate layout quality: the output parse graphs use model-predicted `frame` values and true `geometry`/`atoms` payloads from the split dataset, then can be checked with the spatial audit and visualization scripts.

### `scripts/attach_layout_frame_to_topology_samples.py`

Attaches predicted frames plus placeholder local shapes to generated topology samples.

It decodes generated topology rows, predicts a frame for each renderable `geometry_ref` node, retrieves a local shape by `(role, label, geometry_model)` from a split dataset, and writes full parse-graph targets for downstream spatial audit and visualization.

### `scripts/attach_placeholder_geometry_to_topology_samples.py`

Attaches retrieved real geometry targets to generated manual topology samples as a downstream smoke test before a learned geometry generator exists.

It decodes semantic-valid `MANUAL_TOPOLOGY_V1` sample rows into topology targets, samples placeholder geometry from a manual split dataset by `(role, label, geometry_model)` with fallbacks, and writes full `parse_graph` JSONs plus a manifest and summary. The output is diagnostic, not final generation quality, because geometry is retrieved independently of the generated topology.

### `scripts/audit_manual_parse_graph_targets.py`

Audits full manual `parse_graph` targets, including placeholder-geometry outputs.

It resolves a single JSON, a `graphs/` directory, or an output root with `manifest.jsonl`, then checks whether each target can be re-encoded by both the legacy manual tokenizer and compact manual tokenizer. It also reports missing renderable geometry payloads, node/relation statistics, token-length statistics, and role/label histograms.

### `scripts/audit_manual_parse_graph_spatial.py`

Audits spatial placement of renderable polygon nodes in full manual `parse_graph` targets.

It uses the same local-to-world transform as the visualization script, then reports whether polygon bboxes intersect the canvas, whether origins or bbox centers are near edges/corners, bbox size statistics, and origin/role/label histograms. This is useful for diagnosing geometry generators that are structurally valid but render mostly blank or collapsed into corners.

### `scripts/visualize_manual_parse_graph_target.py`

Renders full manual `parse_graph` targets to PNG.

It draws renderable `polygon_code` nodes by applying each node's `frame` transform to local polygon vertices. This is intended for quick qualitative checks of placeholder-geometry outputs and future geometry-generator outputs.

### `scripts/materialize_manual_split_targets.py`

Rebuilds full manual `parse_graph` targets from a topology/geometry split manifest.

It reads `topology_path` plus `geometry_paths`, attaches the true split `frame` and local geometry payloads back onto topology nodes with `geometry_ref`, then writes a normal `graphs/` plus `manifest.jsonl` output root. Use this before spatial audit when the source is `data/remote_256_generator_targets_manual_split_full/{train,val}` rather than a full target directory.

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
