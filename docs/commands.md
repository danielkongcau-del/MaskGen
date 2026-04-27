# Commands

Commands assume PowerShell from the repository root.

The expected conda environment is:

```powershell
conda run -n lmf ...
```

## Prepare `256x256` Data

```powershell
conda run -n lmf python scripts/prepare_remote_256.py
```

Output:

```text
data/remote_256
```

## Build Partition Graph Dataset

```powershell
conda run -n lmf python scripts/build_partition_graph_dataset.py `
  --input-root data/remote_256 `
  --output-root data/remote_256_partition
```

Output:

```text
data/remote_256_partition
```

## Build Geometry Approximation For One Face

Example:

```powershell
conda run -n lmf python scripts/build_geometry_approx_single.py `
  --split val `
  --stem 83 `
  --face-id 16 `
  --output data/remote_256_geometry_approx_debug/val/graphs/83_face16.json
```

Output:

```text
data/remote_256_geometry_approx_debug/val/graphs/83_face16.json
```

## Visualize Geometry Approximation

```powershell
conda run -n lmf python scripts/visualize_geometry_approx.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_geometry_approx.png
```

## Build Full-Image Shared-Arc Approximation

This builds a global planar approximation where adjacent faces reference the same maximal boundary arcs. Shared arcs are transferred from one face-level geometry approximation when the replacement keeps the full-image map valid.

```powershell
conda run -n lmf python scripts/build_global_approx_partition_single.py `
  --partition-graph data/remote_256_partition/val/graphs/83.json `
  --output outputs/visualizations/global_approx_val83.json `
  --face-simplify-tolerance 1.5
```

## Visualize Full-Image Shared-Arc Approximation

```powershell
conda run -n lmf python scripts/visualize_global_approx_partition.py `
  --global-json outputs/visualizations/global_approx_val83.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/global_approx_val83.png
```

## Regularize Full-Image Arcs

This straightens staircase-like shared arcs while keeping arc endpoints fixed and accepting only replacements that pass full-image validation.

```powershell
conda run -n lmf python scripts/build_regularized_global_approx_single.py `
  --global-json outputs/visualizations/global_approx_val37_risk.json `
  --partition-graph data/remote_256_partition/val/graphs/37.json `
  --output outputs/visualizations/global_approx_val37_regularized.json `
  --simplify-tolerance 1.25 `
  --max-distance 1.25
```

More aggressive cleanup, allowing larger local semantic boundary movement:

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

More aggressive cleanup when small semantic boundary shifts are acceptable:

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

## Visualize Arc Regularization

```powershell
conda run -n lmf python scripts/visualize_global_arc_regularization.py `
  --before-json outputs/visualizations/global_approx_val37_risk.json `
  --after-json outputs/visualizations/global_approx_val37_regularized.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 37 `
  --output outputs/visualizations/global_approx_val37_regularized.png
```

## Build Convex Partition From Geometry Approximation

```powershell
conda run -n lmf python scripts/build_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json
```

Output:

```text
data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json
```

## Build Bridged Convex Partition From Geometry Approximation

This branch records bridge candidates for holes and then uses CGAL when available. Simple polygons run directly through `CGAL::optimal_convex_partition_2`; polygons with holes use `epsilon_slit_snap_v1` to cut holes open before CGAL. If CGAL is not available, it falls back to the current CDT + greedy convex merge and marks `optimal=false`.

Build the CGAL CLI backend first if you want true optimal partitioning for simple polygons without holes:

```powershell
cmd /c 'call "D:\Microsoft Visual Studio\VC\Auxiliary\Build\vcvars64.bat" && cmake -Wno-dev -S tools -B build\cgal_tools -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake && cmake --build build\cgal_tools --config Release'
```

```powershell
conda run -n lmf python scripts/build_bridged_convex_partition_from_approx_single.py `
  --approx-json data/remote_256_geometry_approx_debug/val/graphs/83_face16.json `
  --output data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json
```

## Visualize Bridged Convex Partition

```powershell
conda run -n lmf python scripts/visualize_bridged_convex_partition.py `
  --partition-json data/remote_256_bridged_convex_partition_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_bridged_convex_partition.png
```

## Benchmark Convex Splitters

Compare the CDT + greedy baseline against the bridged CGAL splitter on geometry approximation JSON files.

```powershell
conda run -n lmf python scripts/benchmark_convex_splitters.py `
  --approx-root data/remote_256_geometry_approx_debug `
  --split val `
  --output outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --backend auto `
  --cut-slit-scales 1e-7 1e-6 1e-5
```

Summarize the JSONL benchmark:

```powershell
conda run -n lmf python scripts/summarize_convex_splitter_benchmark.py `
  --input outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output outputs/benchmarks/convex_splitter_benchmark_val.md
```

Export visual checks for low-IoU, fallback, rejected-bridge, or non-improving cases:

```powershell
conda run -n lmf python scripts/export_convex_splitter_failures.py `
  --benchmark-jsonl outputs/benchmarks/convex_splitter_benchmark_val.jsonl `
  --output-dir outputs/visualizations/convex_splitter_failures
```

## Visualize Convex Partition

```powershell
conda run -n lmf python scripts/visualize_convex_partition.py `
  --partition-json data/remote_256_convex_partition_from_approx_debug/val/graphs/83_face16.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 83 `
  --output outputs/visualizations/face83_16_convex_partition_from_approx.png
```

## Raw Polygon Convex Partition

This bypasses the geometry approximator and runs directly on a face from `data/remote_256_partition`.

```powershell
conda run -n lmf python scripts/build_convex_partition_single.py `
  --partition-root data/remote_256_partition `
  --split val `
  --stem 83 `
  --face-id 16 `
  --output data/remote_256_convex_partition_debug/val/graphs/83_face16.json
```

This is useful for comparison, but not the current preferred path.

## Manual Topology AR Evaluation

Train the topology-only manual parse-graph generator with optional sampling-based topology validation. When
`--topology-eval-samples` is positive, each eval interval samples unconstrained topology sequences, writes
`topology_eval_iter_<iter>.json`, logs `topology_eval` rows, and saves `ckpt_best_topology_valid.pt` when the
sampled semantic-valid rate improves. Grammar `valid_rate` checks token syntax; `semantic_valid_rate` additionally
checks role attributes, insert-group children, relation endpoint roles, duplicate/self relations, and unused
`REL_BLOCK_OTHER`. It also estimates target structure means from the training token set and saves
`ckpt_best_topology_structure.pt` when the validity-plus-structure score improves.

```powershell
conda run -n lmf python scripts/train_manual_topology_ar.py `
  --train-token-root data/remote_256_generator_tokens_manual_split_full/train `
  --val-token-root data/remote_256_generator_tokens_manual_split_full/val `
  --output-dir outputs/manual_topology_ar `
  --run-name topology_v1 `
  --topology-eval-samples 100 `
  --topology-eval-temperature 0.7 `
  --topology-eval-top-k 50
```

Evaluate a saved checkpoint and write both validity and structural-distribution metrics:

```powershell
conda run -n lmf python scripts/evaluate_manual_topology_ar.py `
  --checkpoint outputs/manual_topology_ar/topology_v1/ckpt_iter_5000.pt `
  --output-json outputs/manual_topology_ar/topology_v1/eval_iter5000_t0.7.json `
  --summary-md outputs/manual_topology_ar/topology_v1/eval_iter5000_t0.7.md `
  --output-samples outputs/manual_topology_ar/topology_v1/samples_iter5000_t0.7.jsonl `
  --num-samples 100 `
  --temperature 0.7 `
  --top-k 50
```

Use grammar-constrained evaluation for deployment-style topology sampling. The default `--max-nodes` is 512,
which covers the current long-tail topology training samples.

```powershell
conda run -n lmf python scripts/evaluate_manual_topology_ar.py `
  --checkpoint outputs/manual_topology_ar/topology_v1/ckpt_iter_5000.pt `
  --output-json outputs/manual_topology_ar/topology_v1/eval_iter5000_constrained_t0.7.json `
  --summary-md outputs/manual_topology_ar/topology_v1/eval_iter5000_constrained_t0.7.md `
  --output-samples outputs/manual_topology_ar/topology_v1/samples_iter5000_constrained_t0.7.jsonl `
  --num-samples 100 `
  --temperature 0.7 `
  --top-k 50 `
  --constrained
```

## Historical Commands

Training and sampling commands for topology, geometry, boundary, and pair-boundary models still exist under `scripts/`.

They are documented in [python_scripts.md](python_scripts.md), but they are not the current recommended workflow.
