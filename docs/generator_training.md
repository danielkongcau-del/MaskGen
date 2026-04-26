# Generator Training Plan

This document describes the first generator-training data path. The current goal is not to train a final model immediately; it is to build a stable `parse_graph` target dataset that can be rendered and validated.

## Current Target

The first generator target is the weak parse graph profile:

```text
weak_convex_face_atoms_v1
```

It contains:

- `label_group` nodes
- `semantic_face` nodes
- `convex_atom` nodes
- `label_group_contains` relations
- `atom_part_of` relations
- `face_adjacent` relations

This target avoids strong semantic role decisions such as `support_region`, `divider_region`, or `insert_object`. Those roles remain future compression layers.

## Dataset Build Pipeline

The batch builder runs:

```text
partition graph
  -> global approx partition
  -> explanation evidence
  -> weak explanation
  -> weak render validation
  -> generator target JSON
```

The output directory is:

```text
data/remote_256_generator_targets/<split>/graphs/<stem>.json
data/remote_256_generator_targets/<split>/manifest.jsonl
```

Each graph file is a `maskgen_generator_target_v1` JSON:

```json
{
  "format": "maskgen_generator_target_v1",
  "target_type": "parse_graph",
  "size": [256, 256],
  "parse_graph": {
    "nodes": [],
    "relations": [],
    "residuals": []
  },
  "metadata": {}
}
```

## CGAL First

For the first generator target dataset, the requested convex backend is CGAL:

```powershell
conda run -n lmf python scripts/build_generator_targets.py `
  --partition-root data/remote_256_partition `
  --split val `
  --output-root data/remote_256_generator_targets_cgal `
  --convex-backend cgal `
  --mask-root data/remote_256
```

The builder records the actual backend distribution per sample:

```json
{
  "convex_backend_requested": "cgal",
  "convex_backend_counts": {
    "cgal": 12,
    "cgal_bridge_cut": 3
  },
  "convex_failure_count": 0
}
```

If CGAL fails for a face, the sample is not silently treated as equivalent to a clean target. The manifest records `convex_failure_count`, `residual_count`, and `training_usable`.

## Training Usability

A sample is marked as training-usable only if:

```text
render_valid == true
weak_valid == true
convex_failure_count == 0
residual_count == 0
```

The first AR Transformer dataset should read `manifest.jsonl` and only train on:

```text
success == true
training_usable == true
```

Invalid or non-usable targets can still be written for debugging unless `--only-training-usable` is used.

## Training Sanitization

By default, `scripts/build_generator_targets.py` removes fields that should not be learned by the generator:

- `node.evidence`
- `node.source_face_id`
- `node.source_atom_id`
- `node.features`
- `relation.source_face_ids`
- `relation.arc_ids`
- `residual.face_ids`

For `semantic_face` nodes, source-boundary geometry is replaced by:

```json
{
  "geometry_model": "convex_atom_union",
  "geometry": {
    "atom_ids": ["atom_0", "atom_1"]
  }
}
```

The actual geometry is learned through `convex_atom` nodes. This avoids forcing the generator to emit source-specific arc ids.

Use `--keep-evidence-fields` only for debugging.

## Recommended First Runs

Smoke test:

```powershell
conda run -n lmf python scripts/build_generator_targets.py `
  --partition-root data/remote_256_partition `
  --split val `
  --output-root data/remote_256_generator_targets_cgal_smoke `
  --max-samples 5 `
  --convex-backend cgal `
  --mask-root data/remote_256
```

Full validation split:

```powershell
conda run -n lmf python scripts/build_generator_targets.py `
  --partition-root data/remote_256_partition `
  --split val `
  --output-root data/remote_256_generator_targets_cgal `
  --convex-backend cgal `
  --mask-root data/remote_256
```

Train smoke:

```powershell
conda run -n lmf python scripts/build_generator_targets.py `
  --partition-root data/remote_256_partition `
  --split train `
  --output-root data/remote_256_generator_targets_cgal `
  --max-samples 100 `
  --convex-backend cgal `
  --mask-root data/remote_256
```

## Next Step After Dataset Build

After the target dataset is stable, the next implementation step is a tokenizer:

```text
maskgen_generator_target_v1
  -> canonical token sequence
  -> AR Transformer training examples
```

The tokenizer must canonicalize node order, relation order, numeric coordinate quantization, and ID references. The neural generator should not train directly on raw JSON strings.

## Tokenization

The first tokenizer is implemented by:

```text
partition_gen/parse_graph_tokenizer.py
scripts/tokenize_generator_targets.py
```

It uses a fixed weak-parse-graph grammar:

```text
BOS
  WEAK_PARSE_GRAPH_V1
  SIZE H W
  LABEL_GROUPS ...
  FACES ...
    FACE label frame atoms...
  ADJ ...
  RESIDUALS ...
EOS
```

It intentionally does not train on raw JSON punctuation or source-specific evidence fields.

Example:

```powershell
conda run -n lmf python scripts/tokenize_generator_targets.py `
  --target-root data/remote_256_generator_targets_cdt_greedy `
  --split train `
  --output-root data/remote_256_generator_tokens_cdt_greedy
```

By default, only manifest rows with `training_usable == true` are tokenized.

Useful filtering:

```powershell
conda run -n lmf python scripts/tokenize_generator_targets.py `
  --target-root data/remote_256_generator_targets_cdt_greedy `
  --split train `
  --output-root data/remote_256_generator_tokens_cdt_greedy_4096 `
  --max-tokens 4096
```

This allows the first AR Transformer baseline to train only on short and medium-complexity samples.
