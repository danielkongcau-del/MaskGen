# Manual Target Split Generation

This document describes two prototype tools for reducing long manual-rule generator target sequences without changing the original manual-rule JSON semantics.

## Why Polygon Simplification Is Not Enough

The current manual-rule target is stable, but long samples are not caused only by polygon vertices. In val50, the longest sample also has many nodes and relations. Polygon simplification helps, but does not remove enough structure overhead by itself.

## Compact Tokenizer V1

`MANUAL_PARSE_GRAPH_COMPACT_V1` is a parallel tokenizer profile. It does not replace `encode_generator_target()`.

The compact tokenizer keeps the same parse graph semantics but changes how the graph is encoded:

- Node headers are shorter.
- `insert_object_group.children` directly encodes group membership.
- `contains` relations are not emitted one by one, because they are derivable from group children.
- `inserted_in`, `divides`, and `adjacent_to` are encoded as relation blocks.
- Relation endpoints use `inserted_in_container()` and `divides_target()`, so both new `container` / `target` and legacy `support` fields are supported.
- Polygon and convex atom geometry use the same low-level geometry encoding as the old manual tokenizer.

Use:

```bash
python scripts/benchmark_manual_compact_tokenizer.py \
  --target-root data/remote_256_generator_targets_manual_rule/val50_component_split/graphs \
  --output outputs/benchmarks/manual_compact_tokenizer_val50.jsonl \
  --summary-md outputs/benchmarks/manual_compact_tokenizer_val50.md
```

The expected benefit is lower relation and node-header overhead, especially when a sample has many `contains` relations. It is not expected to solve large polygon geometry by itself.

## Topology / Geometry Split

The split target prototype separates graph structure from node shape:

- `manual_parse_graph_topology_v1` keeps nodes, roles, labels, relations, and geometry references.
- `manual_parse_graph_geometry_v1` stores one renderable node geometry per target.

Topology targets do not contain polygon vertices, frames, or atoms. Renderable topology nodes keep `geometry_model` and `geometry_ref`, and the matching geometry target stores the actual frame and geometry payload.

Use:

```bash
python scripts/benchmark_manual_topology_geometry_split.py \
  --target-root data/remote_256_generator_targets_manual_rule/val50_component_split/graphs \
  --output outputs/benchmarks/manual_topology_geometry_split_val50.jsonl \
  --summary-md outputs/benchmarks/manual_topology_geometry_split_val50.md
```

Add `--write-split-samples --output-root <dir>` to write topology and geometry JSON samples.

## What The Split Does Not Do

This is not a renderer and does not compute `render_iou`.

This is not model training. It only creates benchmarkable target forms and token sequences.

This does not modify role specs, manual-rule explanation, global approximation, geometry approximation, or convex partitioning.

This does not simplify geometry. Geometry simplification can be benchmarked separately and later applied to per-node geometry targets if needed.

## Evaluation Criteria

For compact tokenizer:

- Old vs compact total tokens.
- Relation token reduction.
- Number of skipped `contains` relations.
- Longest remaining compact samples.

For topology / geometry split:

- Old total tokens.
- Topology token length.
- Maximum single geometry target length.
- `max_single_sequence_tokens = max(topology_tokens, geometry_tokens_max)`.
- Count of samples still above 4096 or 6144 tokens.

The practical goal is to prevent one image from becoming one very long sequence. If topology remains long, relation and node blocks need more compression. If a geometry target remains long, per-node geometry simplification or simpler geometry models should be evaluated next.
