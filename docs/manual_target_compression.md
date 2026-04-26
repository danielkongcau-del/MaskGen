# Manual Target Compression

This document describes the offline compression diagnostics for manual-rule generator targets. The tools in this layer do not change `manual_rule_explainer`, the role spec, global approximation, geometry approximation, convex partitioning, or the tokenizer contract used by existing data.

## Purpose

Manual-rule targets are stable parse graphs, but their token length is often dominated by geometry detail. The first compression stage is therefore diagnostic:

- Measure where tokens are spent.
- Test training-only polygon simplification profiles.
- Compare token reduction against geometry validity and area error.
- Keep original targets unchanged.

The simplified targets are experimental training artifacts written to a separate output directory.

## Token Attribution

Use:

```bash
python scripts/analyze_manual_target_token_lengths.py \
  --target-root data/remote_256_generator_targets_manual_rule/val/graphs \
  --output outputs/benchmarks/manual_target_token_stats.jsonl \
  --summary-md outputs/benchmarks/manual_target_token_stats_summary.md \
  --top-k 50
```

The analyzer reports:

- `total_tokens`, computed directly with `encode_generator_target`.
- Node counts, renderable/non-renderable counts, and reference-only counts.
- Polygon, convex atom, node-header, relation, and residual token estimates.
- Polygon vertex, component, and hole statistics.
- Longest nodes by estimated token count.
- `attribution_gap`, which should be `0` for manual-rule targets covered by the current attribution rules.

## Simplification Profiles

Use:

```bash
python scripts/benchmark_manual_target_simplification.py \
  --target-root data/remote_256_generator_targets_manual_rule/val/graphs \
  --output-root outputs/benchmarks/manual_target_simplify \
  --profiles light medium aggressive \
  --max-samples 200
```

Add `--write-simplified-samples` only when you want the simplified JSON files written to `output-root/<profile>/graphs`.

Profiles:

| profile | pixel tolerance | max ring vertices |
| --- | ---: | ---: |
| `none` | `0.0` | none |
| `light` | `0.25` | `64` |
| `medium` | `0.5` | `48` |
| `aggressive` | `1.0` | `32` |

The tolerance is interpreted in pixel units and converted through each node's `frame.scale` before simplifying local coordinates.

## What Gets Simplified

Simplification only touches nodes that are:

- `renderable != false`
- `geometry_model == "polygon_code"`

It skips:

- `geometry_model == "none"`
- `is_reference_only == true`
- `insert_object_group`
- `convex_atoms`

Node IDs, relation endpoints, roles, labels, frames, `renderable`, and `is_reference_only` are preserved.

## Geometry Checks

For each polygon payload, the simplifier:

1. Removes near-collinear ring points.
2. Runs Shapely `simplify(..., preserve_topology=True)`.
3. Caps rings with ordered importance sampling when needed.
4. Checks validity, emptiness, minimum area, and area error.
5. Falls back to the original node geometry on failure.

The benchmark reports invalid counts, failed-node counts, total area error, mean area-error ratio, and the largest changed nodes.

## Choosing A Profile

Choose a simplification profile by comparing:

- Token reduction ratio.
- Longest sequence length after simplification.
- Invalid and failed geometry counts.
- Mean and max node area-error ratio.
- Whether the visual quality remains acceptable on sampled outputs.

The goal is not to minimize tokens at all costs. The practical target is to reduce long-tail sequences enough for generator training while preserving stable geometry.

## Future Compression Work

This stage focuses on polygon tokens because they are usually the largest contributor. Later compression work can target:

- `contains` relation block encoding.
- Relation block encoding by relation type.
- Two-stage topology and geometry generation.

Those changes would affect tokenizer design and should be evaluated separately from geometry simplification.
