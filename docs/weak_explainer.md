# 弱解释器设计（Weak Explainer）

版本：`weak_convex_face_atoms_v1`

## 目标

弱解释器的目标是把 `explanation evidence` 打包成稳定、可训练、可追溯的 `parse_graph`，但不强行判断 `support_region`、`divider_region`、`insert_object` 这类高层语义角色。

它回答的问题是：

```text
这张图由哪些语义 face 构成？
每个 face 属于哪个 label？
每个 face 可以用哪些凸多边形 atom 表示？
face 之间如何邻接？
同一 label 的 face 如何分组？
```

它暂时不回答：

```text
道路是否切开田地？
房子是否嵌入平原？
某个区域是否是支撑区域？
```

## 输入

输入是 `maskgen_explanation_evidence_v1`。

必须使用已经存在的 evidence 层：

```text
global approx partition
  -> explanation evidence
  -> weak explainer
```

弱解释器不直接读取原始 mask，不重新做全图近似，也不修改几何近似器或凸分割器。

## 输出

输出仍然是：

```json
{
  "format": "maskgen_explanation_v1",
  "explainer_profile": "weak_convex_face_atoms_v1",
  "generator_target": {
    "format": "maskgen_generator_target_v1",
    "target_type": "parse_graph",
    "parse_graph": {
      "nodes": [],
      "relations": [],
      "residuals": []
    }
  }
}
```

区别在于 `parse_graph` 使用结构中性的 node role：

```text
label_group
semantic_face
convex_atom
```

## 节点

### `label_group`

表示同一 label 的 face 集合。

```json
{
  "id": "label_group_0",
  "role": "label_group",
  "label": 6,
  "children": ["face_0", "face_1"],
  "count": 2
}
```

### `semantic_face`

表示一个全图共享边界下的语义 face。

如果 evidence 中有 arc 引用，优先使用 `boundary_arcs`，保证和全图共享边界一致。

```json
{
  "id": "face_0",
  "role": "semantic_face",
  "label": 6,
  "source_face_id": 12,
  "geometry_model": "boundary_arcs",
  "geometry": {
    "outer_arc_refs": [],
    "hole_arc_refs": []
  },
  "atom_ids": ["atom_0", "atom_1"]
}
```

如果没有 arc 引用，则回退到 `polygon_code`。

### `convex_atom`

表示某个 face 的凸分割 atom。

```json
{
  "id": "atom_0",
  "role": "convex_atom",
  "label": 6,
  "parent_face": "face_0",
  "geometry_model": "convex_polygon",
  "geometry": {
    "type": "quad",
    "outer_local": [],
    "vertex_count": 4,
    "area": 123.0
  }
}
```

`convex_atom` 是生成器的低层几何学习单元。它不表示高层语义原因，只表示 face 的可学习几何组成。

## 关系

### `label_group_contains`

```json
{
  "type": "label_group_contains",
  "parent": "label_group_0",
  "child": "face_0"
}
```

### `atom_part_of`

```json
{
  "type": "atom_part_of",
  "atom": "atom_0",
  "face": "face_0"
}
```

### `face_adjacent`

```json
{
  "type": "face_adjacent",
  "faces": ["face_0", "face_1"],
  "source_face_ids": [12, 18],
  "labels": [6, 2],
  "arc_ids": [33, 34],
  "shared_length": 42.0
}
```

## 为什么这比强解释器稳定

强解释器需要决定某个区域是 support、divider 还是 insert。这个判断依赖上下文、数据集语义和局部结构，容易变成大量手调规则。

弱解释器只保留确定性结构：

- face 是什么。
- label 是什么。
- face 如何邻接。
- face 如何被凸 atom 组成。
- 同类 face 如何分组。

因此它更适合作为第一版生成器训练目标。

## 和凸分割器的关系

弱解释器依赖凸分割器，但不改变凸分割器。

```text
semantic_face
  -> convex_partition
  -> convex_atom nodes
```

如果某个 face 没有可用凸 atom，弱解释器不会让整图失败，而是在 `residuals` 中记录：

```json
{
  "face_node_id": "face_12",
  "reason": "missing_convex_atoms"
}
```

## Renderer / validator

弱解释器现在有一个验证用 renderer：

```text
weak parse_graph
  -> convex_atom local coordinates
  -> world-space polygons
  -> per-face atom union
  -> rendered partition / mask
  -> validation
```

它主要验证两件事：

1. 每个 face 的 convex atoms 能否 union 回原 evidence face。
2. 所有 rendered faces 在全图上是否存在 gap / overlap。

输出格式：

```json
{
  "format": "maskgen_weak_rendered_partition_v1",
  "faces": [],
  "validation": {
    "is_valid": true,
    "full_iou": 1.0,
    "overlap_area": 0.0,
    "gap_area": 0.0,
    "low_iou_face_ids": []
  }
}
```

注意：

- `full_iou` 比较的是 rendered atom union 与 evidence geometry，不是原始像素 mask。
- `mask_pixel_accuracy` 如果存在，是 rendered mask 与原始 mask 的像素一致率。由于当前表示已经经过近似，它通常不会等于 1。
- 这个 renderer 是验证工具，不是最终生成器 renderer。

## 当前限制

- `convex_atom` 使用 face-local 坐标，但还没有训练用 tokenizer。
- 该 profile 不输出 support/divider/insert，因此不能表达强语义关系。
- 如果后续需要强解释器，可以在该弱表示之上增加模板选择层。
- 当前 renderer 只闭环验证 weak profile，尚未支持强解释器的覆盖式 `inserted_in` 渲染。

## 推荐命令

从已有 evidence 构建：

```powershell
conda run -n lmf python scripts/build_weak_explanation_single.py `
  --evidence-json outputs/visualizations/explanation_evidence_val37_initial_micro_absorbed.json `
  --output outputs/visualizations/weak_explanation_val37_initial_micro_absorbed.json
```

可视化：

```powershell
conda run -n lmf python scripts/visualize_weak_explanation.py `
  --weak-json outputs/visualizations/weak_explanation_val37_initial_micro_absorbed.json `
  --evidence-json outputs/visualizations/explanation_evidence_val37_initial_micro_absorbed.json `
  --mask-root data/remote_256 `
  --split val `
  --stem 37 `
  --output outputs/visualizations/weak_explanation_val37_initial_micro_absorbed.png
```

渲染和验证：

```powershell
conda run -n lmf python scripts/render_weak_explanation.py `
  --weak-json outputs/visualizations/weak_explanation_val37_initial_micro_absorbed.json `
  --evidence-json outputs/visualizations/explanation_evidence_val37_initial_micro_absorbed.json `
  --output-json outputs/visualizations/weak_render_val37_initial_micro_absorbed.json `
  --output-mask outputs/visualizations/weak_render_val37_initial_micro_absorbed.png `
  --output-validation outputs/visualizations/weak_render_val37_initial_micro_absorbed_validation.json `
  --mask-root data/remote_256
```

批量 benchmark：

```powershell
conda run -n lmf python scripts/benchmark_weak_explainer.py `
  --partition-root data/remote_256_partition `
  --split val `
  --output outputs/benchmarks/weak_explainer_benchmark_val.jsonl `
  --convex-backend fallback_cdt_greedy `
  --mask-root data/remote_256
```

汇总 benchmark：

```powershell
conda run -n lmf python scripts/summarize_weak_explainer_benchmark.py `
  --input outputs/benchmarks/weak_explainer_benchmark_val.jsonl `
  --output outputs/benchmarks/weak_explainer_benchmark_val.md
```

导出问题样本：

```powershell
conda run -n lmf python scripts/export_weak_explainer_failures.py `
  --benchmark-jsonl outputs/benchmarks/weak_explainer_benchmark_val.jsonl `
  --output-dir outputs/visualizations/weak_explainer_failures `
  --mask-root data/remote_256
```
