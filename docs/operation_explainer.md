# Operation-Level 强解释器

版本：`operation_level_mdl_v1`

本模块实现第四层解释器：`operation_explainer`。它不是新的几何近似器、不是新的凸分割器，也不是神经网络生成器。它读取 `maskgen_explanation_evidence_v1`，生成局部高层操作候选，计算压缩收益，并用 OR-Tools 从候选中选择一组互不冲突、覆盖所有 face 的解释操作。

## 1. 四层解释关系

当前解释相关代码分为四层：

- `weak_explainer.py` 是稳定打包器。它输出 `label_group`、`semantic_face`、`convex_atom`，以及 `label_group_contains`、`atom_part_of`、`face_adjacent` 等基础关系。它不判断强 role。
- `explainer.py` 是 face 级 role prior。它根据面积、邻接、长宽比、紧凑度、凸块数量等信息估计 `support_region`、`divider_region`、`insert_object`、`residual_region` 的候选代价。这只是弱先验。
- `pairwise_relation_explainer.py` 是 label-pair 级 prior。它提供 `support_with_inserts`、`split_by_divider`、`adjacent_supports`、`independent_faces` 等二类关系候选。这也不是最终解释。
- `operation_explainer.py` 是 operation-level 强解释器。它在局部 patch 上生成操作候选，计算压缩收益，再做全局选择。

## 2. 输入

主输入是 `explanation_evidence.py` 输出的：

```json
{
  "format": "maskgen_explanation_evidence_v1",
  "size": [256, 256],
  "faces": [],
  "arcs": [],
  "adjacency": [],
  "statistics": {},
  "evidence_validation": {}
}
```

每个 face 应包含：

- `geometry`
- `features`
- `convex_partition`

`operation_explainer` 只消费这些已有证据，不重新做全图近似、不重新做凸分割。

可选输入包括：

- weak explanation
- face role prior
- pairwise relation prior

如果调用方不提供，`operation_explainer` 会调用现有模块临时生成。

## 3. 操作类型

第一版支持四类操作：

- `OVERLAY_INSERT`：支撑区域上嵌入若干小对象，例如平地上嵌入房屋。
- `DIVIDE_BY_REGION`：细长区域分隔或组织支撑区域，例如道路切分田地。
- `PARALLEL_SUPPORTS`：多个大区域并列相邻，没有明显包含或分隔关系。
- `RESIDUAL`：无法压缩解释的 face 直接保留为残差。

所有 face 都会拥有一个 `RESIDUAL` 候选，确保全局选择总能覆盖输入。

## 4. 压缩收益

每个候选都会计算：

```text
compression_gain = independent_cost - operation_cost
```

其中：

- `independent_cost` 表示不用高层解释，直接用 semantic face / convex atoms 描述这些 face 的代价。
- `operation_cost` 表示使用某个高层操作解释这些 face 的代价。

如果 `compression_gain > 0`，说明该操作比低层独立描述更短。非 residual 候选默认只有在压缩收益为正时才允许被全局选择器选中。

## 5. OR-Tools 全局选择

`operation_selector.py` 使用 OR-Tools CP-SAT：

```text
变量：x_i = 是否选择候选 i

约束：
  每个 face 恰好被一个候选覆盖
  非法候选不可选
  非 residual 且压缩收益不足的候选不可选

目标：
  最大化总 compression_gain
```

如果 OR-Tools 求到最优，输出：

```json
{
  "selection_method": "ortools_cp_sat",
  "global_optimal": true
}
```

如果 OR-Tools 不可用或求解失败，允许 greedy fallback，但必须记录：

```json
{
  "selection_method": "greedy_fallback",
  "global_optimal": false
}
```

## 6. 输出

输出格式：

```json
{
  "format": "maskgen_operation_explanation_v1",
  "explainer_profile": "operation_level_mdl_v1",
  "selected_operations": [],
  "candidate_summary": {},
  "generator_target": {
    "format": "maskgen_generator_target_v1",
    "target_type": "parse_graph",
    "size": [256, 256],
    "parse_graph": {
      "nodes": [],
      "relations": [],
      "residuals": []
    },
    "metadata": {}
  },
  "diagnostics": {},
  "validation": {}
}
```

`generator_target` 使用嵌套 `parse_graph` 格式。第一版不会输出正式 `program_sequence`。

## 7. 当前限制

- 第一版不实现 operation-level renderer，因此 `render_iou` 必须为 `null`。
- 第一版不做中心线抽取，不强制 `skeleton_width_graph`。
- 第一版不把 label 硬编码成 role；label 只作为弱证据。
- 第一版不会修改 `weak_explainer.py`、`explainer.py`、`pairwise_relation_explainer.py`、`explanation_evidence.py`、全图近似器或凸分割器。

## 8. 单样本命令

```powershell
conda run -n lmf python scripts/build_operation_explanation_single.py `
  --evidence-json outputs/visualizations/explanation_evidence_val83.json `
  --output outputs/visualizations/operation_explanation_val83.json
```

可视化：

```powershell
conda run -n lmf python scripts/visualize_operation_explanation.py `
  --operation-json outputs/visualizations/operation_explanation_val83.json `
  --evidence-json outputs/visualizations/explanation_evidence_val83.json `
  --output outputs/visualizations/operation_explanation_val83.png
```

## 9. v1.1 改进

`operation_level_mdl_v1` 的 v1.1 重点不是新增操作类型，而是让候选代价和诊断更可信。

### 9.1 节点对象代价

非 residual 操作的 `operation_cost` 现在不再只计算 template、geometry 和 relation。它还会计算每个高层节点自身的对象代价：

- `support_region`
- `divider_region`
- `insert_object`
- `insert_object_group`
- `residual_region`

其中 `insert_object_group` 默认使用较低的 `cost_group_object`，其余节点默认使用 `cost_object`。这样可以避免 `OVERLAY_INSERT`、`DIVIDE_BY_REGION`、`PARALLEL_SUPPORTS` 因缺少对象成本而被系统性低估。

### 9.2 False Cover 惩罚

潜在父区域可能会错误覆盖候选以外的其他 face，尤其是 `convex_hull_fill`。v1.1 会对 latent geometry 做 false-cover 诊断：

```text
false_cover = latent_geometry ∩ non_covered_faces
```

诊断会记录：

- 被错误覆盖的面积
- false-cover ratio
- 被覆盖的 face id 和 label
- false-cover cost
- 是否 hard invalid

这不是 renderer，也不计算 `render_iou`。它只是判断 latent parent geometry 是否明显越界。

### 9.3 Patch 截断保留 Seed

当 patch 超过 `max_patch_size` 时，v1.1 会保证 `seed_face_id` 不被截断。其余 face 按与 seed 的直接邻接、共享边界长度、role prior 相关性和面积相关性排序。patch metadata 会记录：

- `trimmed`
- `original_face_count`
- `final_face_count`
- `dropped_face_ids`

### 9.4 Benchmark

批量评估：

```powershell
conda run -n lmf python scripts/benchmark_operation_explainer.py `
  --evidence-root outputs/evidence `
  --output outputs/benchmarks/operation_explainer.jsonl
```

汇总：

```powershell
conda run -n lmf python scripts/summarize_operation_explainer_benchmark.py `
  --input outputs/benchmarks/operation_explainer.jsonl `
  --output outputs/benchmarks/operation_explainer.md
```

benchmark 会统计 residual 比例、操作分布、压缩收益、false-cover、OR-Tools 状态和失败原因。

### 9.5 当前仍然不是最终强解释器

当前版本仍然使用：

```text
每个 face 恰好被一个 selected operation 覆盖
```

这意味着它还不能完整表达“同一个 support 同时被道路分隔、又承载房屋嵌入”的多重操作关系。后续版本需要引入：

- `consumed_face_ids`
- `referenced_face_ids`
- operation dependency graph

在这些机制实现前，v1.1 的定位是更可信的 operation 候选选择器和诊断器，而不是最终生成程序。

## 10. Cost Profiles

当前支持两个代价模式：

### 10.1 `heuristic_v1`

旧的启发式加权代价，保留用于回归测试和对比。它会把 object、relation、vertex、atom、template、false-cover 等项加权求和。

该模式工程上可诊断，但不是长期稳定目标，因为不同项的量纲并不完全一致。

### 10.2 `token_length_v1`

新的默认模式。它把代价解释为编码长度，也就是描述一个结构大致需要多少个符号 / token。

低层独立描述长度：

```text
independent_length =
  semantic_face token
+ polygon tokens
+ convex atom tokens
+ adjacency tokens
```

高层操作长度：

```text
operation_length =
  template tokens
+ node tokens
+ relation tokens
+ geometry tokens
+ residual tokens
+ latent policy tokens
+ exception tokens
```

压缩收益：

```text
compression_gain = independent_length - operation_length
```

这仍然是加法形式，但每一项都尽量使用同一单位：结构编码 token 数，而不是混合量纲的经验权重。

### 10.3 False Cover

在 `token_length_v1` 中，false cover 首先是硬约束和诊断，而不是大权重软惩罚。

如果 latent parent geometry 明显覆盖了候选之外的其他 face：

```text
false_cover_ratio > false_cover_ratio_invalid
```

该候选会被标记为 invalid，并增加 exception token。`render_iou` 仍然不会被伪造；当前只输出 proxy validation。
