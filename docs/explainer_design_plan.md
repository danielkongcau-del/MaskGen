# 解释器设计计划（Explainer Design Plan）

版本：`explainer_plan_v1`

目标：指导解释器的第一版实现。解释器负责把低层几何证据压缩成 `docs/generator_io_spec.md` 中定义的 `parse_graph`，并输出可追溯、可验证、可训练的生成器目标。

本文件不是生成器训练计划，也不是新的几何近似器或分割器设计。

当前推荐先实现并评估弱解释器 profile：`weak_convex_face_atoms_v1`。该 profile 不强制判断 support/divider/insert，而是把 evidence 打包成 semantic faces、convex atoms、label groups 和 adjacency。强语义解释器可以作为后续压缩层，而不是当前唯一主线。

---

## 1. 解释器在整体管线中的位置

当前目标管线是：

```text
raw mask
  -> partition graph
  -> global approx partition
  -> explanation evidence
  -> explainer
  -> generator target parse_graph
```

解释器只处理 `explanation evidence` 之后的数据。

解释器不应该直接读取原始 mask，也不应该重新做全图边界近似、几何近似或凸分割。

---

## 2. 解释器输入

### 2.1 主输入：`maskgen_explanation_evidence_v1`

解释器的主输入应该是 evidence JSON，而不是 raw partition graph。

建议格式：

```json
{
  "format": "maskgen_explanation_evidence_v1",
  "source_global_approx": "...",
  "source_partition_graph": "...",
  "source_mask": "...",
  "size": [256, 256],
  "faces": [],
  "arcs": [],
  "adjacency": [],
  "global_validation": {},
  "evidence_validation": {},
  "statistics": {}
}
```

如果当前工作区还没有该 evidence builder，则解释器实现前应先补齐 evidence builder。解释器不应该自己到处调用 `global_approx_partition`、`bridged_convex_partition` 或旧 CDT baseline。

### 2.2 `faces`

每个 face 至少应包含：

```json
{
  "id": 12,
  "label": 4,
  "bbox": [x0, y0, x1, y1],
  "outer_arc_refs": [],
  "hole_arc_refs": [],
  "geometry": {
    "outer": [],
    "holes": []
  },
  "features": {
    "area": 0.0,
    "area_ratio": 0.0,
    "centroid": [0.0, 0.0],
    "perimeter": 0.0,
    "compactness": 0.0,
    "solidity": 0.0,
    "oriented_aspect_ratio": 0.0,
    "degree": 0,
    "shared_boundary_length": 0.0,
    "touches_border": false,
    "hole_count": 0,
    "is_thin": false,
    "is_compact": false
  },
  "convex_partition": {
    "valid": true,
    "piece_count": 0,
    "atoms": [],
    "validation": {}
  }
}
```

解释器需要依赖这些字段判断某个 face 更像：

```text
support_region
divider_region
insert_object
hole
residual_region
```

### 2.3 `arcs`

每条 arc 至少应包含：

```json
{
  "id": 33,
  "incident_faces": [5, 12],
  "is_shared": true,
  "is_border": false,
  "points": [],
  "features": {
    "length": 0.0,
    "vertex_count": 0,
    "compression_ratio": 0.0,
    "straight_distance": 0.0,
    "endpoint_distance": 0.0
  }
}
```

解释器需要用 arc 判断相邻关系、共享边界长度、分隔证据和渲染追溯。

### 2.4 `adjacency`

相邻 face 的聚合记录：

```json
{
  "faces": [5, 12],
  "labels": [1, 4],
  "arc_ids": [33, 34],
  "shared_length": 42.0,
  "arc_count": 2
}
```

解释器应优先通过 `adjacency` 构建局部 patch，而不是重新做几何相交。

### 2.5 可选输入

可选输入包括：

- `class_map`：类别名，仅用于可视化和人工检查，不应作为核心规则依赖。
- `explainer_config`：控制候选数量、是否启用某些模板、是否输出调试信息。
- `renderer_config`：控制解释结果如何渲染回 partition/mask。

---

## 3. 解释器输出

### 3.1 主输出：`maskgen_explanation_v1`

解释器输出建议格式：

```json
{
  "format": "maskgen_explanation_v1",
  "source_evidence": "...",
  "selected_explanations": [],
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

### 3.2 `selected_explanations`

记录解释器选择的局部解释。

```json
{
  "patch_id": "patch_012",
  "evidence": {
    "face_ids": [5, 6, 7, 12],
    "arc_ids": [20, 21, 22]
  },
  "selected_candidate_id": "candidate_0",
  "score_gap": 18.4,
  "selected_template": "split_by_divider",
  "generated_node_ids": ["support_0", "divider_0"],
  "generated_relation_ids": ["relation_0"],
  "cost": {
    "total": 31.2,
    "template": 3.0,
    "topology": 6.0,
    "geometry": 18.0,
    "residual": 4.2,
    "invalid": 0.0,
    "evidence_bonus": -1.3
  }
}
```

### 3.3 `parse_graph`

这是生成器正式训练目标。

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
  "metadata": {
    "source_explanation": "...",
    "code_length": 0.0,
    "render_iou": 0.0,
    "valid": true
  }
}
```

`parse_graph` 必须遵守 `generator_io_spec.md`：

- 节点使用 `role`，不使用旧的 `type` 表示结构角色。
- `role`、`label`、`geometry_model` 分离。
- 当前正式目标只有 `parse_graph`。
- `program_sequence` 不在第一版解释器输出中作为主目标。

---

## 4. 解释器应该如何处理输入数据

### 4.0 弱解释器基线

在强解释器之前，先建立一个确定性弱解释器：

```text
explanation evidence
  -> label_group nodes
  -> semantic_face nodes
  -> convex_atom nodes
  -> atom_part_of / face_adjacent / label_group_contains relations
  -> weak parse_graph
```

该基线不解决“谁是 support、谁是 divider、谁是 insert”。它只保留稳定几何组成结构，让生成器先学习“用凸多边形拼搭 face”和“face 之间如何邻接”。

这一路线的验收标准是：

```text
1. 每个 evidence face 都有 semantic_face node。
2. 每个可用 convex atom 都有 convex_atom node。
3. 每个 atom 通过 atom_part_of 指回 parent face。
4. 每条 evidence adjacency 变成 face_adjacent relation。
5. 同一 label 的 face 可以被 label_group 聚合。
6. 输出仍然是 maskgen_generator_target_v1 / parse_graph。
```

弱解释器不是最终上限。它是稳定训练目标和强解释器对照组。

### 4.1 阶段一：加载与基础验证

输入 evidence 后，解释器首先检查：

```text
1. format 是否正确
2. size 是否存在
3. faces / arcs / adjacency 是否存在
4. face geometry 是否有效
5. global_validation 是否可用
6. 每个 adjacency 是否引用存在的 face 和 arc
```

如果 evidence 本身不可用，解释器应停止并输出明确错误，不应继续生成伪 parse graph。

### 4.2 阶段二：构建工作图

解释器内部构建一个 face adjacency graph：

```text
node = face
edge = adjacency
edge attributes = shared_length, arc_ids, labels
node attributes = geometry, features, convex atoms
```

该工作图用于：

- 找局部 patch。
- 判断候选 support。
- 判断候选 divider。
- 判断候选 insert。
- 计算解释代价。

### 4.3 阶段三：生成基础候选角色

对每个 face 生成可能角色候选：

```text
support_region candidate
divider_region candidate
insert_object candidate
hole candidate
residual_region candidate
```

第一版可以基于 evidence features 生成候选，但不能硬编码特定遥感类别。

可用证据包括：

- 面积和面积占比。
- 细长度和最小旋转矩形长宽比。
- 邻接度。
- 共享边界长度。
- 是否接触边界。
- convex atom 数量。
- hole_count。
- compactness / solidity。

这些证据只用于生成候选和评分，不应被写成不可迁移的类别规则。

### 4.3.1 图内 label-role 一致性

仅靠 face-local cost 会导致同一张图中同一 `label` 被大量解释成不同 `role`。这会增加生成器学习难度。

但 `role` 本质上不是绝对属性，而是关系属性。因此，第一版解释器不应该只靠 face-local role 分类，而应先做 label-pair relation pass：

```text
1. 找出图中相邻的 label pair。
2. 对每个 label pair 生成候选解释：
   - A support, B insert
   - B support, A insert
   - A support, B divider
   - B support, A divider
   - A/B adjacent supports
   - independent / residual
3. 为每个候选计算 code length。
4. 选择低成本候选。
5. 把候选中的 role 作为该 label 在当前图中的关系证据。
```

当前初版实现采用 `partition_gen/pairwise_relation_explainer.py`。它不是直接按面积或类别名判定 role，而是把每个相邻 label pair 构造成二类子场景：

```text
label A geometry + label B geometry
  -> 生成候选解释
  -> 对候选中的 support 补全策略进行评分
  -> 调用已有 convex partition 估计几何 code length
  -> 选择低表示代价解释
```

其中：

- `support_with_inserts` 要求补全后的 support 能覆盖 insert；否则该候选会被惩罚。
- insert 不能比 support 更大；单个过大的连通块也不应被当作普通嵌入物。
- `split_by_divider` 要求 divider 被补全后的 support 覆盖；这种覆盖是预期行为，不应作为 false cover。
- divider 候选会考虑细长度、上下文共享强度、面积比例和碎片数量。
- `convex_hull_fill` 不是免费操作；support 补全面积越大，候选代价越高。

这一步的目的不是得到最终全局最优解释，而是给 image-level consistency pass 提供更可靠的关系证据。

然后再做 image-level consistency pass：

```text
1. 对每个 face 保留所有 role candidate cost。
2. 优先使用 label-pair relation pass 给出的 preferred role。
3. 如果某个 label 没有足够 pairwise 证据，再看本地最低代价 role 的面积占比。
4. 对偏离 dominant role 的 face 加一个结构不一致成本。
5. 重新选择每个 face 的 role。
```

这里使用关系证据和面积占比，而不是简单 face 数量，是为了避免大量小碎片压过少数大主体区域。

这不是硬规则：

```text
不是 label -> role 固定映射
而是同一图内 label -> role 倾向
```

如果某个 face 的局部证据足够强，它仍然可以偏离 dominant role。解释器必须在 diagnostics 中记录这些例外：

```json
{
  "label_role_summary": {
    "4": {
      "dominant_role": "divider_region",
      "counts_before": {},
      "counts_after_consistency": {},
      "counts_final": {},
      "override_face_ids": []
    }
  }
}
```

这一步仍然是基于 code length 的相对代价，不应写成 `label == X -> role == Y`。

### 4.4 阶段四：构建局部 patch

解释器应把全图拆成若干局部 patch，再在每个 patch 内比较解释模板。

第一版 patch 可以来自：

```text
1. 一个候选 support face 及其邻居
2. 一个候选 divider face 及其两侧或多侧邻居
3. 一个大 face 及其内部/邻接的小 face 组
4. 若干强邻接 face 组成的连通子图
```

patch 必须保存其覆盖的 `face_ids` 和 `arc_ids`，用于后续冲突消解和追溯。

### 4.5 阶段五：为每个 patch 生成模板候选

第一版只支持三个模板。

#### 4.5.1 `support_with_inserts`

解释：

```text
一个支撑区域中放置若干嵌入物
```

该模板的协议语义是：

```text
support_region 可以先被解释为完整无洞区域；
insert_object 通过 inserted_in / contains 关系嵌入其中；
renderer 在最终 partition / mask 中让 insert_object 覆盖 support_region 的对应位置。
```

因此，解释器不应因为 support 内部存在房屋、设施等独立语义对象，就强制把 support 输出成带洞多边形。只有当内部区域是真正没有独立语义对象的负空间时，才应考虑 `hole`。

适合情况：

- 一个大区域邻接或包含多个较小、较紧凑对象。
- 小对象之间不构成明显分隔网络。
- 用“支撑 + 嵌入物组”比独立表示每个 face 更短。

输出：

```text
support_region
insert_object_group
insert_object
inserted_in / contains relations
```

#### 4.5.2 `split_by_divider`

解释：

```text
某个 divider_region 切分或组织一个 support_region
```

适合情况：

- 一个或多个细长区域与多个支撑区域邻接。
- 支撑区域被 divider 分成若干片段。
- 用“support + divider”比“多个独立 support face”更短。

输出：

```text
support_region
divider_region
divides relation
```

注意：

`divider_region` 的 `geometry_model` 第一版建议优先使用 `polygon_code` 或 `convex_atoms`。不要强制抽取中心线。

#### 4.5.3 `independent_faces`

解释：

```text
当前 patch 无法可靠解释，保留为 residual 或独立 face
```

输出：

```text
residual_region
adjacent_to relations
```

这是第一版解释器的安全出口。

### 4.6 阶段六：计算候选解释代价

每个候选解释都输出统一 cost：

```json
{
  "total": 42.7,
  "template": 3.0,
  "topology": 8.0,
  "geometry": 20.0,
  "residual": 6.0,
  "invalid": 0.0,
  "evidence_bonus": -1.3
}
```

第一版目标不是数学全局最优，而是稳定、可解释、可调试。

建议含义：

- `template`：使用模板的固定编码代价。
- `topology`：节点数、关系数、子对象数。
- `geometry`：几何参数量、顶点数、atoms 数。
- `residual`：未被模板解释的面积或复杂度。
- `invalid`：几何非法、关系非法、无法渲染。
- `evidence_bonus`：强证据奖励，例如细长区域支持 `divider_region`。

如果某个候选无法渲染或违反不变量，`invalid` 应足够大，或者直接标记为不可选。

### 4.7 阶段七：选择非冲突候选并合成全图解释

候选之间可能覆盖相同 face，因此需要选择一组互不冲突或可合并的候选，组成 scene-level explanation。

第一版可以采用贪心策略：

```text
1. 按 cost improvement 排序
2. 依次选择不冲突候选
3. 未覆盖 face 进入 residual
```

但文档上必须承认：这不是全局最优。

如果要严格优化，需要一个能够达成以下目标的算法：

```text
从一组互相冲突的局部候选解释中，选择覆盖全图 face 集合且总代价最低的候选组合。
```

这本质上接近加权集合覆盖、加权集合打包或整数规划问题。第一版可以先用贪心 baseline，但如果贪心不稳定，需要引入正式优化器或明确算法。

### 4.8 阶段八：生成规范化 `parse_graph`

选择完成后，解释器生成全图 `parse_graph`：

```text
generator_target
  format = maskgen_generator_target_v1
  target_type = parse_graph
  size
  parse_graph
    nodes
    relations
    residuals
  metadata
```

必须做规范化：

- 节点 ID 重写为 `support_0`、`divider_0`、`insert_0`、`residual_0`。
- 节点按固定顺序排序。
- 关系按固定顺序排序。
- 所有几何转换到局部坐标。
- 所有 evidence 保留在解释文件中，但训练 target 可以剥离。

### 4.9 阶段九：渲染与验证

解释器输出不能只停留在 JSON 层，必须验证其可渲染性。

最低验证要求：

```text
1. parse_graph 所有关系引用存在节点
2. 每个节点 geometry_model 可被 renderer 识别
3. 每个节点 geometry 合法
4. residual 合法
5. 渲染后 partition 没有明显 gap / overlap
6. render_iou 达到记录标准
```

如果 renderer 尚未实现，解释器第一版至少要输出：

```text
render_validation.status = "not_implemented"
```

不能伪造 `render_iou`。

---

## 5. 解释器配置

建议配置：

```python
@dataclass(frozen=True)
class ExplainerConfig:
    max_patch_size: int = 32
    max_candidates_per_patch: int = 8
    enable_support_with_inserts: bool = True
    enable_split_by_divider: bool = True
    enable_independent_faces: bool = True
    residual_allowed: bool = True
    keep_top_k: int = 3
```

这些参数只控制搜索预算和输出规模，不应该承担数据集语义规则。

---

## 6. 第一版实现边界

第一版应该实现：

```text
1. 读取 evidence JSON
2. 构建 face adjacency graph
3. 生成基础 role candidates
4. 支持三个模板：
   - support_with_inserts
   - split_by_divider
   - independent_faces
5. 输出 selected_explanations
6. 输出 parse_graph
7. 输出 diagnostics
8. 输出可视化
```

第一版不应该实现：

```text
1. 新的 geometry approximator
2. 新的 global approximation
3. 新的 convex partitioner
4. 神经网络生成器
5. 强制中心线抽取
6. program_sequence 主训练格式
```

---

## 7. 需要特别谨慎的困难算法

以下部分不要靠临时规则伪装成正式解法。如果第一版暂时不能解决，应明确 fallback 或输出 residual。

### 7.1 从面状 divider 中提取中心线宽度图

目标：

```text
给定一个细长面状区域，提取稳定的 skeleton / centerline graph，并估计每条边的宽度。
```

这是一个独立困难算法。第一版不要强制做。

如果需要做，应寻找或实现能够达成该目标的可靠算法，例如稳定的 medial axis、straight skeleton、形态学骨架加图清理，或其他有明确文献/库支持的方法。

当前建议：

```text
divider_region 第一版优先使用 polygon_code 或 convex_atoms。
```

### 7.2 从被 divider 切开的多个 face 反推原始 support

目标：

```text
给定若干同类或相近 face，以及一个或多个 divider face，判断这些 face 是否应该被解释为同一个 support 被 divider 切开。
```

这不是简单邻接判断。它需要考虑：

- support 片段的类别一致性。
- divider 的位置和连通性。
- 片段合并后的几何合理性。
- 与 independent_faces 表示相比的 code length。

第一版可以做局部启发式 baseline，但如果结果不稳定，需要一个明确的图搜索或结构匹配算法。

### 7.3 候选解释的全局冲突消解

目标：

```text
从局部候选解释集合中，选择一组覆盖全图且总代价最低的解释。
```

这接近加权集合覆盖、加权集合打包或整数规划。

第一版可以用贪心 baseline，但必须在 diagnostics 中记录：

```text
selection_method = "greedy_baseline"
global_optimal = false
```

如果后续需要更稳定结果，需要引入能够求解该目标的优化算法。

### 7.4 parse_graph 到 rendered partition 的 renderer

目标：

```text
把 parse_graph 渲染成全图共享边界 partition，并验证 gap / overlap / topology。
```

这是解释器闭环验证的关键。

如果 renderer 不存在，解释器只能输出结构解释，不能声称 render_iou 已验证。

### 7.5 泛化类别语义

目标：

```text
在不同数据集中，让 label 与 role 的对应关系由 evidence 和统计规律决定，而不是硬编码遥感类别。
```

第一版可以使用 label 作为弱证据，但不能写成：

```text
label == road -> divider
label == building -> insert
```

这会破坏泛化性。

---

## 8. 诊断与可视化

解释器必须输出足够诊断信息，避免黑箱。

建议可视化四联图：

```text
1. evidence faces / arcs
2. candidate patches
3. selected explanations with role colors
4. generated parse_graph / residual overlay
```

建议 diagnostics 包含：

```json
{
  "face_count": 0,
  "patch_count": 0,
  "candidate_count": 0,
  "selected_candidate_count": 0,
  "residual_face_count": 0,
  "residual_area_ratio": 0.0,
  "total_code_length": 0.0,
  "selection_method": "greedy_baseline",
  "global_optimal": false,
  "template_histogram": {},
  "role_histogram": {},
  "failure_reasons": []
}
```

---

## 9. 推荐新增文件

建议新增：

```text
partition_gen/explainer.py
partition_gen/explainer_costs.py
partition_gen/explainer_templates.py
partition_gen/explainer_canonicalize.py
scripts/build_explanation_single.py
scripts/visualize_explanation.py
docs/explainer_design_plan.md
```

如果 `explanation_evidence` 尚未实现，还应先新增：

```text
partition_gen/explanation_evidence.py
scripts/build_explanation_evidence_single.py
scripts/visualize_explanation_evidence.py
docs/explanation_evidence.md
```

---

## 10. 验收标准

第一版解释器完成时，应满足：

```text
1. 能读取一个 evidence JSON。
2. 能输出 maskgen_explanation_v1。
3. 输出中包含 selected_explanations。
4. 输出中包含完整的 generator_target 对象。
5. generator_target.parse_graph 中包含 nodes / relations / residuals。
6. parse_graph 使用 role / label / geometry_model 分离设计。
7. 不输出正式 program_sequence。
8. 不强制中心线宽度图。
9. 所有未解释 face 都进入 residual。
10. diagnostics 明确记录 residual 比例和选择方法。
11. 可视化能看出哪些区域被解释成 support / divider / insert / residual。
```

如果某个困难算法尚未实现，必须明确记录：

```text
status = "not_implemented"
fallback = "residual" 或 "polygon_code"
```

不能用临时规则伪装成稳定算法。
