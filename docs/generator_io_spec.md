# 生成器输入与输出数据定义（Generator Target Specification）

版本：`generator_target_v1`

用途：定义生成器最终应该学习和输出的结构化目标，让解释器可以从低层几何证据反推出更适合生成器学习的高层表示。

本文件是协议文件，不是解释器算法说明，也不是神经网络结构说明。

---

## 1. 总体目标

本项目的最终目标不是让生成器直接生成像素级 `mask`，也不是让生成器直接生成每个 `face` 的凸多边形分解。

生成器应该学习生成一种更高层的结构表示：

```text
parse_graph
  -> rendered global partition
  -> semantic faces
  -> raster mask
```

也就是说，生成器的主学习目标应该是：

```text
生成一个可渲染、可验证、可解释的解析图
```

而不是：

```text
生成 256x256 像素
```

也不是：

```text
独立生成每个 face 的凸块集合
```

当前版本只把 `parse_graph` 作为正式训练目标。`program_sequence` 暂时不作为主目标；如果后续需要自回归训练，可以从 `parse_graph` 通过确定性规则序列化得到。

---

## 2. 当前几何层与生成器目标的关系

当前已有几何表示大致是：

```text
原始 mask
  -> partition graph
  -> global approx partition
  -> explanation evidence
  -> explainer
  -> generator target
```

其中：

- `partition graph` 负责从 mask 中提取低层面分区和邻接关系。
- `global approx partition` 负责全图共享边界一致性和边界简化。
- `explanation evidence` 负责打包 faces、arcs、adjacency、convex atoms、features。
- `explainer` 负责把低层证据解释成高层结构。
- `generator target` 是生成器最终要学习的数据格式。

解释器可以有两个层级。

强解释器的职责是从低层 evidence 中提取：

```text
support region
divider region
insert object
hole / island
residual
relation
```

生成器学习的应该是这些结构对象及其关系。

弱解释器不强行判断这些高层语义角色，而是导出更稳定的几何组成结构：

```text
semantic face
convex atom
label group
face adjacency
```

当前如果强语义解释不稳定，推荐先使用弱解释器 profile：`weak_convex_face_atoms_v1`。

---

## 3. 生成器不应该主学什么

### 3.1 不应该主学像素

不推荐主目标：

```text
noise -> 256x256 mask
```

原因：

- 像素隐藏拓扑关系。
- 共享边界不显式。
- 连通性、洞、相邻关系、分隔关系等结构需要模型自己隐式学。
- 容易局部合理但全局不合法。

### 3.2 不应该主学每个 face 的凸块

也不推荐主目标：

```text
noise -> face_0 convex atoms
      -> face_1 convex atoms
      -> face_2 convex atoms
```

原因：

- convex atoms 是低层几何证据，不是高层生成原因。
- 生成器仍然需要隐式学习哪些 atoms 属于同一结构。
- face 之间的因果关系、分隔关系、嵌入关系仍然没有显式表达。
- 生成序列长，结构压缩不足。

### 3.3 凸块的正确位置

凸多边形仍然有用，但应该作为：

```text
低层残差表示
局部形状细节
fallback geometry
解释器证据
```

而不是最终主生成语言。

---

## 4. 生成器应该学习什么

生成器应该学习：

```text
p(parse_graph)
```

`parse_graph` 中包含：

1. 场景中有哪些高层对象。
2. 高层对象分别承担什么结构角色。
3. 每个对象对应什么语义类别。
4. 高层对象之间有什么关系。
5. 每个对象用什么几何模型表示。
6. 哪些区域无法解释，需要 residual 表达。

---

## 5. 核心设计原则

### 5.1 对象角色与语义类别必须分离

每个节点至少有两个不同字段：

```text
role
label
```

其中：

- `label` 是原始 mask 类别，例如道路、田地、建筑、水体、背景。
- `role` 是解释器赋予的结构角色，例如支撑区域、分隔区域、嵌入物、残差。

同一个 `label` 在不同数据集中可能承担不同 `role`。同一个 `role` 也可能对应不同 `label`。

例如：

```json
{
  "id": "node_0",
  "role": "support_region",
  "label": 1
}
```

不要把类别含义硬编码进结构角色。

### 5.2 对象角色与几何表达必须分离

每个节点还应该有：

```text
geometry_model
geometry
```

`role` 表示这个对象在解释中的作用，`geometry_model` 表示这个对象如何被渲染。

同一种 `role` 可以使用不同 `geometry_model`。例如 `divider_region` 可以使用简单多边形，也可以使用骨架宽度图。

### 5.3 不强制使用遥感专用范式

早期讨论中的“道路中心线 + 宽度 + 图拓扑”是一种有效几何模型，但它不应该成为协议的强制核心。

为了保证泛化性，本协议使用更中性的 `divider_region` 作为结构角色。中心线宽度图只作为可选的 `geometry_model`：

```text
skeleton_width_graph
```

这样遥感道路、水渠、田埂可以使用它；其他数据集可以继续使用 `polygon_code`、`convex_atoms` 或其他模型。

---

## 6. 几何模型类型

### 6.1 `polygon_code`

用一个或多个多边形环表示对象。

适用于：

- 大支撑区域。
- 普通面状区域。
- 简单嵌入物。
- 暂时不适合进一步结构化的对象。

字段示例：

```json
{
  "geometry_model": "polygon_code",
  "geometry": {
    "outer_local": [[-0.5, -0.4], [0.6, -0.3], [0.5, 0.5], [-0.4, 0.6]],
    "holes_local": []
  }
}
```

### 6.2 `convex_atoms`

用若干凸多边形表示对象。

适用于：

- residual。
- fallback geometry。
- 复杂局部形状。
- 无法稳定解释为更高层结构的区域。

字段示例：

```json
{
  "geometry_model": "convex_atoms",
  "atoms": [
    {
      "type": "triangle",
      "outer_local": [[-0.2, -0.1], [0.3, -0.1], [0.0, 0.2]]
    }
  ]
}
```

### 6.3 `boundary_arcs`

用全图共享 arc 引用表示对象边界。

适用于：

- 需要严格复用全图共享边界的对象。
- renderer 需要回到 global partition 的场景。
- 调试和追溯。

字段示例：

```json
{
  "geometry_model": "boundary_arcs",
  "geometry": {
    "outer_arc_refs": [{"arc_id": 12, "direction": 1}],
    "hole_arc_refs": []
  }
}
```

### 6.4 `skeleton_width_graph`

用骨架中心线、宽度和图拓扑表示细长区域。

适用于：

- 道路。
- 水渠。
- 田埂。
- 其他明确具有“中心线 + 宽度”结构的细长对象。

这是可选模型，不是所有 `divider_region` 都必须使用它。

字段示例：

```json
{
  "geometry_model": "skeleton_width_graph",
  "network": {
    "nodes": [
      {"id": "n0", "xy_local": [-0.3, -0.5], "degree": 1},
      {"id": "n1", "xy_local": [0.0, 0.0], "degree": 3}
    ],
    "edges": [
      {
        "id": "e0",
        "u": "n0",
        "v": "n1",
        "centerline_local": [[-0.3, -0.5], [0.0, 0.0]],
        "width": 0.04
      }
    ]
  }
}
```

### 6.5 `renderable` 与 `is_reference_only`

`parse_graph.nodes` 中允许出现只用于关系上下文的引用节点。

这类节点必须满足：

```json
{
  "id": "support_ref_0",
  "role": "support_region",
  "label": 5,
  "geometry_model": "none",
  "is_reference_only": true,
  "renderable": false
}
```

语义：

- `is_reference_only=true` 表示该节点不拥有任何 face，只是作为高层关系的上下文端点。
- `renderable=false` 表示生成器和 renderer 不应把该节点当作几何目标生成或渲染。
- 所有 `geometry_model="none"` 的节点默认应视为 `renderable=false`，包括 `insert_object_group` 等分组节点。
- 这类节点可以继续被 `inserted_in`、`divides`、`adjacent_to` 等关系引用。
- 训练目标中的 reference-only 节点不应包含 `frame`、`geometry` 或 `atoms`。
- renderer / training target 构建时应只把 `renderable != false` 的节点作为几何生成目标。

这样可以避免同一 evidence face 被真实 owning node 和 reference-only context node 重复生成。

---

## 7. 核心对象角色

### 7.1 支撑区域：`support_region`

支撑区域表示大背景、大底板、大田块、大平原、大水面等承载其他结构的区域。

典型例子：

```text
一整片农田
一块平原
一大片背景区域
一个主要语义区域
```

字段示例：

```json
{
  "id": "support_0",
  "role": "support_region",
  "label": 1,
  "frame": {
    "origin": [128.0, 128.0],
    "scale": 96.0,
    "orientation": 0.0
  },
  "geometry_model": "polygon_code",
  "geometry": {
    "outer_local": [[-0.5, -0.4], [0.6, -0.3], [0.5, 0.5], [-0.4, 0.6]],
    "holes_local": []
  },
  "evidence": {
    "face_ids": [3, 7, 8],
    "arc_ids": [12, 18, 22]
  }
}
```

### 7.2 分隔区域：`divider_region`

分隔区域表示把其他区域分开、切开或组织起来的结构。

典型例子：

```text
道路切开田地
水渠切开农田
隔板分隔空间
狭长边界带分隔不同语义区域
```

`divider_region` 不强制使用中心线宽度图。它可以用不同几何模型表达：

```text
polygon_code
convex_atoms
boundary_arcs
skeleton_width_graph
```

字段示例：

```json
{
  "id": "divider_0",
  "role": "divider_region",
  "label": 4,
  "frame": {
    "origin": [120.0, 90.0],
    "scale": 80.0,
    "orientation": 0.2
  },
  "geometry_model": "polygon_code",
  "geometry": {
    "outer_local": [[-0.4, -0.1], [0.5, -0.08], [0.5, 0.08], [-0.4, 0.1]],
    "holes_local": []
  },
  "evidence": {
    "face_ids": [12],
    "arc_ids": [33, 34, 35]
  }
}
```

如果证据强烈支持中心线结构，也可以写成：

```json
{
  "id": "divider_0",
  "role": "divider_region",
  "label": 4,
  "geometry_model": "skeleton_width_graph",
  "network": {
    "nodes": [],
    "edges": []
  }
}
```

### 7.3 嵌入物组：`insert_object_group`

嵌入物组表示一组被放置到某个支撑区域中的小对象。

典型例子：

```text
平原上的房子
农田中的小设施
背景中的多个小块建筑
```

字段示例：

```json
{
  "id": "insert_group_0",
  "role": "insert_object_group",
  "support_id": "support_0",
  "label": 3,
  "count": 3,
  "children": ["insert_0", "insert_1", "insert_2"],
  "layout": {
    "relative_positions": [[0.2, 0.3], [0.5, 0.6], [0.7, 0.4]],
    "relative_areas": [0.02, 0.015, 0.018]
  }
}
```

### 7.4 嵌入物：`insert_object`

单个嵌入对象。

字段示例：

```json
{
  "id": "insert_0",
  "role": "insert_object",
  "label": 3,
  "parent_group": "insert_group_0",
  "support_id": "support_0",
  "frame": {
    "origin_local": [0.2, 0.3],
    "scale": 0.08,
    "orientation": 0.1
  },
  "geometry_model": "polygon_code",
  "geometry": {
    "outer_local": [[-0.3, -0.2], [0.3, -0.2], [0.25, 0.25], [-0.25, 0.2]],
    "holes_local": []
  },
  "evidence": {
    "face_ids": [21],
    "arc_ids": [44, 45, 46, 47]
  }
}
```

### 7.5 洞组：`hole_group`

洞组表示某个支撑区域中的多个洞或嵌套结构。

字段示例：

```json
{
  "id": "hole_group_0",
  "role": "hole_group",
  "support_id": "support_0",
  "holes": ["hole_0", "hole_1"]
}
```

### 7.6 洞：`hole`

洞表示支撑区域中被挖掉的局部空间。

注意：

- `hole` 表示没有独立语义对象占据的空洞或负空间。
- 如果被挖出的区域实际由另一个语义对象占据，例如“房子嵌入平原”，应优先使用 `insert_object` 和 `inserted_in`，而不是把平原直接建模成带洞多边形。

字段示例：

```json
{
  "id": "hole_0",
  "role": "hole",
  "support_id": "support_0",
  "geometry_model": "polygon_code",
  "geometry": {
    "outer_local": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]
  },
  "contains_islands": []
}
```

### 7.7 残差区域：`residual_region`

残差区域表示暂时无法由高层模板解释的几何。

字段示例：

```json
{
  "id": "residual_0",
  "role": "residual_region",
  "label": 2,
  "geometry_model": "convex_atoms",
  "atoms": [
    {
      "outer_local": [[10, 20], [14, 20], [12, 24]],
      "type": "triangle"
    }
  ],
  "reason": "no_low_cost_template",
  "evidence": {
    "face_ids": [31]
  }
}
```

说明：

- residual 不代表失败。
- residual 是解释器的安全出口。
- 第一版解释器应该允许较多 residual，避免错误解释。
- residual 仍然必须可渲染、可验证、可追溯。

---

## 8. 关系类型

解释器输出不仅要有节点，还要有关系。

### 8.1 `contains`

表示包含关系。

```json
{
  "type": "contains",
  "parent": "support_0",
  "child": "insert_0"
}
```

### 8.2 `inserted_in`

表示嵌入关系。

```json
{
  "type": "inserted_in",
  "object": "insert_0",
  "support": "support_0"
}
```

`inserted_in` 的渲染语义是覆盖式嵌入：

```text
1. 先渲染 support_region 的完整几何；
2. 再渲染 insert_object；
3. insert_object 在最终 partition / mask 中覆盖对应 support 区域；
4. 被覆盖部分在渲染结果中等效为从 support 中扣除。
```

因此，`parse_graph` 中的 `support_region` 可以是一整块没有洞的区域；只要存在 `inserted_in` 关系，renderer 就应该在最终语义面中让嵌入物占据其位置。

这和 `hole` 不同：

```text
inserted_in = 有语义对象嵌入并覆盖 support
hole        = support 中存在没有独立语义对象的负空间
```

### 8.3 `divides`

表示某个分隔区域组织、切分或分隔某个支撑区域。

```json
{
  "type": "divides",
  "divider": "divider_0",
  "support": "support_0",
  "induced_face_ids": [5, 6, 7, 8]
}
```

### 8.4 `adjacent_to`

表示普通邻接关系。

```json
{
  "type": "adjacent_to",
  "a": "support_0",
  "b": "residual_0",
  "arc_ids": [88]
}
```

### 8.5 `has_residual`

表示某个结构仍带有残差。

```json
{
  "type": "has_residual",
  "owner": "support_0",
  "residual": "residual_0"
}
```

---

## 9. 生成器训练目标格式

当前版本只定义一个正式训练目标：

```text
parse_graph
```

完整示例：

```json
{
  "format": "maskgen_generator_target_v1",
  "target_type": "parse_graph",
  "size": [256, 256],
  "parse_graph": {
    "nodes": [
      {
        "id": "support_0",
        "role": "support_region",
        "label": 1,
        "frame": {
          "origin": [128.0, 128.0],
          "scale": 96.0,
          "orientation": 0.0
        },
        "geometry_model": "polygon_code",
        "geometry": {
          "outer_local": [[-0.5, -0.4], [0.6, -0.3], [0.5, 0.5], [-0.4, 0.6]],
          "holes_local": []
        }
      },
      {
        "id": "divider_0",
        "role": "divider_region",
        "label": 4,
        "frame": {
          "origin": [118.0, 120.0],
          "scale": 90.0,
          "orientation": 0.1
        },
        "geometry_model": "polygon_code",
        "geometry": {
          "outer_local": [[-0.5, -0.05], [0.5, -0.04], [0.5, 0.05], [-0.5, 0.04]],
          "holes_local": []
        }
      },
      {
        "id": "insert_group_0",
        "role": "insert_object_group",
        "support_id": "support_0",
        "label": 3,
        "count": 2,
        "children": ["insert_0", "insert_1"]
      }
    ],
    "relations": [
      {
        "type": "divides",
        "divider": "divider_0",
        "support": "support_0",
        "induced_face_ids": [5, 6, 7, 8]
      },
      {
        "type": "inserted_in",
        "object": "insert_0",
        "support": "support_0"
      }
    ],
    "residuals": []
  },
  "metadata": {
    "source_explanation": "data/remote_256_explanations/val/83.json",
    "code_length": 128.4,
    "render_iou": 0.997,
    "valid": true
  }
}
```

### 9.1 关于 `program_sequence`

`program_sequence` 暂时不是正式训练目标。

如果后续需要自回归模型，可以从 `parse_graph` 通过确定性序列化得到：

```text
parse_graph
  -> canonical node ordering
  -> canonical relation ordering
  -> program_sequence
```

因此当前阶段不维护两套等价主表示，避免协议膨胀和不一致。

---

## 10. 生成器输出格式

生成器的输出不应该直接是最终 PNG，而应该先输出结构表示。

推荐输出链路：

```text
generator output
  -> generated parse_graph
  -> renderer
  -> generated global partition
  -> raster mask
```

因此生成器输出可以有三层。

### 10.1 第一层输出：生成的结构目标

```json
{
  "format": "maskgen_generator_sample_v1",
  "sample_id": "sample_000001",
  "target_type": "parse_graph",
  "parse_graph": {
    "nodes": [],
    "relations": [],
    "residuals": []
  },
  "sampling_metadata": {
    "model": "graph_transformer_v1",
    "seed": 123,
    "temperature": 0.8
  }
}
```

### 10.2 第二层输出：渲染后的全图 partition

```json
{
  "format": "maskgen_rendered_partition_v1",
  "source_sample": "sample_000001",
  "size": [256, 256],
  "arcs": [],
  "faces": [],
  "validation": {
    "is_valid": true,
    "all_faces_valid": true,
    "overlap_area": 0.0,
    "gap_area": 0.0,
    "face_count": 28,
    "arc_count": 64
  }
}
```

### 10.3 第三层输出：最终 mask

最终 mask 是二维整数矩阵，或保存为 PNG。

```json
{
  "format": "maskgen_generated_mask_v1",
  "source_sample": "sample_000001",
  "size": [256, 256],
  "mask_path": "outputs/generated/sample_000001.png",
  "class_map": {
    "0": "background",
    "1": "field",
    "2": "road",
    "3": "building"
  }
}
```

---

## 11. 规范化要求

### 11.1 坐标归一化

高层对象内部尽量使用局部坐标。

每个主要对象都有 `frame`：

```json
{
  "origin": [cx, cy],
  "scale": s,
  "orientation": theta
}
```

该对象内部的点用局部坐标：

```text
x_local = R(-theta) * (x - cx) / s
```

其中 `R(-theta)` 表示旋转到局部坐标系。

第一版建议：

- `origin` 使用对象几何中心或最小旋转矩形中心。
- `scale` 使用最小旋转矩形长边长度，或其他固定规则。
- `orientation` 使用最小旋转矩形长轴方向。

具体规则必须在实现中固定，不能对同类对象随机选择。

### 11.2 节点排序必须固定

为了避免同一个结构有多种等价序列，解释器导出的节点必须有固定顺序。

建议：

```text
support_region:
  按面积从大到小

divider_region:
  按面积或共享边界总长度从大到小

insert_object_group:
  按所属 support_id，再按 label，再按 count

insert_object:
  按局部 y，再按局部 x

residual_region:
  按面积从大到小
```

### 11.3 关系排序必须固定

建议顺序：

```text
divides
inserted_in
contains
adjacent_to
has_residual
```

同类关系内按节点 id 排序。

### 11.4 ID 必须局部可重建

训练目标中的 ID 应该是规范化 ID：

```text
support_0
support_1
divider_0
insert_group_0
insert_0
residual_0
```

不要直接使用原始 face id 作为生成器 ID。

原始 face id 只放在 `evidence` 里。

### 11.5 必须保存 evidence，但训练时可忽略

解释器输出的人类调试文件中应该保留：

```text
face_ids
arc_ids
atom_ids
score
cost breakdown
```

但训练 generator 时可以剥离这些字段。

---

## 12. 解释器应该怎样导出 generator target

解释器输出应该包含：

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
  "diagnostics": {}
}
```

其中：

- `selected_explanations` 是解释器自己的完整输出。
- `generator_target.parse_graph` 是给生成器学习的图内容。
- `generator_target.metadata` 记录训练、渲染和验证相关元数据。
- `diagnostics` 包含代价、残差、稳定性、渲染验证等信息。

---

## 13. 解释器候选解释格式

解释器应该对每个局部 patch 输出 top-k 候选。

```json
{
  "patch_id": "patch_012",
  "evidence": {
    "face_ids": [5, 6, 7, 12],
    "arc_ids": [20, 21, 22]
  },
  "selected_candidate_id": "candidate_0",
  "score_gap": 18.4,
  "candidates": [
    {
      "id": "candidate_0",
      "template": "split_by_divider",
      "nodes": ["support_0", "divider_0"],
      "relations": [
        {
          "type": "divides",
          "divider": "divider_0",
          "support": "support_0"
        }
      ],
      "cost": {
        "total": 31.2,
        "template": 3.0,
        "topology": 6.0,
        "geometry": 18.0,
        "residual": 4.2,
        "invalid": 0.0
      }
    },
    {
      "id": "candidate_1",
      "template": "independent_faces",
      "cost": {
        "total": 58.1
      }
    }
  ]
}
```

---

## 14. 代价字段

解释器应该为每个候选解释输出代价分解。

```json
{
  "cost": {
    "total": 42.7,
    "template": 3.0,
    "topology": 8.0,
    "geometry": 20.0,
    "residual": 6.0,
    "invalid": 0.0,
    "evidence_bonus": -1.3
  }
}
```

含义：

- `template`：使用某个模板的固定代价。
- `topology`：对象数、关系数、网络节点数、网络边数等。
- `geometry`：多边形顶点数、中心线点数、参数数等。
- `residual`：模板解释不了的剩余区域。
- `invalid`：非法几何或拓扑的惩罚。
- `evidence_bonus`：强证据带来的奖励，例如细长区域支持 divider。

---

## 15. 第一版支持的模板

第一版解释器只需要支持三个模板。

### 15.1 `support_with_inserts`

含义：

```text
一个大支撑区域中嵌入若干小对象
```

输出节点：

```text
support_region
insert_object_group
insert_object
```

关系：

```text
inserted_in
contains
```

### 15.2 `split_by_divider`

含义：

```text
一个支撑区域被某种分隔区域切分或组织
```

输出节点：

```text
support_region
divider_region
```

关系：

```text
divides
```

说明：

- `divider_region` 可以是道路，也可以是其他数据集中的分隔结构。
- 它不强制使用中心线宽度图。
- 第一版可以优先使用 `polygon_code` 或 `convex_atoms`，以后再为特定场景增加 `skeleton_width_graph`。

### 15.3 `independent_faces`

含义：

```text
无法用高层模板解释，保持独立 face / residual
```

输出节点：

```text
residual_region
```

关系：

```text
adjacent_to
```

---

## 16. 解释器设计的核心目标

解释器不是为了让几何块数最少。

解释器的目标是：

```text
把低层 faces / arcs / convex atoms
压缩成更短、更稳定、更可生成的 parse_graph
```

因此，解释器好坏应该用以下指标衡量：

```text
1. 生成目标 token 数是否减少
2. code length 是否低于低层 atom baseline
3. 渲染回 mask 是否有效
4. 拓扑是否合法
5. 残差比例是否可控
6. top-1 与 top-2 解释的 score gap 是否足够大
7. 同类结构是否得到稳定解释
```

---

## 17. 不变量

任何 generator target 都必须满足：

```text
1. 可渲染
2. 可验证
3. 可序列化
4. 可训练
5. 可追溯到 evidence
```

具体要求：

- 所有主要节点有 `role` 和 `label`。
- 所有主要节点有明确的 `geometry_model`。
- 所有几何点必须在合法坐标范围内。
- 所有关系引用存在的节点。
- residual 可以为空，但字段必须存在。
- metadata 中必须记录 valid / render_iou / code_length。
- `parse_graph` 应该能被 renderer 转成合法 partition 或明确失败。

---

## 18. 推荐文件路径

建议解释器最终输出：

```text
data/remote_256_explanations/<split>/graphs/<stem>.json
```

建议 generator target 输出：

```text
data/remote_256_generator_targets/<split>/graphs/<stem>.json
```

其中每个 generator target JSON 包含：

```json
{
  "format": "maskgen_generator_target_v1",
  "target_type": "parse_graph",
  "size": [256, 256],
  "source_explanation": "...",
  "parse_graph": {
    "nodes": [],
    "relations": [],
    "residuals": []
  },
  "metadata": {
    "render_validation": {},
    "training_metadata": {}
  }
}
```

---

## 19. 给 Codex 的实现方向

Codex 在设计解释器时，应该从这个目标反推过程：

```text
解释器最终必须导出 maskgen_generator_target_v1。
所以解释器需要：
  1. 从 explanation evidence 中找候选 patch；
  2. 为每个 patch 生成模板候选；
  3. 计算每个候选的 code length；
  4. 选择低代价解释；
  5. 把解释合并成 scene-level parse_graph；
  6. 导出 parse_graph；
  7. 验证它能重新渲染成合法 mask / partition。
```

不要从“如何继续合并多边形”出发。

要从“生成器最终应该学习什么解析图”出发。

---

## 20. 当前版本的明确边界

当前版本不做以下事情：

- 不把 `program_sequence` 作为正式训练目标。
- 不强制使用 `centerline + width + graph topology`。
- 不把遥感道路作为协议核心假设。
- 不要求第一版解释器解释所有区域。
- 不要求所有对象都脱离 residual。

当前版本必须做到：

- `parse_graph` 是唯一正式训练目标。
- `role`、`label`、`geometry_model` 三者分离。
- residual 是合法输出。
- 所有目标必须可追溯到 evidence。
- 所有目标必须可以被 renderer 验证。

---

## 21. 弱解释器 profile：`weak_convex_face_atoms_v1`

当强解释器需要大量规则才能判断 `support_region`、`divider_region`、`insert_object` 时，可以先使用更保守的弱解释器 profile。

该 profile 让生成器学习：

```text
全图 semantic faces
每个 face 的 label
每个 face 的 convex atoms
face adjacency
label groups
```

### 21.1 节点 role

弱 profile 使用以下结构中性 role：

```text
label_group
semantic_face
convex_atom
```

示例：

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
    "area": 120.0
  }
}
```

### 21.2 关系类型

弱 profile 使用：

```text
label_group_contains
atom_part_of
face_adjacent
```

示例：

```json
{
  "type": "atom_part_of",
  "atom": "atom_0",
  "face": "face_0"
}
```

```json
{
  "type": "face_adjacent",
  "faces": ["face_0", "face_1"],
  "source_face_ids": [12, 18],
  "labels": [6, 2],
  "arc_ids": [33],
  "shared_length": 42.0
}
```

### 21.3 使用场景

弱 profile 的优点：

- 不依赖数据集专用语义。
- 不需要判断道路、房屋、田地等类别的固定 role。
- 仍然利用凸分割器，把复杂 face 拆成简单凸多边形。
- 保留全图 adjacency 和共享边界 evidence。

弱 profile 的缺点：

- 序列通常比强解释器长。
- 不直接表达“嵌入”“分隔”等高层生成原因。
- 后续如果需要更强压缩，可以在弱 profile 之上再训练或搜索强解释模板。

### 21.4 渲染验证

弱 profile 可以通过 `convex_atom` 闭环验证：

```text
convex_atom local polygons
  -> world-space polygons
  -> per-face atom union
  -> rendered partition
  -> compare against evidence geometry
```

推荐记录：

```json
{
  "format": "maskgen_weak_rendered_partition_v1",
  "validation": {
    "is_valid": true,
    "full_iou": 1.0,
    "overlap_area": 0.0,
    "gap_area": 0.0,
    "low_iou_face_ids": []
  }
}
```

这里的 `full_iou` 是 rendered atom union 与 evidence geometry 的 IoU。它不是和原始像素 mask 的 IoU。
