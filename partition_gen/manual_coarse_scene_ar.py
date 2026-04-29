from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import copy
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence

import torch

from partition_gen.manual_geometry_constrained_sampling import _sample_from_logits
from partition_gen.manual_geometry_conditioning import iter_jsonl
from partition_gen.manual_layout_residual import geometry_renderable_local_points
from partition_gen.manual_layout_retrieval import load_split_row, write_jsonl
from partition_gen.parse_graph_relations import divides_target, inserted_in_container
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    TokenReader,
    _manual_geometry_token,
    _manual_role_token,
    dequantize,
    int_token,
    q_token,
    token_int,
    token_q,
    tokens_to_ids,
)


COARSE_SCENE_START_TOKEN = "MANUAL_COARSE_SCENE_V1"
COARSE_SCENE_TARGET_TOKEN = "COARSE_SCENE_TARGET"
EPS = 1.0e-6

ACTION_TO_ROLE = {
    "ACTION_SUPPORT": {"support_region", "residual_region"},
    "ACTION_ADJACENT_SUPPORT": {"support_region", "residual_region"},
    "ACTION_INSERT_GROUP": {"insert_object_group"},
    "ACTION_INSERT": {"insert_object"},
    "ACTION_DIVIDER": {"divider_region"},
}
ROLE_TOKEN_TO_ROLE = {
    "ROLE_SUPPORT": "support_region",
    "ROLE_DIVIDER": "divider_region",
    "ROLE_INSERT": "insert_object",
    "ROLE_INSERT_GROUP": "insert_object_group",
    "ROLE_RESIDUAL": "residual_region",
    "ROLE_UNKNOWN": "unknown",
}
GEOMETRY_TOKEN_TO_MODEL = {
    "GEOM_NONE": "none",
    "GEOM_POLYGON_CODE": "polygon_code",
    "GEOM_CONVEX_ATOMS": "convex_atoms",
    "GEOM_UNKNOWN": "unknown",
}
ROLE_ID_PREFIX = {
    "support_region": "support",
    "residual_region": "residual",
    "insert_object_group": "insert_group",
    "insert_object": "insert",
    "divider_region": "divider",
}


@dataclass(frozen=True)
class CoarseSceneSamplerConfig:
    tokenizer_config: ParseGraphTokenizerConfig = field(default_factory=ParseGraphTokenizerConfig)
    max_actions: int = 256
    max_label: int = 6
    allowed_action_tokens: tuple[str, ...] = (
        "ACTION_SUPPORT",
        "ACTION_ADJACENT_SUPPORT",
        "ACTION_INSERT_GROUP",
        "ACTION_INSERT",
        "ACTION_DIVIDER",
    )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tokenizer_config"] = asdict(self.tokenizer_config)
        return payload


def _wrap_angle(value: float) -> float:
    wrapped = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(float(high), max(float(low), float(value))))


def _canvas_size(target: dict) -> tuple[float, float]:
    size = target.get("size", [256, 256]) or [256, 256]
    width = float(size[0]) if len(size) >= 1 else 256.0
    height = float(size[1]) if len(size) >= 2 else width
    return max(EPS, width), max(EPS, height)


def _bbox_metrics(bbox: Sequence[float]) -> dict:
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    width = max(EPS, max_x - min_x)
    height = max(EPS, max_y - min_y)
    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width": float(width),
        "height": float(height),
        "center_x": float((min_x + max_x) / 2.0),
        "center_y": float((min_y + max_y) / 2.0),
        "area": float(width * height),
    }


def _bbox_from_center_size(center_x: float, center_y: float, width: float, height: float) -> list[float]:
    half_w = max(EPS, float(width)) / 2.0
    half_h = max(EPS, float(height)) / 2.0
    return [float(center_x - half_w), float(center_y - half_h), float(center_x + half_w), float(center_y + half_h)]


def _union_bboxes(values: Sequence[Sequence[float]]) -> list[float] | None:
    bboxes = [list(value) for value in values if value is not None]
    if not bboxes:
        return None
    return [
        float(min(bbox[0] for bbox in bboxes)),
        float(min(bbox[1] for bbox in bboxes)),
        float(max(bbox[2] for bbox in bboxes)),
        float(max(bbox[3] for bbox in bboxes)),
    ]


def _frame_origin(frame: dict) -> tuple[float, float]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return float(origin[0]), float(origin[1])


def _fallback_bbox_from_frame(frame: dict | None) -> list[float]:
    frame = frame or {}
    origin_x, origin_y = _frame_origin(frame)
    scale = max(1.0, float(frame.get("scale", 1.0)))
    return [origin_x - scale / 2.0, origin_y - scale / 2.0, origin_x + scale / 2.0, origin_y + scale / 2.0]


def world_bbox_from_geometry_target(geometry_target: dict) -> list[float]:
    points = geometry_renderable_local_points(geometry_target)
    if not points:
        return _fallback_bbox_from_frame(geometry_target.get("frame", {}))
    frame = geometry_target.get("frame", {}) or {}
    origin_x, origin_y = _frame_origin(frame)
    scale = max(EPS, float(frame.get("scale", 1.0)))
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    world: list[tuple[float, float]] = []
    for local_x, local_y in points:
        x = float(local_x) * scale
        y = float(local_y) * scale
        world.append((origin_x + x * cos_theta - y * sin_theta, origin_y + x * sin_theta + y * cos_theta))
    xs = [point[0] for point in world]
    ys = [point[1] for point in world]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _relation_maps(topology_target: dict) -> dict:
    contains_parent_by_child: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    inserted_container_by_object: dict[str, str] = {}
    divides_targets_by_divider: dict[str, list[str]] = defaultdict(list)
    adjacent_by_node: dict[str, list[str]] = defaultdict(list)
    for relation in (topology_target.get("parse_graph", {}) or {}).get("relations", []) or []:
        relation_type = str(relation.get("type", ""))
        if relation_type == "contains":
            parent = relation.get("parent")
            child = relation.get("child")
            if parent is not None and child is not None:
                contains_parent_by_child[str(child)] = str(parent)
                children_by_parent[str(parent)].append(str(child))
        elif relation_type == "inserted_in":
            obj = relation.get("object")
            container = inserted_in_container(relation)
            if obj is not None and container is not None:
                inserted_container_by_object[str(obj)] = str(container)
        elif relation_type == "divides":
            divider = relation.get("divider")
            target = divides_target(relation)
            if divider is not None and target is not None:
                divides_targets_by_divider[str(divider)].append(str(target))
        elif relation_type == "adjacent_to":
            faces = [str(value) for value in relation.get("faces", []) or []]
            if len(faces) >= 2:
                left, right = faces[0], faces[1]
                adjacent_by_node[left].append(right)
                adjacent_by_node[right].append(left)
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    for node in nodes:
        node_id = str(node.get("id", ""))
        if str(node.get("role", "")) == "insert_object_group":
            for child in node.get("children", []) or []:
                children_by_parent[node_id].append(str(child))
                contains_parent_by_child.setdefault(str(child), node_id)
    return {
        "contains_parent_by_child": contains_parent_by_child,
        "children_by_parent": {key: list(dict.fromkeys(values)) for key, values in children_by_parent.items()},
        "inserted_container_by_object": inserted_container_by_object,
        "divides_targets_by_divider": {key: list(dict.fromkeys(values)) for key, values in divides_targets_by_divider.items()},
        "adjacent_by_node": {key: list(dict.fromkeys(values)) for key, values in adjacent_by_node.items()},
    }


def _node_bboxes(topology_target: dict, geometry_targets: Sequence[dict]) -> dict[str, list[float]]:
    graph = topology_target.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    maps = _relation_maps(topology_target)
    cache: dict[str, list[float]] = {}

    def resolve(node_id: str, stack: set[str] | None = None) -> list[float]:
        if node_id in cache:
            return cache[node_id]
        stack = set() if stack is None else set(stack)
        if node_id in stack:
            cache[node_id] = _fallback_bbox_from_frame({})
            return cache[node_id]
        stack.add(node_id)
        node = node_by_id.get(node_id, {})
        geometry_ref = node.get("geometry_ref", node_id)
        geometry = geometry_by_id.get(str(geometry_ref))
        if geometry is not None:
            cache[node_id] = world_bbox_from_geometry_target(geometry)
            return cache[node_id]
        child_bboxes = [
            resolve(str(child), stack)
            for child in maps["children_by_parent"].get(node_id, [])
            if str(child) in node_by_id
        ]
        union = _union_bboxes(child_bboxes)
        if union is not None:
            cache[node_id] = union
            return union
        cache[node_id] = _fallback_bbox_from_frame(node.get("frame", {}))
        return cache[node_id]

    for node_id in node_by_id:
        resolve(node_id)
    return cache


def _frame_for_node(node: dict, geometry_by_id: dict[str, dict], bbox: Sequence[float]) -> dict:
    geometry_ref = node.get("geometry_ref", node.get("id"))
    geometry = geometry_by_id.get(str(geometry_ref))
    if geometry is not None and "frame" in geometry:
        frame = copy.deepcopy(geometry["frame"])
        metrics = _bbox_metrics(bbox)
        frame.setdefault("origin", [metrics["center_x"], metrics["center_y"]])
        frame.setdefault("scale", max(metrics["width"], metrics["height"]))
        frame.setdefault("orientation", 0.0)
        return frame
    if "frame" in node:
        return copy.deepcopy(node["frame"])
    metrics = _bbox_metrics(bbox)
    return {
        "origin": [float(metrics["center_x"]), float(metrics["center_y"])],
        "scale": float(max(metrics["width"], metrics["height"])),
        "orientation": 0.0,
    }


def _node_dependencies(topology_target: dict) -> tuple[dict[str, set[str]], dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    node_ids = {str(node.get("id")) for node in nodes if node.get("id") is not None}
    maps = _relation_maps(topology_target)
    dependencies: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    missing_dependency_count = 0
    for node in nodes:
        node_id = str(node.get("id", ""))
        role = str(node.get("role", ""))
        candidate: str | None = None
        if role == "insert_object_group":
            candidate = maps["inserted_container_by_object"].get(node_id)
        elif role == "insert_object":
            candidate = maps["contains_parent_by_child"].get(node_id) or maps["inserted_container_by_object"].get(node_id)
        elif role == "divider_region":
            for candidate in maps["divides_targets_by_divider"].get(node_id, []):
                if candidate in node_ids and candidate != node_id:
                    dependencies[node_id].add(candidate)
                else:
                    missing_dependency_count += 1
            candidate = None
        if candidate is not None:
            if candidate in node_ids and candidate != node_id:
                dependencies[node_id].add(candidate)
            else:
                missing_dependency_count += 1
    return dependencies, {"missing_dependency_count": int(missing_dependency_count)}


def parent_first_node_order(topology_target: dict) -> tuple[list[int], dict]:
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    node_index_by_id = {str(node.get("id")): index for index, node in enumerate(nodes) if node.get("id") is not None}
    dependencies, diagnostics = _node_dependencies(topology_target)
    dependents: dict[str, set[str]] = defaultdict(set)
    remaining_deps = {node_id: set(values) for node_id, values in dependencies.items()}
    for node_id, deps in remaining_deps.items():
        for dep in deps:
            dependents[dep].add(node_id)
    ready = deque(sorted([node_id for node_id, deps in remaining_deps.items() if not deps], key=lambda value: node_index_by_id[value]))
    emitted: list[str] = []
    emitted_set: set[str] = set()
    cycle_fallback_count = 0
    while len(emitted) < len(remaining_deps):
        if not ready:
            unresolved = [node_id for node_id in remaining_deps if node_id not in emitted_set]
            if not unresolved:
                break
            node_id = sorted(unresolved, key=lambda value: node_index_by_id[value])[0]
            cycle_fallback_count += 1
        else:
            node_id = ready.popleft()
            if node_id in emitted_set:
                continue
        emitted.append(node_id)
        emitted_set.add(node_id)
        for dependent in sorted(dependents.get(node_id, []), key=lambda value: node_index_by_id[value]):
            if dependent in emitted_set:
                continue
            remaining_deps[dependent].discard(node_id)
            if not remaining_deps[dependent]:
                ready.append(dependent)
        ready = deque(sorted(dict.fromkeys(ready), key=lambda value: node_index_by_id[value]))
    order = [int(node_index_by_id[node_id]) for node_id in emitted if node_id in node_index_by_id]
    diagnostics.update(
        {
            "dependency_fallback_count": int(cycle_fallback_count + diagnostics.get("missing_dependency_count", 0)),
            "cycle_fallback_count": int(cycle_fallback_count),
            "forward_reference_count": 0,
        }
    )
    return order, diagnostics


def _abs_bbox_values(bbox: Sequence[float], *, canvas_width: float, canvas_height: float, orientation: float) -> dict:
    metrics = _bbox_metrics(bbox)
    aspect = math.log(max(EPS, metrics["width"]) / max(EPS, metrics["height"]))
    return {
        "center_x_norm": metrics["center_x"] / max(EPS, canvas_width),
        "center_y_norm": metrics["center_y"] / max(EPS, canvas_height),
        "width_norm": metrics["width"] / max(EPS, canvas_width),
        "height_norm": metrics["height"] / max(EPS, canvas_height),
        "log_aspect": aspect,
        "orientation": _wrap_angle(orientation),
    }


def _rel_bbox_values(bbox: Sequence[float], anchor_bbox: Sequence[float], *, orientation: float) -> dict:
    metrics = _bbox_metrics(bbox)
    anchor = _bbox_metrics(anchor_bbox)
    aspect = math.log(max(EPS, metrics["width"]) / max(EPS, metrics["height"]))
    return {
        "center_x_rel": (metrics["center_x"] - anchor["min_x"]) / max(EPS, anchor["width"]),
        "center_y_rel": (metrics["center_y"] - anchor["min_y"]) / max(EPS, anchor["height"]),
        "width_rel": metrics["width"] / max(EPS, anchor["width"]),
        "height_rel": metrics["height"] / max(EPS, anchor["height"]),
        "log_aspect": aspect,
        "orientation": _wrap_angle(orientation),
    }


def _append_abs_frame_tokens(tokens: list[str], values: dict, *, config: ParseGraphTokenizerConfig) -> None:
    tokens.extend(
        [
            "FRAME_ABS_COARSE",
            q_token(values["center_x_norm"], low=0.0, high=1.0, bins=config.coarse_grid_bins),
            q_token(values["center_y_norm"], low=0.0, high=1.0, bins=config.coarse_grid_bins),
            q_token(values["width_norm"], low=0.0, high=1.5, bins=config.coarse_size_bins),
            q_token(values["height_norm"], low=0.0, high=1.5, bins=config.coarse_size_bins),
            q_token(values["log_aspect"], low=-2.0, high=2.0, bins=config.coarse_aspect_bins),
            q_token(values["orientation"], low=-math.pi, high=math.pi, bins=config.coarse_angle_bins),
        ]
    )


def _append_rel_frame_tokens(tokens: list[str], values: dict, *, config: ParseGraphTokenizerConfig) -> None:
    tokens.extend(
        [
            "FRAME_REL_COARSE",
            q_token(values["center_x_rel"], low=-1.0, high=2.0, bins=config.coarse_grid_bins),
            q_token(values["center_y_rel"], low=-1.0, high=2.0, bins=config.coarse_grid_bins),
            q_token(values["width_rel"], low=0.0, high=2.0, bins=config.coarse_size_bins),
            q_token(values["height_rel"], low=0.0, high=2.0, bins=config.coarse_size_bins),
            q_token(values["log_aspect"], low=-2.0, high=2.0, bins=config.coarse_aspect_bins),
            q_token(values["orientation"], low=-math.pi, high=math.pi, bins=config.coarse_angle_bins),
        ]
    )


def _intersection_bbox(left: Sequence[float], right: Sequence[float]) -> list[float] | None:
    min_x = max(float(left[0]), float(right[0]))
    min_y = max(float(left[1]), float(right[1]))
    max_x = min(float(left[2]), float(right[2]))
    max_y = min(float(left[3]), float(right[3]))
    if max_x <= min_x or max_y <= min_y:
        return None
    return [min_x, min_y, max_x, max_y]


def _bbox_gap(left: Sequence[float], right: Sequence[float]) -> float:
    dx = max(float(right[0]) - float(left[2]), float(left[0]) - float(right[2]), 0.0)
    dy = max(float(right[1]) - float(left[3]), float(left[1]) - float(right[3]), 0.0)
    return float(math.hypot(dx, dy))


def _bbox_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    inter = _intersection_bbox(left, right)
    if inter is None:
        return 0.0
    return float(_bbox_metrics(inter)["area"] / max(EPS, min(_bbox_metrics(left)["area"], _bbox_metrics(right)["area"])))


def _bbox_canvas_overflow(bbox: Sequence[float], bounds: Sequence[float]) -> float:
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    bmin_x, bmin_y, bmax_x, bmax_y = [float(value) for value in bounds]
    overflow = max(0.0, bmin_x - min_x) + max(0.0, bmin_y - min_y) + max(0.0, max_x - bmax_x) + max(0.0, max_y - bmax_y)
    scale = max(EPS, max(bmax_x - bmin_x, bmax_y - bmin_y))
    return float(overflow / scale)


def _frame_from_bbox(bbox: Sequence[float], *, orientation: float) -> dict:
    metrics = _bbox_metrics(bbox)
    return {
        "origin": [float(metrics["center_x"]), float(metrics["center_y"])],
        "scale": float(max(metrics["width"], metrics["height"])),
        "orientation": float(_wrap_angle(orientation)),
    }


def _with_bbox(decoded: dict, bbox: Sequence[float], *, orientation: float | None = None) -> dict:
    output = copy.deepcopy(decoded)
    theta = float(output.get("frame", {}).get("orientation", 0.0) if orientation is None else orientation)
    output["bbox"] = [float(value) for value in bbox]
    output["frame"] = _frame_from_bbox(output["bbox"], orientation=theta)
    return output


def _shift_bbox(bbox: Sequence[float], dx: float, dy: float) -> list[float]:
    return [
        float(bbox[0]) + float(dx),
        float(bbox[1]) + float(dy),
        float(bbox[2]) + float(dx),
        float(bbox[3]) + float(dy),
    ]


def _transform_bbox_between(bbox: Sequence[float], source_bbox: Sequence[float], target_bbox: Sequence[float]) -> list[float]:
    source = _bbox_metrics(source_bbox)
    target = _bbox_metrics(target_bbox)
    scale_x = target["width"] / max(EPS, source["width"])
    scale_y = target["height"] / max(EPS, source["height"])
    return [
        float(target["min_x"] + (float(bbox[0]) - source["min_x"]) * scale_x),
        float(target["min_y"] + (float(bbox[1]) - source["min_y"]) * scale_y),
        float(target["min_x"] + (float(bbox[2]) - source["min_x"]) * scale_x),
        float(target["min_y"] + (float(bbox[3]) - source["min_y"]) * scale_y),
    ]


def _bbox_center_in_bbox(child: Sequence[float], container: Sequence[float]) -> bool:
    child_metrics = _bbox_metrics(child)
    return bool(
        float(container[0]) <= child_metrics["center_x"] <= float(container[2])
        and float(container[1]) <= child_metrics["center_y"] <= float(container[3])
    )


def _bbox_containment_ratio(child: Sequence[float], container: Sequence[float]) -> float:
    inter = _intersection_bbox(child, container)
    if inter is None:
        return 0.0
    return float(_bbox_metrics(inter)["area"] / max(EPS, _bbox_metrics(child)["area"]))


def _adjacent_overlap_span_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    left_metrics = _bbox_metrics(left)
    right_metrics = _bbox_metrics(right)
    horizontal = abs(left_metrics["center_x"] - right_metrics["center_x"]) >= abs(left_metrics["center_y"] - right_metrics["center_y"])
    if horizontal:
        span = max(0.0, min(left_metrics["max_y"], right_metrics["max_y"]) - max(left_metrics["min_y"], right_metrics["min_y"]))
        denom = min(left_metrics["height"], right_metrics["height"])
    else:
        span = max(0.0, min(left_metrics["max_x"], right_metrics["max_x"]) - max(left_metrics["min_x"], right_metrics["min_x"]))
        denom = min(left_metrics["width"], right_metrics["width"])
    return float(span / max(EPS, denom))


def _set_node_coarse_bbox(node: dict, bbox: Sequence[float]) -> None:
    values = [float(value) for value in bbox]
    orientation = float((node.get("frame", {}) or {}).get("orientation", 0.0))
    node["coarse_bbox"] = values
    node["frame"] = _frame_from_bbox(values, orientation=orientation)


def _clamp_bbox_inside(bbox: Sequence[float], container_bbox: Sequence[float], *, padding: float = 0.0) -> list[float]:
    child = _bbox_metrics(bbox)
    container = _bbox_metrics(container_bbox)
    pad = max(0.0, min(float(padding), container["width"] / 2.0 - 0.5, container["height"] / 2.0 - 0.5))
    min_x = container["min_x"] + pad
    min_y = container["min_y"] + pad
    max_x = container["max_x"] - pad
    max_y = container["max_y"] - pad
    available_w = max(1.0, max_x - min_x)
    available_h = max(1.0, max_y - min_y)
    width = min(max(1.0, child["width"]), available_w)
    height = min(max(1.0, child["height"]), available_h)
    center_x = _clamp(child["center_x"], min_x + width / 2.0, max_x - width / 2.0)
    center_y = _clamp(child["center_y"], min_y + height / 2.0, max_y - height / 2.0)
    return _bbox_from_center_size(center_x, center_y, width, height)


def _clamp_bbox_to_bounds(bbox: Sequence[float], bounds: Sequence[float]) -> list[float]:
    return _clamp_bbox_inside(bbox, bounds, padding=0.0)


def _adjacent_bbox_from_anchor(
    bbox: Sequence[float],
    anchor_bbox: Sequence[float],
    *,
    side_bucket: int,
    gap_bucket: int,
    config: ParseGraphTokenizerConfig,
    existing_bboxes: Sequence[Sequence[float]] | None = None,
    canvas_bounds: Sequence[float] | None = None,
) -> list[float]:
    child = _bbox_metrics(bbox)
    anchor = _bbox_metrics(anchor_bbox)
    width = max(1.0, child["width"])
    height = max(1.0, child["height"])
    if canvas_bounds is not None:
        canvas = _bbox_metrics(canvas_bounds)
        width = min(width, canvas["width"])
        height = min(height, canvas["height"])
    side = max(0, min(int(side_bucket), 3))
    decoded_gap = dequantize(int(gap_bucket), low=0.0, high=1.0, bins=config.coarse_relation_bins)
    gap = min(4.0, decoded_gap * max(anchor["width"], anchor["height"]))
    peer_bboxes = [list(value) for value in existing_bboxes or [] if value is not None]

    def cross_centers(axis: str) -> list[float]:
        if axis == "y":
            span = min(height, anchor["height"])
            low = anchor["min_y"] + span / 2.0
            high = anchor["max_y"] - span / 2.0
            decoded = child["center_y"]
        else:
            span = min(width, anchor["width"])
            low = anchor["min_x"] + span / 2.0
            high = anchor["max_x"] - span / 2.0
            decoded = child["center_x"]
        if high < low:
            high = low
        centers = [
            _clamp(decoded, low, high),
            (low + high) / 2.0,
            low,
            high,
            low + (high - low) * 0.25,
            low + (high - low) * 0.75,
        ]
        return list(dict.fromkeys(float(value) for value in centers))

    def make_candidate(candidate_side: int, cross_center: float) -> list[float]:
        if candidate_side == 0:
            return _bbox_from_center_size(anchor["max_x"] + gap + width / 2.0, cross_center, width, height)
        if candidate_side == 1:
            return _bbox_from_center_size(anchor["min_x"] - gap - width / 2.0, cross_center, width, height)
        if candidate_side == 2:
            return _bbox_from_center_size(cross_center, anchor["max_y"] + gap + height / 2.0, width, height)
        return _bbox_from_center_size(cross_center, anchor["min_y"] - gap - height / 2.0, width, height)

    side_order = [side, *[candidate for candidate in (0, 1, 2, 3) if candidate != side]]
    candidates: list[tuple[int, list[float]]] = []
    for candidate_side in side_order:
        axis = "y" if candidate_side in {0, 1} else "x"
        for center in cross_centers(axis):
            candidates.append((candidate_side, make_candidate(candidate_side, center)))

    def score(candidate: tuple[int, list[float]]) -> tuple[float, float, float, int, float]:
        candidate_side, candidate_bbox = candidate
        anchor_overlap = _bbox_overlap_ratio(candidate_bbox, anchor_bbox)
        relation_gap = _bbox_gap(candidate_bbox, anchor_bbox)
        peer_overlap = max((_bbox_overlap_ratio(candidate_bbox, peer) for peer in peer_bboxes), default=0.0)
        overflow = 0.0 if canvas_bounds is None else _bbox_canvas_overflow(candidate_bbox, canvas_bounds)
        relation_penalty = max(0.0, relation_gap - 4.0) + max(0.0, anchor_overlap - 0.1) * 10.0
        peer_penalty = max(0.0, peer_overlap - 0.1) * 10.0
        return (relation_penalty, peer_penalty, overflow, 0 if candidate_side == side else 1, peer_overlap)

    return [float(value) for value in min(candidates, key=score)[1]]


def _divider_bbox_from_anchor(
    bbox: Sequence[float],
    anchor_bbox: Sequence[float],
    *,
    axis_bucket: int,
    thickness_bucket: int,
    config: ParseGraphTokenizerConfig,
) -> list[float]:
    child = _bbox_metrics(bbox)
    anchor = _bbox_metrics(anchor_bbox)
    axis = max(0, min(int(axis_bucket), 3))
    thickness_ratio = dequantize(int(thickness_bucket), low=0.0, high=1.0, bins=config.coarse_relation_bins)
    thickness = max(1.0, min(max(anchor["width"], anchor["height"]) * 0.08, max(anchor["width"], anchor["height"]) * thickness_ratio))
    center_x = _clamp(child["center_x"], anchor["min_x"], anchor["max_x"])
    center_y = _clamp(child["center_y"], anchor["min_y"], anchor["max_y"])
    if axis == 1:
        width = min(max(1.0, thickness), anchor["width"])
        height = max(1.0, anchor["height"])
    else:
        width = max(1.0, anchor["width"])
        height = min(max(1.0, thickness), anchor["height"])
    return _bbox_from_center_size(center_x, center_y, width, height)


def _repair_decoded_frame(
    action_token: str,
    decoded: dict,
    *,
    anchor_bbox: Sequence[float] | None,
    relation_extra: dict,
    config: ParseGraphTokenizerConfig,
    canvas_bounds: Sequence[float] | None = None,
    existing_nodes: Sequence[dict] | None = None,
) -> dict:
    orientation = float(decoded.get("frame", {}).get("orientation", 0.0))
    if action_token == "ACTION_SUPPORT" and canvas_bounds is not None:
        return _with_bbox(decoded, _clamp_bbox_to_bounds(decoded["bbox"], canvas_bounds), orientation=orientation)
    if anchor_bbox is None:
        return decoded
    if action_token in {"ACTION_INSERT_GROUP", "ACTION_INSERT"}:
        anchor = _bbox_metrics(anchor_bbox)
        padding = 0.02 * min(anchor["width"], anchor["height"])
        return _with_bbox(decoded, _clamp_bbox_inside(decoded["bbox"], anchor_bbox, padding=padding), orientation=orientation)
    if action_token == "ACTION_ADJACENT_SUPPORT":
        peer_bboxes = [
            node["coarse_bbox"]
            for node in existing_nodes or []
            if str(node.get("role", "")) in {"support_region", "residual_region"} and node.get("coarse_bbox") is not None
        ]
        bbox = _adjacent_bbox_from_anchor(
            decoded["bbox"],
            anchor_bbox,
            side_bucket=int(relation_extra.get("side_bucket", 0)),
            gap_bucket=int(relation_extra.get("gap_bucket", 0)),
            config=config,
            existing_bboxes=peer_bboxes,
            canvas_bounds=canvas_bounds,
        )
        return _with_bbox(decoded, bbox, orientation=orientation)
    if action_token == "ACTION_DIVIDER":
        bbox = _divider_bbox_from_anchor(
            decoded["bbox"],
            anchor_bbox,
            axis_bucket=int(relation_extra.get("axis_bucket", 0)),
            thickness_bucket=int(relation_extra.get("thickness_bucket", 0)),
            config=config,
        )
        return _with_bbox(decoded, bbox, orientation=orientation)
    return decoded


def _layout_relation_pairs(relations: Sequence[dict], relation_type: str) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for relation in relations:
        if str(relation.get("type", "")) != relation_type:
            continue
        if relation_type == "adjacent_to":
            faces = [str(value) for value in relation.get("faces", []) or []]
            output.extend((left, right) for left, right in zip(faces, faces[1:]))
        elif relation_type == "inserted_in":
            obj = relation.get("object")
            container = relation.get("container", relation.get("support"))
            if obj is not None and container is not None:
                output.append((str(obj), str(container)))
        elif relation_type == "contains":
            parent = relation.get("parent")
            child = relation.get("child")
            if parent is not None and child is not None:
                output.append((str(child), str(parent)))
    return output


def _layout_children_and_parents(relations: Sequence[dict]) -> tuple[dict[str, list[str]], dict[str, str]]:
    children: dict[str, list[str]] = defaultdict(list)
    parent_by_child: dict[str, str] = {}

    def add(parent: object, child: object) -> None:
        if parent is None or child is None:
            return
        parent_id = str(parent)
        child_id = str(child)
        if not parent_id or not child_id or parent_id == child_id:
            return
        children[parent_id].append(child_id)
        parent_by_child.setdefault(child_id, parent_id)

    for relation in relations:
        relation_type = str(relation.get("type", ""))
        if relation_type == "contains":
            add(relation.get("parent"), relation.get("child"))
        elif relation_type == "inserted_in":
            add(relation.get("container", relation.get("support")), relation.get("object"))
        elif relation_type == "divides":
            add(relation.get("target", relation.get("support")), relation.get("divider"))
        elif relation_type == "adjacent_to":
            faces = [str(value) for value in relation.get("faces", []) or []]
            if len(faces) >= 2:
                add(faces[0], faces[1])
    return {key: list(dict.fromkeys(values)) for key, values in children.items()}, parent_by_child


def _layout_subtree_node_ids(root_id: str, children: dict[str, list[str]]) -> set[str]:
    output: set[str] = set()
    stack = [str(root_id)]
    while stack:
        node_id = stack.pop()
        if node_id in output:
            continue
        output.add(node_id)
        stack.extend(children.get(node_id, []))
    return output


def _layout_contains_children(relations: Sequence[dict]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        if str(relation.get("type", "")) == "contains" and relation.get("parent") is not None and relation.get("child") is not None:
            output[str(relation["parent"])].append(str(relation["child"]))
    return {key: list(dict.fromkeys(values)) for key, values in output.items()}


def _layout_semantic_bbox(
    node_id: str,
    *,
    node_by_id: dict[str, dict],
    bboxes: dict[str, list[float]],
    contains_children: dict[str, list[str]],
) -> list[float] | None:
    node = node_by_id.get(str(node_id), {})
    if str(node.get("role", "")) == "insert_object_group":
        child_boxes = [
            box
            for child_id in contains_children.get(str(node_id), [])
            if (box := _layout_semantic_bbox(child_id, node_by_id=node_by_id, bboxes=bboxes, contains_children=contains_children)) is not None
        ]
        union = _union_bboxes(child_boxes)
        if union is not None:
            return union
    return bboxes.get(str(node_id))


def _layout_sync_group_bboxes(
    nodes: Sequence[dict],
    bboxes: dict[str, list[float]],
    contains_children: dict[str, list[str]],
) -> dict[str, list[float]]:
    output = {key: [float(value) for value in bbox] for key, bbox in bboxes.items()}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            node_id = str(node.get("id", ""))
            if str(node.get("role", "")) != "insert_object_group":
                continue
            child_boxes = [output[child_id] for child_id in contains_children.get(node_id, []) if child_id in output]
            union = _union_bboxes(child_boxes)
            if union is not None and output.get(node_id) != union:
                output[node_id] = [float(value) for value in union]
                changed = True
    return output


def _layout_apply_transform_to_bboxes(
    bboxes: dict[str, list[float]],
    moved_ids: set[str],
    source_bbox: Sequence[float],
    target_bbox: Sequence[float],
) -> dict[str, list[float]]:
    output = {key: [float(value) for value in bbox] for key, bbox in bboxes.items()}
    for node_id in moved_ids:
        if node_id in output:
            output[node_id] = _transform_bbox_between(output[node_id], source_bbox, target_bbox)
    return output


def _layout_same_parent_support_pairs(nodes: Sequence[dict], parent_by_child: dict[str, str]) -> list[tuple[str, str]]:
    support_by_parent: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        node_id = str(node.get("id", ""))
        if str(node.get("role", "")) in {"support_region", "residual_region"}:
            support_by_parent[parent_by_child.get(node_id, "__root__")].append(node_id)
    pairs: list[tuple[str, str]] = []
    for values in support_by_parent.values():
        ordered = list(dict.fromkeys(values))
        for index, left_id in enumerate(ordered):
            for right_id in ordered[index + 1 :]:
                pairs.append((left_id, right_id))
    return pairs


def _layout_candidate_adjacent_bboxes(
    child_bbox: Sequence[float],
    anchor_bbox: Sequence[float],
    *,
    child_reference_bbox: Sequence[float] | None = None,
    anchor_reference_bbox: Sequence[float] | None = None,
    target_gap: float = 0.0,
) -> list[list[float]]:
    child = _bbox_metrics(child_bbox)
    anchor = _bbox_metrics(anchor_bbox)
    side = _adjacent_side(child_reference_bbox or child_bbox, anchor_reference_bbox or anchor_bbox)
    width = max(1.0, child["width"])
    height = max(1.0, child["height"])

    def cross_centers(axis: str) -> list[float]:
        if axis == "y":
            span = min(height, anchor["height"])
            low = anchor["min_y"] + span / 2.0
            high = anchor["max_y"] - span / 2.0
            decoded = child["center_y"]
        else:
            span = min(width, anchor["width"])
            low = anchor["min_x"] + span / 2.0
            high = anchor["max_x"] - span / 2.0
            decoded = child["center_x"]
        if high < low:
            low = high = (low + high) / 2.0
        values = [
            _clamp(decoded, low, high),
            (low + high) / 2.0,
            low,
            high,
            low + (high - low) * 0.25,
            low + (high - low) * 0.75,
        ]
        return list(dict.fromkeys(float(value) for value in values))

    def make_candidate(candidate_side: int, cross_center: float) -> list[float]:
        gap = max(0.0, float(target_gap))
        if candidate_side == 0:
            return _bbox_from_center_size(anchor["max_x"] + gap + width / 2.0, cross_center, width, height)
        if candidate_side == 1:
            return _bbox_from_center_size(anchor["min_x"] - gap - width / 2.0, cross_center, width, height)
        if candidate_side == 2:
            return _bbox_from_center_size(cross_center, anchor["max_y"] + gap + height / 2.0, width, height)
        return _bbox_from_center_size(cross_center, anchor["min_y"] - gap - height / 2.0, width, height)

    side_order = [side, *[candidate for candidate in (0, 1, 2, 3) if candidate != side]]
    candidates: list[list[float]] = []
    for candidate_side in side_order:
        axis = "y" if candidate_side in {0, 1} else "x"
        for center in cross_centers(axis):
            candidates.append(make_candidate(candidate_side, center))
    return [[float(value) for value in candidate] for candidate in candidates]


def _layout_constraint_score(
    nodes: Sequence[dict],
    relations: Sequence[dict],
    bboxes: dict[str, list[float]],
    *,
    max_adjacent_gap: float,
    max_adjacent_overlap_ratio: float,
    min_adjacent_overlap_span_ratio: float,
    min_containment_ratio: float,
    same_parent_max_overlap_ratio: float,
) -> tuple[int, float]:
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    contains_children = _layout_contains_children(relations)
    synced_bboxes = _layout_sync_group_bboxes(nodes, bboxes, contains_children)
    failure_count = 0
    penalty = 0.0

    for child_id, parent_id in [*_layout_relation_pairs(relations, "contains"), *_layout_relation_pairs(relations, "inserted_in")]:
        child_bbox = _layout_semantic_bbox(child_id, node_by_id=node_by_id, bboxes=synced_bboxes, contains_children=contains_children)
        parent_bbox = _layout_semantic_bbox(parent_id, node_by_id=node_by_id, bboxes=synced_bboxes, contains_children=contains_children)
        if child_bbox is None or parent_bbox is None:
            failure_count += 1
            penalty += 1000.0
            continue
        containment = _bbox_containment_ratio(child_bbox, parent_bbox)
        center_ok = _bbox_center_in_bbox(child_bbox, parent_bbox)
        miss = max(0.0, float(min_containment_ratio) - containment)
        if miss > 0.0 or not center_ok:
            failure_count += 1
            penalty += miss * 100.0 + (25.0 if not center_ok else 0.0)

    for left_id, right_id in _layout_relation_pairs(relations, "adjacent_to"):
        left_bbox = synced_bboxes.get(left_id)
        right_bbox = synced_bboxes.get(right_id)
        if left_bbox is None or right_bbox is None:
            failure_count += 1
            penalty += 1000.0
            continue
        gap = _bbox_gap(left_bbox, right_bbox)
        overlap = _bbox_overlap_ratio(left_bbox, right_bbox)
        span = _adjacent_overlap_span_ratio(left_bbox, right_bbox)
        gap_miss = max(0.0, gap - float(max_adjacent_gap))
        overlap_miss = max(0.0, overlap - float(max_adjacent_overlap_ratio))
        span_miss = max(0.0, float(min_adjacent_overlap_span_ratio) - span)
        if gap_miss > 0.0 or overlap_miss > 0.0 or span_miss > 0.0:
            failure_count += 1
            penalty += gap_miss + overlap_miss * 100.0 + span_miss * 20.0

    _children, parent_by_child = _layout_children_and_parents(relations)
    adjacent_pairs = {tuple(sorted(pair)) for pair in _layout_relation_pairs(relations, "adjacent_to")}
    for left_id, right_id in _layout_same_parent_support_pairs(nodes, parent_by_child):
        left_bbox = synced_bboxes.get(left_id)
        right_bbox = synced_bboxes.get(right_id)
        if left_bbox is None or right_bbox is None:
            continue
        allowed = max_adjacent_overlap_ratio if tuple(sorted((left_id, right_id))) in adjacent_pairs else same_parent_max_overlap_ratio
        overlap_miss = max(0.0, _bbox_overlap_ratio(left_bbox, right_bbox) - float(allowed))
        if overlap_miss > 0.0:
            failure_count += 1
            penalty += overlap_miss * 100.0

    for node_id, bbox in synced_bboxes.items():
        original = node_by_id.get(node_id, {}).get("coarse_layout", {}).get("decoded_bbox")
        if isinstance(original, list) and len(original) == 4:
            penalty += _bbox_gap(bbox, original) * 0.001

    return int(failure_count), float(penalty)


def _layout_candidate_maps(
    nodes: Sequence[dict],
    relations: Sequence[dict],
    bboxes: dict[str, list[float]],
    *,
    children: dict[str, list[str]],
    max_adjacent_gap: float,
    max_adjacent_overlap_ratio: float,
    min_adjacent_overlap_span_ratio: float,
    min_containment_ratio: float,
    same_parent_max_overlap_ratio: float,
) -> list[tuple[str, dict[str, list[float]]]]:
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    contains_children = _layout_contains_children(relations)
    synced_bboxes = _layout_sync_group_bboxes(nodes, bboxes, contains_children)
    candidates: list[tuple[str, dict[str, list[float]]]] = []

    for child_id, parent_id in [*_layout_relation_pairs(relations, "inserted_in"), *_layout_relation_pairs(relations, "contains")]:
        child_bbox = _layout_semantic_bbox(child_id, node_by_id=node_by_id, bboxes=synced_bboxes, contains_children=contains_children)
        parent_bbox = _layout_semantic_bbox(parent_id, node_by_id=node_by_id, bboxes=synced_bboxes, contains_children=contains_children)
        if child_bbox is None or parent_bbox is None:
            continue
        containment = _bbox_containment_ratio(child_bbox, parent_bbox)
        if containment >= float(min_containment_ratio) and _bbox_center_in_bbox(child_bbox, parent_bbox):
            continue
        target_bbox = _clamp_bbox_inside(child_bbox, parent_bbox, padding=0.0)
        moved_ids = _layout_subtree_node_ids(child_id, children)
        candidates.append((f"contain:{child_id}->{parent_id}", _layout_apply_transform_to_bboxes(synced_bboxes, moved_ids, child_bbox, target_bbox)))

    for anchor_id, child_id in _layout_relation_pairs(relations, "adjacent_to"):
        anchor_bbox = synced_bboxes.get(anchor_id)
        child_bbox = synced_bboxes.get(child_id)
        if anchor_bbox is None or child_bbox is None:
            continue
        gap = _bbox_gap(anchor_bbox, child_bbox)
        overlap = _bbox_overlap_ratio(anchor_bbox, child_bbox)
        span = _adjacent_overlap_span_ratio(anchor_bbox, child_bbox)
        if gap <= float(max_adjacent_gap) and overlap <= float(max_adjacent_overlap_ratio) and span >= float(min_adjacent_overlap_span_ratio):
            continue
        moved_ids = _layout_subtree_node_ids(child_id, children)
        for target_bbox in _layout_candidate_adjacent_bboxes(child_bbox, anchor_bbox, target_gap=0.0):
            candidates.append((f"adjacent:{child_id}->{anchor_id}", _layout_apply_transform_to_bboxes(synced_bboxes, moved_ids, child_bbox, target_bbox)))

    _children, parent_by_child = _layout_children_and_parents(relations)
    adjacent_pairs = {tuple(sorted(pair)) for pair in _layout_relation_pairs(relations, "adjacent_to")}
    for left_id, right_id in _layout_same_parent_support_pairs(nodes, parent_by_child):
        left_bbox = synced_bboxes.get(left_id)
        right_bbox = synced_bboxes.get(right_id)
        if left_bbox is None or right_bbox is None:
            continue
        allowed = max_adjacent_overlap_ratio if tuple(sorted((left_id, right_id))) in adjacent_pairs else same_parent_max_overlap_ratio
        if _bbox_overlap_ratio(left_bbox, right_bbox) <= float(allowed):
            continue
        moved_ids = _layout_subtree_node_ids(right_id, children)
        for target_bbox in _layout_candidate_adjacent_bboxes(right_bbox, left_bbox, target_gap=max_adjacent_gap):
            candidates.append((f"separate:{right_id}->{left_id}", _layout_apply_transform_to_bboxes(synced_bboxes, moved_ids, right_bbox, target_bbox)))

    return candidates


def solve_coarse_scene_layout_constraints(
    target: dict,
    *,
    max_iterations: int = 64,
    max_adjacent_gap: float = 4.0,
    max_adjacent_overlap_ratio: float = 0.1,
    min_adjacent_overlap_span_ratio: float = 0.2,
    min_containment_ratio: float = 0.8,
    same_parent_max_overlap_ratio: float = 0.1,
    in_place: bool = False,
) -> dict:
    output = target if in_place else copy.deepcopy(target)
    graph = output.get("parse_graph", {}) or {}
    nodes = list(graph.get("nodes", []) or [])
    relations = list(graph.get("relations", []) or [])
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    bboxes = {
        node_id: [float(value) for value in node.get("coarse_bbox")]
        for node_id, node in node_by_id.items()
        if isinstance(node.get("coarse_bbox"), list) and len(node.get("coarse_bbox")) == 4
    }
    for node_id, node in node_by_id.items():
        if isinstance(node.get("coarse_layout"), dict) and node_id in bboxes:
            node["coarse_layout"].setdefault("decoded_bbox", copy.deepcopy(bboxes[node_id]))

    children, _parent_by_child = _layout_children_and_parents(relations)
    contains_children = _layout_contains_children(relations)
    bboxes = _layout_sync_group_bboxes(nodes, bboxes, contains_children)
    base_score = _layout_constraint_score(
        nodes,
        relations,
        bboxes,
        max_adjacent_gap=max_adjacent_gap,
        max_adjacent_overlap_ratio=max_adjacent_overlap_ratio,
        min_adjacent_overlap_span_ratio=min_adjacent_overlap_span_ratio,
        min_containment_ratio=min_containment_ratio,
        same_parent_max_overlap_ratio=same_parent_max_overlap_ratio,
    )
    initial_score = base_score
    rounds: list[dict] = []
    operation_histogram: Counter[str] = Counter()
    max_iterations = max(0, int(max_iterations))

    for iteration in range(max_iterations):
        candidates = _layout_candidate_maps(
            nodes,
            relations,
            bboxes,
            children=children,
            max_adjacent_gap=max_adjacent_gap,
            max_adjacent_overlap_ratio=max_adjacent_overlap_ratio,
            min_adjacent_overlap_span_ratio=min_adjacent_overlap_span_ratio,
            min_containment_ratio=min_containment_ratio,
            same_parent_max_overlap_ratio=same_parent_max_overlap_ratio,
        )
        if not candidates:
            rounds.append({"iteration": int(iteration), "candidate_count": 0, "accepted": False, "score": list(base_score)})
            break
        best_label: str | None = None
        best_bboxes: dict[str, list[float]] | None = None
        best_score = base_score
        for label, candidate_bboxes in candidates:
            candidate_score = _layout_constraint_score(
                nodes,
                relations,
                candidate_bboxes,
                max_adjacent_gap=max_adjacent_gap,
                max_adjacent_overlap_ratio=max_adjacent_overlap_ratio,
                min_adjacent_overlap_span_ratio=min_adjacent_overlap_span_ratio,
                min_containment_ratio=min_containment_ratio,
                same_parent_max_overlap_ratio=same_parent_max_overlap_ratio,
            )
            if candidate_score < best_score:
                best_score = candidate_score
                best_label = label
                best_bboxes = candidate_bboxes
        if best_bboxes is None or best_label is None:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "candidate_count": int(len(candidates)),
                    "accepted": False,
                    "score": list(base_score),
                }
            )
            break
        rounds.append(
            {
                "iteration": int(iteration),
                "candidate_count": int(len(candidates)),
                "accepted": True,
                "operation": best_label,
                "score_before": list(base_score),
                "score_after": list(best_score),
            }
        )
        operation_histogram[best_label.split(":", 1)[0]] += 1
        bboxes = best_bboxes
        base_score = best_score
        if best_score[0] == 0:
            break

    final_bboxes = _layout_sync_group_bboxes(nodes, bboxes, contains_children)
    changed_node_ids: list[str] = []
    for node_id, bbox in final_bboxes.items():
        node = node_by_id.get(node_id)
        if node is None:
            continue
        old_bbox = node.get("coarse_bbox")
        new_bbox = [float(value) for value in bbox]
        if not isinstance(old_bbox, list) or len(old_bbox) != 4 or any(abs(float(left) - float(right)) > 1e-6 for left, right in zip(old_bbox, new_bbox)):
            changed_node_ids.append(node_id)
        _set_node_coarse_bbox(node, new_bbox)

    summary = {
        "enabled": True,
        "solver": "coarse_layout_constraint_v1",
        "iteration_count": int(len(rounds)),
        "changed_node_count": int(len(changed_node_ids)),
        "changed_node_ids": changed_node_ids,
        "initial_failure_count": int(initial_score[0]),
        "final_failure_count": int(base_score[0]),
        "final_penalty": float(base_score[1]),
        "operation_histogram": dict(operation_histogram),
        "max_iterations": int(max_iterations),
        "max_adjacent_gap": float(max_adjacent_gap),
        "max_adjacent_overlap_ratio": float(max_adjacent_overlap_ratio),
        "min_adjacent_overlap_span_ratio": float(min_adjacent_overlap_span_ratio),
        "min_containment_ratio": float(min_containment_ratio),
        "same_parent_max_overlap_ratio": float(same_parent_max_overlap_ratio),
        "rounds": rounds,
    }
    output.setdefault("metadata", {})["coarse_scene_layout_constraints"] = summary
    return output


def _primary_relation_node_id(
    node_id: str,
    relation_node_ids: Sequence[str],
    *,
    bboxes: dict[str, list[float]],
) -> str | None:
    candidates = [str(value) for value in relation_node_ids if value in bboxes]
    if not candidates:
        return str(relation_node_ids[0]) if relation_node_ids else None
    bbox = bboxes.get(node_id)
    if bbox is None:
        return candidates[0]

    def score(candidate: str) -> tuple[float, float]:
        candidate_bbox = bboxes[candidate]
        inter = _intersection_bbox(bbox, candidate_bbox)
        inter_area = 0.0 if inter is None else _bbox_metrics(inter)["area"]
        bbox_area = max(EPS, _bbox_metrics(bbox)["area"])
        return (inter_area / bbox_area, -_bbox_gap(bbox, candidate_bbox))

    return max(candidates, key=score)


def _margin_ratio(child: Sequence[float], anchor: Sequence[float]) -> float:
    anchor_metrics = _bbox_metrics(anchor)
    margins = [
        float(child[0]) - float(anchor[0]),
        float(child[1]) - float(anchor[1]),
        float(anchor[2]) - float(child[2]),
        float(anchor[3]) - float(child[3]),
    ]
    return float(min(margins) / max(EPS, max(anchor_metrics["width"], anchor_metrics["height"])))


def _divider_axis(orientation: float, bbox: Sequence[float]) -> int:
    metrics = _bbox_metrics(bbox)
    if metrics["width"] > metrics["height"] * 2.0:
        return 0
    if metrics["height"] > metrics["width"] * 2.0:
        return 1
    angle = _wrap_angle(orientation)
    return 2 if math.sin(angle) * math.cos(angle) >= 0.0 else 3


def _adjacent_side(child: Sequence[float], anchor: Sequence[float]) -> int:
    child_metrics = _bbox_metrics(child)
    anchor_metrics = _bbox_metrics(anchor)
    dx = child_metrics["center_x"] - anchor_metrics["center_x"]
    dy = child_metrics["center_y"] - anchor_metrics["center_y"]
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0.0 else 1
    return 2 if dy >= 0.0 else 3


def build_coarse_scene_actions(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> tuple[list[dict], dict]:
    config = config or ParseGraphTokenizerConfig()
    nodes = list((topology_target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    maps = _relation_maps(topology_target)
    order, order_diagnostics = parent_first_node_order(topology_target)
    bboxes = _node_bboxes(topology_target, geometry_targets)
    geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
    id_to_action_index: dict[str, int] = {}
    actions: list[dict] = []
    action_histogram: Counter[str] = Counter()
    anchor_histogram: Counter[str] = Counter()
    relation_histogram: Counter[str] = Counter()
    forward_reference_count = 0

    for original_index in order:
        node = nodes[int(original_index)]
        node_id = str(node.get("id", ""))
        role = str(node.get("role", ""))
        bbox = bboxes.get(node_id, _fallback_bbox_from_frame(node.get("frame", {})))
        frame = _frame_for_node(node, geometry_by_id, bbox)
        orientation = float(frame.get("orientation", 0.0))
        action_token = "ACTION_SUPPORT"
        relation_token: str | None = None
        anchor_node_id: str | None = None
        relation_node_ids: list[str] = []
        if role == "insert_object_group":
            action_token = "ACTION_INSERT_GROUP"
            relation_token = "REL_INSERTED_IN"
            anchor_node_id = maps["inserted_container_by_object"].get(node_id)
        elif role == "insert_object":
            action_token = "ACTION_INSERT"
            relation_token = "REL_CONTAINS"
            anchor_node_id = maps["contains_parent_by_child"].get(node_id) or maps["inserted_container_by_object"].get(node_id)
        elif role == "divider_region":
            action_token = "ACTION_DIVIDER"
            relation_token = "REL_DIVIDES"
            candidates = [value for value in maps["divides_targets_by_divider"].get(node_id, []) if value in id_to_action_index]
            primary = _primary_relation_node_id(node_id, candidates, bboxes=bboxes)
            relation_node_ids = [primary] if primary is not None else []
            anchor_node_id = primary
        elif role in {"support_region", "residual_region"}:
            candidates = [candidate for candidate in maps["adjacent_by_node"].get(node_id, []) if candidate in id_to_action_index]
            primary = _primary_relation_node_id(node_id, candidates, bboxes=bboxes)
            relation_node_ids = [primary] if primary is not None else []
            if primary is not None:
                action_token = "ACTION_ADJACENT_SUPPORT"
                relation_token = "REL_ADJACENT_TO"
                anchor_node_id = primary

        anchor_index = None
        anchor_bbox = None
        anchor_mode = "global"
        if anchor_node_id is not None and anchor_node_id in id_to_action_index:
            anchor_index = int(id_to_action_index[anchor_node_id])
            anchor_bbox = actions[anchor_index]["bbox"]
            anchor_mode = "node"
        elif anchor_node_id is not None:
            forward_reference_count += 1
            action_token = "ACTION_SUPPORT" if role in {"support_region", "residual_region"} else action_token
            relation_token = None if role in {"support_region", "residual_region"} else relation_token
            anchor_node_id = None

        action = {
            "action_token": action_token,
            "source_node_index": int(original_index),
            "source_node_id": node_id,
            "role": role,
            "label": int(node.get("label", 0)),
            "geometry_model": str(node.get("geometry_model", "none")),
            "bbox": [float(value) for value in bbox],
            "frame": copy.deepcopy(frame),
            "orientation": float(orientation),
            "anchor_mode": anchor_mode,
            "anchor_index": anchor_index,
            "anchor_node_id": anchor_node_id,
            "relation_node_ids": list(relation_node_ids or ([anchor_node_id] if anchor_node_id is not None else [])),
            "relation_anchor_indices": [
                int(id_to_action_index[value])
                for value in relation_node_ids
                if value in id_to_action_index
            ],
            "anchor_bbox": copy.deepcopy(anchor_bbox),
            "relation_token": relation_token,
        }
        actions.append(action)
        id_to_action_index[node_id] = len(actions) - 1
        action_histogram[action_token] += 1
        anchor_histogram[anchor_mode] += 1
        if relation_token is not None:
            relation_histogram[relation_token] += 1

    diagnostics = {
        **order_diagnostics,
        "action_count": int(len(actions)),
        "action_histogram": dict(action_histogram),
        "anchor_mode_histogram": dict(anchor_histogram),
        "relation_histogram": dict(relation_histogram),
        "forward_reference_count": int(forward_reference_count),
    }
    diagnostics["dependency_fallback_count"] = int(diagnostics.get("dependency_fallback_count", 0) + forward_reference_count)
    return actions, diagnostics


def _relation_extra_tokens(action: dict, *, config: ParseGraphTokenizerConfig) -> list[str]:
    if action["anchor_bbox"] is None:
        return []
    bbox = action["bbox"]
    anchor_bbox = action["anchor_bbox"]
    anchor_metrics = _bbox_metrics(anchor_bbox)
    if action["action_token"] in {"ACTION_INSERT_GROUP", "ACTION_INSERT"}:
        return [q_token(_margin_ratio(bbox, anchor_bbox), low=-1.0, high=1.0, bins=config.coarse_relation_bins)]
    if action["action_token"] == "ACTION_DIVIDER":
        inter = _intersection_bbox(bbox, anchor_bbox)
        inter_area = 0.0 if inter is None else _bbox_metrics(inter)["area"]
        coverage = inter_area / max(EPS, _bbox_metrics(bbox)["area"])
        thickness = min(_bbox_metrics(bbox)["width"], _bbox_metrics(bbox)["height"]) / max(
            EPS, max(anchor_metrics["width"], anchor_metrics["height"])
        )
        return [
            q_token(_divider_axis(float(action.get("orientation", 0.0)), bbox), low=0.0, high=3.0, bins=4),
            q_token(coverage, low=0.0, high=1.0, bins=config.coarse_relation_bins),
            q_token(thickness, low=0.0, high=1.0, bins=config.coarse_relation_bins),
        ]
    if action["action_token"] == "ACTION_ADJACENT_SUPPORT":
        child_metrics = _bbox_metrics(bbox)
        inter = _intersection_bbox(bbox, anchor_bbox)
        overlap = 0.0
        if inter is not None:
            im = _bbox_metrics(inter)
            overlap = min(im["width"] / max(EPS, min(child_metrics["width"], anchor_metrics["width"])), im["height"] / max(EPS, min(child_metrics["height"], anchor_metrics["height"])))
        gap = _bbox_gap(bbox, anchor_bbox) / max(EPS, max(anchor_metrics["width"], anchor_metrics["height"]))
        return [
            q_token(_adjacent_side(bbox, anchor_bbox), low=0.0, high=3.0, bins=4),
            q_token(gap, low=0.0, high=1.0, bins=config.coarse_relation_bins),
            q_token(overlap, low=0.0, high=1.0, bins=config.coarse_relation_bins),
        ]
    return []


def encode_coarse_scene_target(
    topology_target: dict,
    geometry_targets: Sequence[dict],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> list[str]:
    config = config or ParseGraphTokenizerConfig()
    canvas_width, canvas_height = _canvas_size(topology_target)
    actions, _diagnostics = build_coarse_scene_actions(topology_target, geometry_targets, config=config)
    tokens: list[str] = [
        "<BOS>",
        COARSE_SCENE_START_TOKEN,
        "SIZE",
        int_token(int(round(canvas_width)), config=config),
        int_token(int(round(canvas_height)), config=config),
        "COUNT",
        int_token(len(actions), config=config),
    ]
    for action in actions:
        action_token = str(action["action_token"])
        tokens.extend(
            [
                action_token,
                _manual_role_token(str(action["role"])),
                "LABEL",
                int_token(int(action["label"]), config=config),
                _manual_geometry_token(str(action["geometry_model"])),
            ]
        )
        if action["anchor_mode"] == "node":
            relation_token = str(action["relation_token"])
            tokens.extend([relation_token, int_token(int(action["anchor_index"]), config=config)])
            tokens.extend(["ANCHOR_NODE", int_token(int(action["anchor_index"]), config=config)])
            values = _rel_bbox_values(action["bbox"], action["anchor_bbox"], orientation=float(action.get("orientation", 0.0)))
            _append_rel_frame_tokens(tokens, values, config=config)
            tokens.extend(_relation_extra_tokens(action, config=config))
        else:
            tokens.append("ANCHOR_GLOBAL")
            values = _abs_bbox_values(
                action["bbox"],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                orientation=float(action.get("orientation", 0.0)),
            )
            _append_abs_frame_tokens(tokens, values, config=config)
        tokens.append("END_ACTION")
    tokens.append("<EOS>")
    return tokens


def coarse_scene_start_index(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens):
        if str(token) == COARSE_SCENE_START_TOKEN:
            return int(index)
    raise ValueError(f"{COARSE_SCENE_START_TOKEN} not found")


def _decode_abs_frame(reader: TokenReader, *, size: Sequence[float], config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME_ABS_COARSE")
    width, height = float(size[0]), float(size[1])
    cx = dequantize(reader.next_q(), low=0.0, high=1.0, bins=config.coarse_grid_bins) * width
    cy = dequantize(reader.next_q(), low=0.0, high=1.0, bins=config.coarse_grid_bins) * height
    bbox_w = max(1.0, dequantize(reader.next_q(), low=0.0, high=1.5, bins=config.coarse_size_bins) * width)
    bbox_h = max(1.0, dequantize(reader.next_q(), low=0.0, high=1.5, bins=config.coarse_size_bins) * height)
    log_aspect = dequantize(reader.next_q(), low=-2.0, high=2.0, bins=config.coarse_aspect_bins)
    orientation = dequantize(reader.next_q(), low=-math.pi, high=math.pi, bins=config.coarse_angle_bins)
    bbox = _bbox_from_center_size(cx, cy, bbox_w, bbox_h)
    return {
        "bbox": bbox,
        "coarse": {"log_aspect": float(log_aspect)},
        "frame": {"origin": [float(cx), float(cy)], "scale": float(max(bbox_w, bbox_h)), "orientation": float(orientation)},
    }


def _decode_rel_frame(
    reader: TokenReader,
    *,
    anchor_bbox: Sequence[float],
    config: ParseGraphTokenizerConfig,
) -> dict:
    reader.expect("FRAME_REL_COARSE")
    anchor = _bbox_metrics(anchor_bbox)
    rx = dequantize(reader.next_q(), low=-1.0, high=2.0, bins=config.coarse_grid_bins)
    ry = dequantize(reader.next_q(), low=-1.0, high=2.0, bins=config.coarse_grid_bins)
    rw = dequantize(reader.next_q(), low=0.0, high=2.0, bins=config.coarse_size_bins)
    rh = dequantize(reader.next_q(), low=0.0, high=2.0, bins=config.coarse_size_bins)
    log_aspect = dequantize(reader.next_q(), low=-2.0, high=2.0, bins=config.coarse_aspect_bins)
    orientation = dequantize(reader.next_q(), low=-math.pi, high=math.pi, bins=config.coarse_angle_bins)
    cx = anchor["min_x"] + rx * anchor["width"]
    cy = anchor["min_y"] + ry * anchor["height"]
    bbox_w = max(1.0, rw * anchor["width"])
    bbox_h = max(1.0, rh * anchor["height"])
    bbox = _bbox_from_center_size(cx, cy, bbox_w, bbox_h)
    return {
        "bbox": bbox,
        "coarse": {
            "center_x_rel": float(rx),
            "center_y_rel": float(ry),
            "width_rel": float(rw),
            "height_rel": float(rh),
            "log_aspect": float(log_aspect),
        },
        "frame": {"origin": [float(cx), float(cy)], "scale": float(max(bbox_w, bbox_h)), "orientation": float(orientation)},
    }


def _next_role(reader: TokenReader) -> str:
    token = reader.next()
    if token not in ROLE_TOKEN_TO_ROLE:
        raise ValueError(f"Expected role token, got {token}")
    return ROLE_TOKEN_TO_ROLE[token]


def _next_geometry_model(reader: TokenReader) -> str:
    token = reader.next()
    if token not in GEOMETRY_TOKEN_TO_MODEL:
        raise ValueError(f"Expected geometry token, got {token}")
    return GEOMETRY_TOKEN_TO_MODEL[token]


def _new_node_id(role: str, counters: Counter[str]) -> str:
    prefix = ROLE_ID_PREFIX.get(str(role), "node")
    index = counters[prefix]
    counters[prefix] += 1
    return f"{prefix}_{index}"


def _skip_relation_extra(reader: TokenReader, action_token: str) -> dict:
    if action_token in {"ACTION_INSERT_GROUP", "ACTION_INSERT"}:
        return {"margin_bucket": int(reader.next_q())}
    if action_token == "ACTION_DIVIDER":
        return {"axis_bucket": int(reader.next_q()), "coverage_bucket": int(reader.next_q()), "thickness_bucket": int(reader.next_q())}
    if action_token == "ACTION_ADJACENT_SUPPORT":
        return {"side_bucket": int(reader.next_q()), "gap_bucket": int(reader.next_q()), "overlap_bucket": int(reader.next_q())}
    return {}


def decode_coarse_scene_tokens_to_target(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
    solve_layout_constraints: bool = True,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    tokens = [str(token) for token in tokens]
    reader = TokenReader(tokens)
    reader.expect("<BOS>")
    reader.expect(COARSE_SCENE_START_TOKEN)
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]
    reader.expect("COUNT")
    action_count = reader.next_int()
    nodes: list[dict] = []
    relations: list[dict] = []
    counters: Counter[str] = Counter()
    group_children: dict[int, list[str]] = defaultdict(list)
    diagnostics = {"action_count": int(action_count), "actions": []}
    for action_index in range(int(action_count)):
        action_token = reader.next()
        if action_token not in ACTION_TO_ROLE:
            raise ValueError(f"Expected action token at action {action_index}, got {action_token}")
        role = _next_role(reader)
        if role not in ACTION_TO_ROLE[action_token]:
            raise ValueError(f"{action_token} cannot emit role {role}")
        reader.expect("LABEL")
        label = reader.next_int()
        geometry_model = _next_geometry_model(reader)
        relation_token = None
        anchor_index = None
        anchor_node_id = None
        relation_anchor_indices: list[int] = []
        relation_anchor_node_ids: list[str] = []
        anchor_bbox = None
        if action_token == "ACTION_SUPPORT":
            reader.expect("ANCHOR_GLOBAL")
            decoded = _decode_abs_frame(reader, size=size, config=config)
            decoded = _repair_decoded_frame(
                action_token,
                decoded,
                anchor_bbox=None,
                relation_extra={},
                config=config,
                canvas_bounds=[0.0, 0.0, float(size[0]), float(size[1])],
            )
        else:
            expected_relation = {
                "ACTION_INSERT_GROUP": "REL_INSERTED_IN",
                "ACTION_INSERT": "REL_CONTAINS",
                "ACTION_DIVIDER": "REL_DIVIDES",
                "ACTION_ADJACENT_SUPPORT": "REL_ADJACENT_TO",
            }[action_token]
            relation_token = reader.next()
            if relation_token != expected_relation:
                raise ValueError(f"Expected {expected_relation} for {action_token}, got {relation_token}")
            anchor_index = reader.next_int()
            relation_anchor_indices = [int(anchor_index)]
            for relation_anchor_index in relation_anchor_indices:
                if relation_anchor_index < 0 or relation_anchor_index >= len(nodes):
                    raise ValueError(f"Relation anchor index {relation_anchor_index} is not an already generated node")
            if anchor_index < 0 or anchor_index >= len(nodes):
                raise ValueError(f"Anchor index {anchor_index} is not an already generated node")
            reader.expect("ANCHOR_NODE")
            repeated_anchor = reader.next_int()
            if repeated_anchor != anchor_index:
                raise ValueError(f"Relation anchor {anchor_index} does not match anchor token {repeated_anchor}")
            anchor_node = nodes[int(anchor_index)]
            anchor_node_id = str(anchor_node["id"])
            relation_anchor_node_ids = [str(nodes[int(value)]["id"]) for value in relation_anchor_indices]
            anchor_bbox = anchor_node.get("coarse_bbox")
            if anchor_bbox is None:
                raise ValueError(f"Anchor node {anchor_index} is missing coarse bbox")
            if action_token == "ACTION_INSERT_GROUP" and str(anchor_node.get("role")) not in {"support_region", "insert_object_group"}:
                raise ValueError("Insert group anchor must be a support or insert group")
            if action_token == "ACTION_INSERT" and str(anchor_node.get("role")) != "insert_object_group":
                raise ValueError("Insert object anchor must be an insert group")
            if action_token == "ACTION_DIVIDER":
                for value in relation_anchor_indices:
                    if str(nodes[int(value)].get("role")) not in {"support_region", "insert_object_group", "residual_region"}:
                        raise ValueError("Divider anchor must be a support, residual, or insert group")
            if action_token == "ACTION_ADJACENT_SUPPORT":
                for value in relation_anchor_indices:
                    if str(nodes[int(value)].get("role")) not in {"support_region", "residual_region"}:
                        raise ValueError("Adjacent anchor must be a support or residual")
            decoded = _decode_rel_frame(reader, anchor_bbox=anchor_bbox, config=config)
        relation_extra = _skip_relation_extra(reader, action_token)
        if action_token != "ACTION_SUPPORT":
            decoded = _repair_decoded_frame(
                action_token,
                decoded,
                anchor_bbox=anchor_bbox,
                relation_extra=relation_extra,
                config=config,
                canvas_bounds=[0.0, 0.0, float(size[0]), float(size[1])],
                existing_nodes=nodes,
            )
        reader.expect("END_ACTION")
        node_id = _new_node_id(role, counters)
        node = {
            "id": node_id,
            "role": role,
            "label": int(label),
            "renderable": bool(role != "insert_object_group"),
            "is_reference_only": False,
            "geometry_model": geometry_model,
            "frame": copy.deepcopy(decoded["frame"]),
            "coarse_bbox": [float(value) for value in decoded["bbox"]],
            "coarse_layout": {
                "action_token": action_token,
                "anchor_index": anchor_index,
                "anchor_node_id": anchor_node_id,
                "relation_anchor_indices": list(relation_anchor_indices),
                "relation_anchor_node_ids": list(relation_anchor_node_ids),
                "relation_token": relation_token,
                "decoded_bbox": [float(value) for value in decoded["bbox"]],
                **decoded.get("coarse", {}),
                **relation_extra,
            },
        }
        if role != "insert_object_group" and geometry_model != "none":
            node["geometry_ref"] = node_id
        if role == "insert_object_group":
            node["children"] = []
        nodes.append(node)
        if action_token == "ACTION_INSERT_GROUP":
            relations.append({"type": "inserted_in", "object": node_id, "container": anchor_node_id})
        elif action_token == "ACTION_INSERT":
            relations.append({"type": "contains", "parent": anchor_node_id, "child": node_id})
            if anchor_index is not None:
                group_children[int(anchor_index)].append(node_id)
        elif action_token == "ACTION_DIVIDER":
            for relation_anchor_node_id in relation_anchor_node_ids:
                relations.append({"type": "divides", "divider": node_id, "target": relation_anchor_node_id})
        elif action_token == "ACTION_ADJACENT_SUPPORT":
            for relation_anchor_node_id in relation_anchor_node_ids:
                relations.append({"type": "adjacent_to", "faces": [relation_anchor_node_id, node_id]})
        diagnostics["actions"].append(
            {
                "action_index": int(action_index),
                "action_token": action_token,
                "node_id": node_id,
                "role": role,
                "anchor_index": anchor_index,
                "anchor_node_id": anchor_node_id,
                "relation_anchor_indices": list(relation_anchor_indices),
                "relation_anchor_node_ids": list(relation_anchor_node_ids),
            }
        )
    reader.expect("<EOS>")
    if reader.index != len(tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(tokens) - reader.index}")
    for index, children in group_children.items():
        nodes[index]["children"] = list(dict.fromkeys([*nodes[index].get("children", []), *children]))
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": size,
        "parse_graph": {"nodes": nodes, "relations": relations, "residuals": []},
        "metadata": {
            "coarse_scene_v1": True,
            "coarse_scene_diagnostics": diagnostics,
        },
    }
    if bool(solve_layout_constraints):
        target = solve_coarse_scene_layout_constraints(target, in_place=True)
    return target


def coarse_scene_target_to_parse_graph(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
    solve_layout_constraints: bool = True,
) -> dict:
    return decode_coarse_scene_tokens_to_target(tokens, config=config, solve_layout_constraints=solve_layout_constraints)


def validate_coarse_scene_tokens(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    try:
        target = decode_coarse_scene_tokens_to_target(tokens, config=config)
        return {
            "valid": True,
            "semantic_valid": True,
            "errors": [],
            "semantic_errors": [],
            "target": target,
        }
    except Exception as exc:  # noqa: BLE001 - validation reports parser/semantic failures uniformly.
        return {
            "valid": False,
            "semantic_valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "semantic_errors": [f"{type(exc).__name__}: {exc}"],
            "target": None,
        }


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(math.ceil(float(percentile) * len(ordered))) - 1
    return int(ordered[max(0, min(index, len(ordered) - 1))])


def _numeric_stats(values: Sequence[int | float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "p90": None, "max": None}
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "p90": _percentile([int(value) for value in floats], 0.90),
        "max": float(max(floats)),
    }


def build_coarse_scene_sequence_rows(
    split_root: Path,
    *,
    config: ParseGraphTokenizerConfig,
    vocab: Dict[str, int],
    max_tokens: int | None = None,
    max_samples: int | None = None,
    include_token_ids: bool = True,
) -> tuple[list[dict], dict]:
    split_root = Path(split_root)
    manifest_path = split_root / "manifest.jsonl"
    manifest_rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        manifest_rows = manifest_rows[: int(max_samples)]
    sequence_rows: list[dict] = []
    skipped_too_long = 0
    lengths: list[int] = []
    action_histogram: Counter[str] = Counter()
    anchor_mode_histogram: Counter[str] = Counter()
    relation_histogram: Counter[str] = Counter()
    dependency_fallback_count = 0
    forward_reference_count = 0
    coarse_clipping_count = 0

    for row in manifest_rows:
        topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        tokens = encode_coarse_scene_target(topology_target, geometry_targets, config=config)
        actions, diagnostics = build_coarse_scene_actions(topology_target, geometry_targets, config=config)
        if max_tokens is not None and len(tokens) > int(max_tokens):
            skipped_too_long += 1
            continue
        for action in actions:
            values = _abs_bbox_values(
                action["bbox"],
                canvas_width=_canvas_size(topology_target)[0],
                canvas_height=_canvas_size(topology_target)[1],
                orientation=float(action.get("orientation", 0.0)),
            )
            if any(float(values[key]) < 0.0 or float(values[key]) > 1.0 for key in ("center_x_norm", "center_y_norm")):
                coarse_clipping_count += 1
        sequence_row = {
            "format": "maskgen_tokenized_parse_graph_v1",
            "tokenizer": "manual_coarse_scene_v1",
            "source_topology": str(topology_path.as_posix()),
            "source_target": str(row.get("source_target", topology_path.as_posix())),
            "stem": row.get("stem"),
            "length": int(len(tokens)),
            "action_count": int(len(actions)),
            "loss_start_index": int(coarse_scene_start_index(tokens)),
            "action_histogram": diagnostics.get("action_histogram", {}),
            "anchor_mode_histogram": diagnostics.get("anchor_mode_histogram", {}),
            "relation_histogram": diagnostics.get("relation_histogram", {}),
            "dependency_fallback_count": int(diagnostics.get("dependency_fallback_count", 0)),
            "forward_reference_count": int(diagnostics.get("forward_reference_count", 0)),
            "tokens": tokens,
        }
        if include_token_ids:
            sequence_row["ids"] = tokens_to_ids(tokens, vocab)
        sequence_rows.append(sequence_row)
        lengths.append(len(tokens))
        action_histogram.update(diagnostics.get("action_histogram", {}) or {})
        anchor_mode_histogram.update(diagnostics.get("anchor_mode_histogram", {}) or {})
        relation_histogram.update(diagnostics.get("relation_histogram", {}) or {})
        dependency_fallback_count += int(diagnostics.get("dependency_fallback_count", 0))
        forward_reference_count += int(diagnostics.get("forward_reference_count", 0))

    summary = {
        "format": "maskgen_manual_coarse_scene_tokenized_summary_v1",
        "split_root": str(split_root.as_posix()),
        "sample_count": int(len(manifest_rows)),
        "written_coarse_scene": int(len(sequence_rows)),
        "skipped_too_long": int(skipped_too_long),
        "length_stats": _numeric_stats(lengths),
        "action_histogram": dict(action_histogram),
        "anchor_mode_histogram": dict(anchor_mode_histogram),
        "relation_histogram": dict(relation_histogram),
        "dependency_fallback_count": int(dependency_fallback_count),
        "forward_reference_count": int(forward_reference_count),
        "coarse_clipping_count": int(coarse_clipping_count),
    }
    return sequence_rows, summary


class CoarseSceneGrammarState:
    def __init__(self, config: CoarseSceneSamplerConfig | None = None) -> None:
        self.config = config or CoarseSceneSamplerConfig()
        self.tokenizer_config = self.config.tokenizer_config
        self.phase = "manual"
        self.size_index = 0
        self.action_count: int | None = None
        self.action_index = 0
        self.current_action: str | None = None
        self.current_role: str | None = None
        self.current_anchor_index: int | None = None
        self.current_relation_count: int = 0
        self.current_relation_indices: list[int] = []
        self.frame_value_index = 0
        self.extra_value_index = 0
        self.generated_roles: list[str] = []
        self.done = False
        self.errors: list[str] = []

    def _int_tokens(self, low: int, high: int) -> list[str]:
        return [int_token(value, config=self.tokenizer_config) for value in range(int(low), int(high) + 1)]

    def _q_tokens(self, bins: int) -> list[str]:
        return [f"Q_{index}" for index in range(int(bins))]

    def _compatible_anchor_indices(self) -> list[int]:
        roles = self.generated_roles
        if self.current_action == "ACTION_INSERT_GROUP":
            allowed = {"support_region", "insert_object_group", "residual_region"}
        elif self.current_action == "ACTION_INSERT":
            allowed = {"insert_object_group"}
        elif self.current_action == "ACTION_DIVIDER":
            allowed = {"support_region", "insert_object_group", "residual_region"}
        elif self.current_action == "ACTION_ADJACENT_SUPPORT":
            allowed = {"support_region", "residual_region"}
        else:
            allowed = set()
        return [index for index, role in enumerate(roles) if role in allowed]

    def _allowed_actions(self) -> list[str]:
        if self.action_count is not None and self.action_index >= self.action_count:
            return ["<EOS>"]
        allowed = []
        if "ACTION_SUPPORT" in self.config.allowed_action_tokens:
            allowed.append("ACTION_SUPPORT")
        has_support_anchor = any(role in {"support_region", "residual_region"} for role in self.generated_roles)
        has_container_anchor = any(role in {"support_region", "insert_object_group", "residual_region"} for role in self.generated_roles)
        has_group_anchor = any(role == "insert_object_group" for role in self.generated_roles)
        if has_support_anchor and "ACTION_ADJACENT_SUPPORT" in self.config.allowed_action_tokens:
            allowed.append("ACTION_ADJACENT_SUPPORT")
        if has_container_anchor and "ACTION_INSERT_GROUP" in self.config.allowed_action_tokens:
            allowed.append("ACTION_INSERT_GROUP")
        if has_group_anchor and "ACTION_INSERT" in self.config.allowed_action_tokens:
            allowed.append("ACTION_INSERT")
        if has_container_anchor and "ACTION_DIVIDER" in self.config.allowed_action_tokens:
            allowed.append("ACTION_DIVIDER")
        return list(dict.fromkeys(allowed))

    def allowed_token_strings(self) -> list[str]:
        if self.done:
            return []
        cfg = self.tokenizer_config
        if self.phase == "manual":
            return [COARSE_SCENE_START_TOKEN]
        if self.phase == "size_token":
            return ["SIZE"]
        if self.phase in {"size_w", "size_h"}:
            return self._int_tokens(1, int(cfg.position_max))
        if self.phase == "count_token":
            return ["COUNT"]
        if self.phase == "action_count":
            return self._int_tokens(1, int(self.config.max_actions))
        if self.phase == "action":
            return self._allowed_actions()
        if self.phase == "role":
            if self.current_action in {"ACTION_SUPPORT", "ACTION_ADJACENT_SUPPORT"}:
                return ["ROLE_SUPPORT", "ROLE_RESIDUAL"]
            if self.current_action == "ACTION_INSERT_GROUP":
                return ["ROLE_INSERT_GROUP"]
            if self.current_action == "ACTION_INSERT":
                return ["ROLE_INSERT"]
            if self.current_action == "ACTION_DIVIDER":
                return ["ROLE_DIVIDER"]
        if self.phase == "label_token":
            return ["LABEL"]
        if self.phase == "label":
            return self._int_tokens(0, int(self.config.max_label))
        if self.phase == "geometry_model":
            return ["GEOM_NONE"] if self.current_action == "ACTION_INSERT_GROUP" else ["GEOM_POLYGON_CODE", "GEOM_CONVEX_ATOMS"]
        if self.phase == "relation":
            if self.current_action == "ACTION_INSERT_GROUP":
                return ["REL_INSERTED_IN"]
            if self.current_action == "ACTION_INSERT":
                return ["REL_CONTAINS"]
            if self.current_action == "ACTION_DIVIDER":
                return ["REL_DIVIDES"]
            if self.current_action == "ACTION_ADJACENT_SUPPORT":
                return ["REL_ADJACENT_TO"]
            return ["ANCHOR_GLOBAL"]
        if self.phase in {"relation_anchor", "anchor_index"}:
            if self.phase == "anchor_index" and self.current_relation_indices:
                return [int_token(self.current_relation_indices[0], config=cfg)]
            return [int_token(index, config=cfg) for index in self._compatible_anchor_indices()]
        if self.phase == "anchor_token":
            return ["ANCHOR_NODE"]
        if self.phase == "frame_token":
            return ["FRAME_ABS_COARSE"] if self.current_action == "ACTION_SUPPORT" else ["FRAME_REL_COARSE"]
        if self.phase == "frame_value":
            if self.frame_value_index in {0, 1}:
                return self._q_tokens(int(cfg.coarse_grid_bins))
            if self.frame_value_index in {2, 3}:
                return self._q_tokens(int(cfg.coarse_size_bins))
            if self.frame_value_index == 4:
                return self._q_tokens(int(cfg.coarse_aspect_bins))
            return self._q_tokens(int(cfg.coarse_angle_bins))
        if self.phase == "extra_value":
            if self.current_action in {"ACTION_INSERT_GROUP", "ACTION_INSERT"}:
                return self._q_tokens(int(cfg.coarse_relation_bins))
            if self.current_action in {"ACTION_DIVIDER", "ACTION_ADJACENT_SUPPORT"} and self.extra_value_index == 0:
                return self._q_tokens(4)
            return self._q_tokens(int(cfg.coarse_relation_bins))
        if self.phase == "end_action":
            return ["END_ACTION"]
        return []

    def _extra_count(self) -> int:
        if self.current_action in {"ACTION_INSERT_GROUP", "ACTION_INSERT"}:
            return 1
        if self.current_action in {"ACTION_DIVIDER", "ACTION_ADJACENT_SUPPORT"}:
            return 3
        return 0

    def step(self, token: str) -> bool:
        token = str(token)
        allowed = set(self.allowed_token_strings())
        if token not in allowed:
            self.errors.append(f"illegal_{token}_in_phase_{self.phase}")
            self.done = True
            return False
        if self.phase == "manual":
            self.phase = "size_token"
        elif self.phase == "size_token":
            self.phase = "size_w"
        elif self.phase == "size_w":
            self.phase = "size_h"
        elif self.phase == "size_h":
            self.phase = "count_token"
        elif self.phase == "count_token":
            self.phase = "action_count"
        elif self.phase == "action_count":
            self.action_count = token_int(token)
            self.phase = "action"
        elif self.phase == "action":
            if token == "<EOS>":
                self.done = True
                return True
            self.current_action = token
            self.current_role = None
            self.current_anchor_index = None
            self.current_relation_count = 0
            self.current_relation_indices = []
            self.frame_value_index = 0
            self.extra_value_index = 0
            self.phase = "role"
        elif self.phase == "role":
            self.current_role = ROLE_TOKEN_TO_ROLE[token]
            self.phase = "label_token"
        elif self.phase == "label_token":
            self.phase = "label"
        elif self.phase == "label":
            self.phase = "geometry_model"
        elif self.phase == "geometry_model":
            self.phase = "relation"
        elif self.phase == "relation":
            if token == "ANCHOR_GLOBAL":
                self.phase = "frame_token"
            else:
                self.phase = "relation_anchor"
        elif self.phase == "relation_anchor":
            self.current_anchor_index = token_int(token)
            self.current_relation_indices = [int(self.current_anchor_index)]
            self.phase = "anchor_token"
        elif self.phase == "anchor_token":
            self.phase = "anchor_index"
        elif self.phase == "anchor_index":
            if token_int(token) != self.current_anchor_index:
                self.errors.append("anchor_index_mismatch")
                self.done = True
                return False
            self.phase = "frame_token"
        elif self.phase == "frame_token":
            self.phase = "frame_value"
        elif self.phase == "frame_value":
            self.frame_value_index += 1
            if self.frame_value_index >= 6:
                self.phase = "extra_value" if self._extra_count() > 0 else "end_action"
        elif self.phase == "extra_value":
            self.extra_value_index += 1
            if self.extra_value_index >= self._extra_count():
                self.phase = "end_action"
        elif self.phase == "end_action":
            self.generated_roles.append(str(self.current_role or "unknown"))
            self.action_index += 1
            self.current_action = None
            self.current_role = None
            self.current_anchor_index = None
            self.current_relation_count = 0
            self.current_relation_indices = []
            self.phase = "action"
        return True

    def diagnostics(self) -> dict:
        return {
            "phase": self.phase,
            "done": bool(self.done),
            "errors": list(self.errors),
            "action_count": self.action_count,
            "action_index": int(self.action_index),
            "generated_roles": list(self.generated_roles),
        }


@torch.no_grad()
def sample_coarse_scene_constrained(
    model,
    vocab: Dict[str, int],
    *,
    prefix_tokens: Sequence[str] | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: int | None = None,
    sampler_config: CoarseSceneSamplerConfig | None = None,
    device: torch.device | str | None = None,
    use_cache: bool = True,
) -> dict:
    sampler_config = sampler_config or CoarseSceneSamplerConfig()
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    device = torch.device(device) if device is not None else next(model.parameters()).device
    prefix = list(prefix_tokens or ["<BOS>"])
    missing = [token for token in prefix if token not in vocab]
    if missing:
        raise ValueError(f"Coarse scene prefix contains tokens not in vocab: {missing}")
    state = CoarseSceneGrammarState(sampler_config)
    ids = [int(vocab[token]) for token in prefix]
    tokens = [str(token) for token in prefix]
    stopped_reason = "max_new_tokens"
    block_size = int(getattr(model.config, "block_size", max_new_tokens + len(ids)))
    use_kv_cache = bool(use_cache and getattr(model, "supports_kv_cache", False) and int(max_new_tokens) + len(ids) <= int(block_size))
    past_kv = None

    for _step in range(int(max_new_tokens)):
        allowed_tokens = state.allowed_token_strings()
        allowed_ids = [int(vocab[token]) for token in allowed_tokens if token in vocab]
        if not allowed_ids:
            state.errors.append(f"empty_allowed_set_phase_{state.phase}")
            stopped_reason = "empty_allowed_set"
            break
        if use_kv_cache:
            if past_kv is None:
                input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
                outputs = model(input_ids, use_cache=True)
            else:
                input_ids = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                outputs = model(input_ids, past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
        else:
            input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
            outputs = model(input_ids)
        logits = outputs["logits"][0, -1, :]
        next_id = _sample_from_logits(logits, allowed_ids=allowed_ids, temperature=temperature, top_k=top_k)
        next_token = inverse_vocab.get(int(next_id), "<UNK>")
        ids.append(int(next_id))
        tokens.append(next_token)
        state.step(next_token)
        if next_token == "<EOS>" or state.done:
            stopped_reason = "eos" if next_token == "<EOS>" else "done"
            break
    return {
        "ids": ids,
        "tokens": tokens,
        "length": int(len(ids)),
        "hit_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "stopped_reason": stopped_reason,
        "constraint_diagnostics": state.diagnostics(),
    }


def sample_model_coarse_scene_rows(
    model,
    vocab: Dict[str, int],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device,
    sampler_config: CoarseSceneSamplerConfig | None = None,
    progress_every: int = 0,
    progress_label: str = "coarse_scene_sample",
) -> list[dict]:
    rows: list[dict] = []
    was_training = bool(model.training)
    model.eval()
    try:
        for sample_index in range(int(num_samples)):
            sample = sample_coarse_scene_constrained(
                model,
                vocab,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=top_k,
                sampler_config=sampler_config,
                device=device,
            )
            validation = validate_coarse_scene_tokens(sample["tokens"], config=(sampler_config or CoarseSceneSamplerConfig()).tokenizer_config)
            rows.append(
                {
                    "format": "maskgen_manual_coarse_scene_ar_sample_v1",
                    "sample_index": int(sample_index),
                    "sampling_mode": "coarse_scene_constrained",
                    "length": int(sample["length"]),
                    "hit_eos": bool(sample["hit_eos"]),
                    "ids": [int(value) for value in sample["ids"]],
                    "tokens": list(sample["tokens"]),
                    "valid": bool(validation["valid"]),
                    "semantic_valid": bool(validation["semantic_valid"]),
                    "validation_errors": list(validation["errors"]),
                    "semantic_validation_errors": list(validation["semantic_errors"]),
                    "constraint_diagnostics": sample["constraint_diagnostics"],
                }
            )
            if int(progress_every) > 0 and (sample_index + 1) % int(progress_every) == 0:
                print(f"{progress_label} {sample_index + 1}/{num_samples}", flush=True)
    finally:
        if was_training:
            model.train()
    return rows


def evaluate_coarse_scene_sample_rows(rows: Sequence[dict], *, top_k_invalid: int = 20) -> dict:
    sample_count = int(len(rows))
    valid_count = 0
    semantic_valid_count = 0
    hit_eos_count = 0
    lengths: list[int] = []
    action_counts: list[int] = []
    action_histogram: Counter[str] = Counter()
    failure_histogram: Counter[str] = Counter()
    invalid_samples: list[dict] = []
    for row_index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens", []) or []]
        lengths.append(int(row.get("length", len(tokens))))
        hit_eos_count += int(bool(tokens and tokens[-1] == "<EOS>"))
        validation = validate_coarse_scene_tokens(tokens)
        if not bool(validation["valid"]):
            reason = str(validation["errors"][0]) if validation["errors"] else "unknown"
            failure_histogram[reason] += 1
            if len(invalid_samples) < int(top_k_invalid):
                invalid_samples.append({"sample_index": row.get("sample_index", row_index), "errors": validation["errors"]})
            continue
        valid_count += 1
        semantic_valid_count += int(bool(validation["semantic_valid"]))
        target = validation["target"] or {}
        actions = ((target.get("metadata", {}) or {}).get("coarse_scene_diagnostics", {}) or {}).get("actions", []) or []
        action_counts.append(len(actions))
        action_histogram.update(str(action.get("action_token", "unknown")) for action in actions)
    return {
        "format": "maskgen_manual_coarse_scene_sample_eval_v1",
        "sample_count": int(sample_count),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / sample_count) if sample_count else 0.0,
        "semantic_valid_count": int(semantic_valid_count),
        "semantic_valid_rate": float(semantic_valid_count / sample_count) if sample_count else 0.0,
        "hit_eos_count": int(hit_eos_count),
        "lengths": _numeric_stats(lengths),
        "action_counts": _numeric_stats(action_counts),
        "action_histogram": dict(action_histogram),
        "failure_reason_histogram": dict(failure_histogram.most_common()),
        "invalid_samples": invalid_samples,
    }


def write_coarse_scene_sample_targets(rows: Sequence[dict], output_root: Path, *, config: ParseGraphTokenizerConfig | None = None) -> dict:
    output_root = Path(output_root)
    graph_dir = output_root / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    valid_count = 0
    for fallback_index, row in enumerate(rows):
        sample_index = int(row.get("sample_index", fallback_index))
        validation = validate_coarse_scene_tokens([str(token) for token in row.get("tokens", []) or []], config=config)
        if not bool(validation["valid"]):
            continue
        valid_count += 1
        target = validation["target"]
        target.setdefault("metadata", {}).update(
            {
                "sample_index": int(sample_index),
                "checkpoint": row.get("checkpoint"),
                "coarse_scene_sample": True,
            }
        )
        output_path = graph_dir / f"sample_{sample_index:06d}.json"
        output_path.write_text(json.dumps(target, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        manifest_rows.append({"sample_index": int(sample_index), "output_path": str(output_path.as_posix())})
    write_jsonl(output_root / "manifest.jsonl", manifest_rows)
    summary = {
        "format": "maskgen_manual_coarse_scene_target_write_summary_v1",
        "input_count": int(len(rows)),
        "valid_count": int(valid_count),
        "output_count": int(len(manifest_rows)),
        "output_root": str(output_root.as_posix()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
