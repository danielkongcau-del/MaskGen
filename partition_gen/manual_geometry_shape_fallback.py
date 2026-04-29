from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path
from typing import Sequence

from partition_gen.manual_geometry_conditioning import iter_jsonl
from partition_gen.manual_layout_residual import (
    geometry_local_bbox,
    geometry_renderable_local_points,
    scaled_bbox_metrics,
)
from partition_gen.manual_layout_retrieval import load_split_row


def _geometry_targets_by_source_node_id(geometry_targets: Sequence[dict]) -> dict[str, dict]:
    return {
        str(target.get("source_node_id")): target
        for target in geometry_targets
        if target.get("source_node_id") is not None
    }


def _node_key(value: dict, *, level: str) -> tuple:
    role = str(value.get("role", ""))
    label = int(value.get("label", 0))
    geometry_model = str(value.get("geometry_model", "polygon_code"))
    if level == "exact":
        return role, label, geometry_model
    if level == "role_label":
        return role, label
    if level == "role":
        return (role,)
    return ()


def _frame_origin(frame: dict) -> tuple[float, float]:
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return float(origin[0]), float(origin[1])


def _local_bbox_world_bbox(local_bbox: dict, frame: dict) -> list[float]:
    import math

    width = abs(float(local_bbox.get("width", 1.0)))
    height = abs(float(local_bbox.get("height", 1.0)))
    min_x = float(local_bbox.get("min_x", -width / 2.0))
    min_y = float(local_bbox.get("min_y", -height / 2.0))
    max_x = float(local_bbox.get("max_x", width / 2.0))
    max_y = float(local_bbox.get("max_y", height / 2.0))
    origin_x, origin_y = _frame_origin(frame)
    scale = max(float(frame.get("scale", 1.0)), 1e-8)
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    points = []
    for local_x, local_y in ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)):
        x = local_x * scale
        y = local_y * scale
        points.append(
            (
                origin_x + x * cos_theta - y * sin_theta,
                origin_y + x * sin_theta + y * cos_theta,
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _local_points_world_bbox(points: Sequence[Sequence[float]], frame: dict) -> list[float] | None:
    import math

    if not points:
        return None
    origin_x, origin_y = _frame_origin(frame)
    scale = max(float(frame.get("scale", 1.0)), 1e-8)
    theta = float(frame.get("orientation", 0.0))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    world_points = []
    for point in points:
        local_x = float(point[0])
        local_y = float(point[1])
        x = local_x * scale
        y = local_y * scale
        world_points.append(
            (
                origin_x + x * cos_theta - y * sin_theta,
                origin_y + x * sin_theta + y * cos_theta,
            )
        )
    xs = [point[0] for point in world_points]
    ys = [point[1] for point in world_points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _bbox_metrics(bbox: Sequence[float]) -> dict:
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    return {"width": float(width), "height": float(height), "area": float(width * height)}


def _bbox_intersects_canvas(bbox: Sequence[float], canvas_size: Sequence[float]) -> bool:
    width = float(canvas_size[0]) if len(canvas_size) >= 1 else 256.0
    height = float(canvas_size[1]) if len(canvas_size) >= 2 else width
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    return max_x >= 0.0 and max_y >= 0.0 and min_x <= width and min_y <= height


def local_bbox_quality(
    local_bbox: dict | None,
    frame: dict,
    *,
    canvas_size: Sequence[float] | None = None,
    min_world_bbox_area: float = 1.0,
    min_local_bbox_side: float = 1e-6,
) -> dict:
    return _local_bbox_quality(
        local_bbox,
        frame,
        canvas_size=canvas_size,
        min_world_bbox_area=min_world_bbox_area,
        min_local_bbox_side=min_local_bbox_side,
    )


def geometry_target_quality(
    geometry_target: dict,
    frame: dict,
    *,
    canvas_size: Sequence[float] | None = None,
    min_world_bbox_area: float = 1.0,
    min_local_bbox_side: float = 1e-6,
) -> dict:
    local_bbox = geometry_local_bbox(geometry_target)
    points = geometry_renderable_local_points(geometry_target)
    return _local_bbox_quality(
        local_bbox,
        frame,
        canvas_size=canvas_size,
        min_world_bbox_area=min_world_bbox_area,
        min_local_bbox_side=min_local_bbox_side,
        world_bbox=_local_points_world_bbox(points, frame),
    )


def _local_bbox_quality(
    local_bbox: dict | None,
    frame: dict,
    *,
    canvas_size: Sequence[float] | None = None,
    min_world_bbox_area: float = 1.0,
    min_local_bbox_side: float = 1e-6,
    world_bbox: Sequence[float] | None = None,
) -> dict:
    local_bbox = local_bbox or {"width": 0.0, "height": 0.0}
    local_width = abs(float(local_bbox.get("width", 0.0)))
    local_height = abs(float(local_bbox.get("height", 0.0)))
    scale = float(frame.get("scale", 1.0))
    scaled = scaled_bbox_metrics(local_bbox, scale)
    has_points = bool(local_bbox.get("has_points", True))
    local_side_ok = local_width > float(min_local_bbox_side) and local_height > float(min_local_bbox_side)
    world_bbox = list(world_bbox) if world_bbox is not None else _local_bbox_world_bbox(local_bbox, frame)
    world_metrics = _bbox_metrics(world_bbox)
    world_area_ok = float(world_metrics["area"]) > float(min_world_bbox_area)
    intersects_canvas = True if canvas_size is None else _bbox_intersects_canvas(world_bbox, canvas_size)
    usable = bool(has_points and local_side_ok and world_area_ok and intersects_canvas)
    reasons: list[str] = []
    if not has_points:
        reasons.append("missing_renderable_local_bbox")
    if not local_side_ok:
        reasons.append("degenerate_local_bbox")
    if not world_area_ok:
        reasons.append("tiny_world_bbox")
    if not intersects_canvas:
        reasons.append("off_canvas_bbox")
    return {
        "usable": usable,
        "reasons": reasons,
        "local_bbox": copy.deepcopy(local_bbox),
        "local_width": float(local_width),
        "local_height": float(local_height),
        "local_area": float(local_width * local_height),
        "has_points": bool(has_points),
        "scaled_bbox_width": float(scaled["width"]),
        "scaled_bbox_height": float(scaled["height"]),
        "scaled_bbox_area": float(scaled["area"]),
        "world_width": float(world_metrics["width"]),
        "world_height": float(world_metrics["height"]),
        "world_area": float(world_metrics["area"]),
        "world_bbox": [float(value) for value in world_bbox],
        "bbox_intersects_canvas": bool(intersects_canvas),
        "canvas_size": None if canvas_size is None else [float(value) for value in canvas_size],
        "scale": float(scale),
        "min_world_bbox_area": float(min_world_bbox_area),
        "min_local_bbox_side": float(min_local_bbox_side),
    }


def build_geometry_shape_fallback_library(
    split_root: Path,
    *,
    max_samples: int | None = None,
    min_local_bbox_side: float = 1e-6,
) -> tuple[dict, dict]:
    split_root = Path(split_root)
    manifest_path = split_root / "manifest.jsonl"
    rows = list(iter_jsonl(manifest_path))
    if max_samples is not None:
        rows = rows[: int(max_samples)]

    by_exact = defaultdict(list)
    by_role_label = defaultdict(list)
    by_role = defaultdict(list)
    by_source: dict[tuple[str, str], dict] = {}
    global_shapes: list[dict] = []
    skipped_missing = 0
    skipped_degenerate = 0

    for row in rows:
        _topology_path, topology_target, geometry_targets = load_split_row(
            row,
            split_root=split_root,
            manifest_parent=manifest_path.parent,
        )
        geometry_by_id = _geometry_targets_by_source_node_id(geometry_targets)
        graph = topology_target.get("parse_graph", {}) or {}
        for node in graph.get("nodes", []) or []:
            geometry_ref = node.get("geometry_ref")
            if not geometry_ref:
                continue
            geometry_target = geometry_by_id.get(str(geometry_ref))
            if geometry_target is None:
                skipped_missing += 1
                continue
            local_bbox = geometry_local_bbox(geometry_target)
            if (
                not bool(local_bbox.get("has_points", True))
                or abs(float(local_bbox.get("width", 0.0))) <= float(min_local_bbox_side)
                or abs(float(local_bbox.get("height", 0.0))) <= float(min_local_bbox_side)
            ):
                skipped_degenerate += 1
                continue
            shape = {
                "role": str(node.get("role", geometry_target.get("role", ""))),
                "label": int(node.get("label", geometry_target.get("label", 0))),
                "geometry_model": str(
                    node.get("geometry_model", geometry_target.get("geometry_model", "polygon_code"))
                ),
                "source_node_id": str(geometry_target.get("source_node_id", geometry_ref)),
                "source_stem": row.get("stem"),
                "geometry": copy.deepcopy(geometry_target.get("geometry")),
                "atoms": copy.deepcopy(geometry_target.get("atoms")),
                "local_bbox": local_bbox,
                "local_area": float(
                    abs(float(local_bbox.get("width", 0.0))) * abs(float(local_bbox.get("height", 0.0)))
                ),
            }
            by_exact[_node_key(shape, level="exact")].append(shape)
            by_role_label[_node_key(shape, level="role_label")].append(shape)
            by_role[_node_key(shape, level="role")].append(shape)
            by_source[(str(row.get("stem")), str(geometry_ref))] = shape
            global_shapes.append(shape)

    def sort_shapes(values: list[dict]) -> list[dict]:
        return sorted(values, key=lambda item: float(item.get("local_area", 0.0)), reverse=True)

    library = {
        "exact": {key: sort_shapes(values) for key, values in by_exact.items()},
        "role_label": {key: sort_shapes(values) for key, values in by_role_label.items()},
        "role": {key: sort_shapes(values) for key, values in by_role.items()},
        "source": {key: copy.deepcopy(value) for key, value in by_source.items()},
        "global": sort_shapes(global_shapes),
    }
    summary = {
        "format": "maskgen_geometry_shape_fallback_library_summary_v1",
        "split_root": str(split_root.as_posix()),
        "input_count": int(len(rows)),
        "shape_count": int(len(global_shapes)),
        "skipped_missing_geometry_count": int(skipped_missing),
        "skipped_degenerate_geometry_count": int(skipped_degenerate),
    }
    return library, summary


def select_fallback_geometry_shape(node: dict, library: dict) -> tuple[dict | None, str]:
    for level in ("exact", "role_label", "role"):
        values = (library.get(level, {}) or {}).get(_node_key(node, level=level), [])
        if values:
            return copy.deepcopy(values[0]), f"fallback_true_shape_{level}"
    values = library.get("global", []) or []
    if values:
        return copy.deepcopy(values[0]), "fallback_true_shape_global"
    return None, "fallback_true_shape_missing"


def geometry_target_from_fallback_shape(shape: dict, *, source_node_id: str, frame: dict) -> dict:
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": str(source_node_id),
        "role": str(shape.get("role", "")),
        "label": int(shape.get("label", 0)),
        "geometry_model": str(shape.get("geometry_model", "polygon_code")),
        "frame": copy.deepcopy(frame),
    }
    if shape.get("geometry") is not None:
        target["geometry"] = copy.deepcopy(shape["geometry"])
    if shape.get("atoms") is not None:
        target["atoms"] = copy.deepcopy(shape["atoms"])
    return target
