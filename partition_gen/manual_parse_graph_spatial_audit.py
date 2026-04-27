from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Sequence

from partition_gen.manual_parse_graph_target_audit import load_json
from partition_gen.manual_parse_graph_visualization import polygon_world_rings


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    index = int(math.ceil(float(percentile) * len(sorted_values))) - 1
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _numeric_stats(values: Sequence[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    floats = [float(value) for value in values]
    return {
        "count": int(len(floats)),
        "mean": float(mean(floats)),
        "min": float(min(floats)),
        "median": float(median(floats)),
        "p90": _percentile(floats, 0.90),
        "p95": _percentile(floats, 0.95),
        "p99": _percentile(floats, 0.99),
        "max": float(max(floats)),
    }


def _sort_histogram(counter: Counter) -> dict:
    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        key, _value = item
        return (0, int(key)) if str(key).isdigit() else (1, str(key))

    return dict(sorted(counter.items(), key=sort_key))


def _canvas_size(target: dict) -> tuple[float, float]:
    size = target.get("size", [256, 256]) or [256, 256]
    width = float(size[0]) if len(size) >= 1 else 256.0
    height = float(size[1]) if len(size) >= 2 else width
    return width, height


def _bbox_from_points(points: Sequence[Sequence[float]]) -> list[float] | None:
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _flatten_polygon_points(rings: Iterable[tuple[List[List[float]], List[List[List[float]]]]]) -> List[List[float]]:
    points: List[List[float]] = []
    for outer, holes in rings:
        points.extend(outer)
        for hole in holes:
            points.extend(hole)
    return points


def _bbox_intersects_canvas(bbox: Sequence[float], *, width: float, height: float) -> bool:
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    return max_x >= 0.0 and max_y >= 0.0 and min_x <= width and min_y <= height


def _bbox_inside_canvas(bbox: Sequence[float], *, width: float, height: float) -> bool:
    min_x, min_y, max_x, max_y = [float(value) for value in bbox]
    return min_x >= 0.0 and min_y >= 0.0 and max_x <= width and max_y <= height


def _point_corner(x: float, y: float, *, width: float, height: float, margin: float) -> str | None:
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        return None
    left = x <= margin
    right = x >= width - margin
    top = y <= margin
    bottom = y >= height - margin
    if left and top:
        return "top_left"
    if right and top:
        return "top_right"
    if left and bottom:
        return "bottom_left"
    if right and bottom:
        return "bottom_right"
    return None


def _point_near_edge(x: float, y: float, *, width: float, height: float, margin: float) -> bool:
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        return False
    return x <= margin or y <= margin or x >= width - margin or y >= height - margin


def _point_quadrant(x: float, y: float, *, width: float, height: float) -> str:
    horizontal = "left" if x < width / 2.0 else "right"
    vertical = "top" if y < height / 2.0 else "bottom"
    return f"{vertical}_{horizontal}"


def _origin_from_frame(node: dict) -> tuple[float, float]:
    frame = node.get("frame", {}) or {}
    origin = frame.get("origin", [0.0, 0.0]) or [0.0, 0.0]
    return float(origin[0]), float(origin[1])


def _scale_from_frame(node: dict) -> float:
    frame = node.get("frame", {}) or {}
    return float(frame.get("scale", 1.0))


def _is_renderable_polygon_node(node: dict) -> bool:
    return (
        bool(node.get("renderable", True))
        and not bool(node.get("is_reference_only", False))
        and str(node.get("geometry_model", "none")) == "polygon_code"
    )


def audit_manual_parse_graph_target_spatial(
    target: dict,
    *,
    source: str | None = None,
    edge_margin: float = 16.0,
    min_bbox_area: float = 1.0,
) -> dict:
    width, height = _canvas_size(target)
    nodes = list((target.get("parse_graph", {}) or {}).get("nodes", []) or [])
    node_rows: List[dict] = []
    role_histogram = Counter()
    label_histogram = Counter()
    origin_corner_histogram = Counter()
    bbox_center_corner_histogram = Counter()
    origin_quadrant_histogram = Counter()

    for node in nodes:
        if not _is_renderable_polygon_node(node):
            continue

        role = str(node.get("role", ""))
        label = str(node.get("label", 0))
        role_histogram[role] += 1
        label_histogram[label] += 1

        rings = list(polygon_world_rings(node))
        points = _flatten_polygon_points(rings)
        bbox = _bbox_from_points(points)
        origin_x, origin_y = _origin_from_frame(node)
        scale = _scale_from_frame(node)
        polygon_count = len(rings)
        has_bbox = bbox is not None
        bbox_width = 0.0
        bbox_height = 0.0
        bbox_area = 0.0
        bbox_center_x = None
        bbox_center_y = None
        intersects_canvas = False
        inside_canvas = False
        bbox_center_near_edge = False
        bbox_center_corner = None
        if bbox is not None:
            bbox_width = max(0.0, float(bbox[2]) - float(bbox[0]))
            bbox_height = max(0.0, float(bbox[3]) - float(bbox[1]))
            bbox_area = bbox_width * bbox_height
            bbox_center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            bbox_center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            intersects_canvas = _bbox_intersects_canvas(bbox, width=width, height=height)
            inside_canvas = _bbox_inside_canvas(bbox, width=width, height=height)
            bbox_center_near_edge = _point_near_edge(
                bbox_center_x,
                bbox_center_y,
                width=width,
                height=height,
                margin=float(edge_margin),
            )
            bbox_center_corner = _point_corner(
                bbox_center_x,
                bbox_center_y,
                width=width,
                height=height,
                margin=float(edge_margin),
            )
            if bbox_center_corner is not None:
                bbox_center_corner_histogram[bbox_center_corner] += 1

        origin_inside_canvas = 0.0 <= origin_x <= width and 0.0 <= origin_y <= height
        origin_near_edge = _point_near_edge(origin_x, origin_y, width=width, height=height, margin=float(edge_margin))
        origin_corner = _point_corner(origin_x, origin_y, width=width, height=height, margin=float(edge_margin))
        if origin_corner is not None:
            origin_corner_histogram[origin_corner] += 1
        origin_quadrant = _point_quadrant(origin_x, origin_y, width=width, height=height)
        origin_quadrant_histogram[origin_quadrant] += 1

        node_rows.append(
            {
                "node_id": str(node.get("id", "")),
                "role": role,
                "label": label,
                "polygon_count": int(polygon_count),
                "origin": [float(origin_x), float(origin_y)],
                "scale": float(scale),
                "bbox": [float(value) for value in bbox] if bbox is not None else None,
                "bbox_width": float(bbox_width),
                "bbox_height": float(bbox_height),
                "bbox_area": float(bbox_area),
                "bbox_center": [float(bbox_center_x), float(bbox_center_y)] if bbox_center_x is not None else None,
                "has_world_bbox": bool(has_bbox),
                "bbox_intersects_canvas": bool(intersects_canvas),
                "bbox_inside_canvas": bool(inside_canvas),
                "bbox_tiny": bool(bbox_area <= float(min_bbox_area)),
                "bbox_center_near_edge": bool(bbox_center_near_edge),
                "bbox_center_corner": bbox_center_corner,
                "origin_inside_canvas": bool(origin_inside_canvas),
                "origin_near_edge": bool(origin_near_edge),
                "origin_corner": origin_corner,
                "origin_quadrant": origin_quadrant,
            }
        )

    visible_rows = [
        row
        for row in node_rows
        if bool(row["bbox_intersects_canvas"]) and not bool(row["bbox_tiny"])
    ]
    invisible_rows = [
        row
        for row in node_rows
        if not bool(row["bbox_intersects_canvas"]) or bool(row["bbox_tiny"])
    ]

    return {
        "source": source,
        "canvas_size": [float(width), float(height)],
        "edge_margin": float(edge_margin),
        "min_bbox_area": float(min_bbox_area),
        "node_count": int(len(nodes)),
        "renderable_polygon_node_count": int(len(node_rows)),
        "visible_polygon_node_count": int(len(visible_rows)),
        "invisible_polygon_node_count": int(len(invisible_rows)),
        "missing_world_bbox_count": int(sum(1 for row in node_rows if not row["has_world_bbox"])),
        "tiny_bbox_count": int(sum(1 for row in node_rows if row["bbox_tiny"])),
        "bbox_intersects_canvas_count": int(sum(1 for row in node_rows if row["bbox_intersects_canvas"])),
        "bbox_inside_canvas_count": int(sum(1 for row in node_rows if row["bbox_inside_canvas"])),
        "bbox_center_near_edge_count": int(sum(1 for row in node_rows if row["bbox_center_near_edge"])),
        "bbox_center_corner_count": int(sum(1 for row in node_rows if row["bbox_center_corner"] is not None)),
        "origin_inside_canvas_count": int(sum(1 for row in node_rows if row["origin_inside_canvas"])),
        "origin_near_edge_count": int(sum(1 for row in node_rows if row["origin_near_edge"])),
        "origin_corner_count": int(sum(1 for row in node_rows if row["origin_corner"] is not None)),
        "origin_x_stats": _numeric_stats([row["origin"][0] for row in node_rows]),
        "origin_y_stats": _numeric_stats([row["origin"][1] for row in node_rows]),
        "scale_stats": _numeric_stats([row["scale"] for row in node_rows]),
        "bbox_width_stats": _numeric_stats([row["bbox_width"] for row in node_rows if row["has_world_bbox"]]),
        "bbox_height_stats": _numeric_stats([row["bbox_height"] for row in node_rows if row["has_world_bbox"]]),
        "bbox_area_stats": _numeric_stats([row["bbox_area"] for row in node_rows if row["has_world_bbox"]]),
        "role_histogram": dict(sorted(role_histogram.items())),
        "label_histogram": _sort_histogram(label_histogram),
        "origin_quadrant_histogram": dict(sorted(origin_quadrant_histogram.items())),
        "origin_corner_histogram": dict(sorted(origin_corner_histogram.items())),
        "bbox_center_corner_histogram": dict(sorted(bbox_center_corner_histogram.items())),
        "nodes": node_rows,
    }


def audit_manual_parse_graph_targets_spatial(
    paths: Sequence[Path],
    *,
    edge_margin: float = 16.0,
    min_bbox_area: float = 1.0,
) -> dict:
    rows: List[dict] = []
    load_errors: List[dict] = []
    for path in paths:
        try:
            target = load_json(Path(path))
            rows.append(
                audit_manual_parse_graph_target_spatial(
                    target,
                    source=str(Path(path).as_posix()),
                    edge_margin=float(edge_margin),
                    min_bbox_area=float(min_bbox_area),
                )
            )
        except Exception as exc:
            load_errors.append({"source": str(Path(path).as_posix()), "error": f"{type(exc).__name__}:{exc}"})

    role_histogram = Counter()
    label_histogram = Counter()
    origin_quadrant_histogram = Counter()
    origin_corner_histogram = Counter()
    bbox_center_corner_histogram = Counter()
    all_nodes: List[dict] = []
    for row in rows:
        role_histogram.update(row["role_histogram"])
        label_histogram.update(row["label_histogram"])
        origin_quadrant_histogram.update(row["origin_quadrant_histogram"])
        origin_corner_histogram.update(row["origin_corner_histogram"])
        bbox_center_corner_histogram.update(row["bbox_center_corner_histogram"])
        all_nodes.extend(row["nodes"])

    return {
        "format": "maskgen_manual_parse_graph_spatial_audit_v1",
        "input_path_count": int(len(paths)),
        "loaded_count": int(len(rows)),
        "load_error_count": int(len(load_errors)),
        "edge_margin": float(edge_margin),
        "min_bbox_area": float(min_bbox_area),
        "renderable_polygon_node_count": int(sum(row["renderable_polygon_node_count"] for row in rows)),
        "visible_polygon_node_count": int(sum(row["visible_polygon_node_count"] for row in rows)),
        "invisible_polygon_node_count": int(sum(row["invisible_polygon_node_count"] for row in rows)),
        "missing_world_bbox_count": int(sum(row["missing_world_bbox_count"] for row in rows)),
        "tiny_bbox_count": int(sum(row["tiny_bbox_count"] for row in rows)),
        "bbox_intersects_canvas_count": int(sum(row["bbox_intersects_canvas_count"] for row in rows)),
        "bbox_inside_canvas_count": int(sum(row["bbox_inside_canvas_count"] for row in rows)),
        "bbox_center_near_edge_count": int(sum(row["bbox_center_near_edge_count"] for row in rows)),
        "bbox_center_corner_count": int(sum(row["bbox_center_corner_count"] for row in rows)),
        "origin_inside_canvas_count": int(sum(row["origin_inside_canvas_count"] for row in rows)),
        "origin_near_edge_count": int(sum(row["origin_near_edge_count"] for row in rows)),
        "origin_corner_count": int(sum(row["origin_corner_count"] for row in rows)),
        "origin_x_stats": _numeric_stats([node["origin"][0] for node in all_nodes]),
        "origin_y_stats": _numeric_stats([node["origin"][1] for node in all_nodes]),
        "scale_stats": _numeric_stats([node["scale"] for node in all_nodes]),
        "bbox_width_stats": _numeric_stats([node["bbox_width"] for node in all_nodes if node["has_world_bbox"]]),
        "bbox_height_stats": _numeric_stats([node["bbox_height"] for node in all_nodes if node["has_world_bbox"]]),
        "bbox_area_stats": _numeric_stats([node["bbox_area"] for node in all_nodes if node["has_world_bbox"]]),
        "role_histogram": dict(sorted(role_histogram.items())),
        "label_histogram": _sort_histogram(label_histogram),
        "origin_quadrant_histogram": dict(sorted(origin_quadrant_histogram.items())),
        "origin_corner_histogram": dict(sorted(origin_corner_histogram.items())),
        "bbox_center_corner_histogram": dict(sorted(bbox_center_corner_histogram.items())),
        "load_errors": load_errors,
        "rows": rows,
    }
