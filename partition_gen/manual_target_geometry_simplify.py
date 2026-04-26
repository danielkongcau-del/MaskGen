from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from partition_gen.manual_target_token_stats import polygon_payload_stats

Point = Tuple[float, float]


@dataclass(frozen=True)
class ManualTargetSimplifyConfig:
    profile: str = "light"
    simplify_tolerance: float | None = None
    max_ring_vertices: int | None = None
    tolerance_units: str = "pixel"
    remove_collinear_eps: float = 1e-6
    preserve_topology: bool = True
    min_valid_area: float = 1e-8
    simplify_renderable_only: bool = True
    simplify_polygon_code_only: bool = True


PROFILE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "none": {"simplify_tolerance": 0.0, "max_ring_vertices": None},
    "light": {"simplify_tolerance": 0.25, "max_ring_vertices": 64},
    "medium": {"simplify_tolerance": 0.5, "max_ring_vertices": 48},
    "aggressive": {"simplify_tolerance": 1.0, "max_ring_vertices": 32},
}


def resolve_simplify_config(config: ManualTargetSimplifyConfig) -> ManualTargetSimplifyConfig:
    defaults = PROFILE_DEFAULTS.get(str(config.profile))
    if defaults is None and str(config.profile) != "custom":
        raise ValueError(f"Unknown simplify profile: {config.profile}")
    if defaults is None:
        return config
    updates = {}
    if config.simplify_tolerance is None:
        updates["simplify_tolerance"] = defaults["simplify_tolerance"]
    if config.max_ring_vertices is None:
        updates["max_ring_vertices"] = defaults["max_ring_vertices"]
    return replace(config, **updates)


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
    ring = [(float(point[0]), float(point[1])) for point in points]
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1e-12:
        ring = ring[:-1]
    return ring


def _ring_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        total += point[0] * other[1] - other[0] * point[1]
    return total * 0.5


def _point_line_distance(point: Point, left: Point, right: Point) -> float:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    denom = math.hypot(dx, dy)
    if denom <= 1e-12:
        return math.hypot(point[0] - left[0], point[1] - left[1])
    return abs(dx * (left[1] - point[1]) - (left[0] - point[0]) * dy) / denom


def remove_near_collinear_points(points: Sequence[Sequence[float]], eps: float = 1e-6) -> List[List[float]]:
    ring = _trim_ring(points)
    if len(ring) <= 3:
        return [[float(x), float(y)] for x, y in ring]
    deduped: List[Point] = []
    for point in ring:
        if not deduped or math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) > eps:
            deduped.append(point)
    if len(deduped) >= 2 and math.hypot(deduped[0][0] - deduped[-1][0], deduped[0][1] - deduped[-1][1]) <= eps:
        deduped.pop()
    ring = deduped
    changed = True
    while changed and len(ring) > 3:
        changed = False
        kept: List[Point] = []
        for index, point in enumerate(ring):
            prev_point = ring[index - 1]
            next_point = ring[(index + 1) % len(ring)]
            if _point_line_distance(point, prev_point, next_point) <= eps and len(ring) - 1 >= 3:
                changed = True
                continue
            kept.append(point)
        if len(kept) < 3:
            break
        ring = kept
    return [[float(x), float(y)] for x, y in ring]


def _importance(points: Sequence[Point], index: int) -> float:
    point = points[index]
    left = points[index - 1]
    right = points[(index + 1) % len(points)]
    return _point_line_distance(point, left, right)


def limit_ring_vertices(points: Sequence[Sequence[float]], max_vertices: int | None) -> List[List[float]]:
    ring = _trim_ring(points)
    if max_vertices is None or len(ring) <= int(max_vertices):
        return [[float(x), float(y)] for x, y in ring]
    max_vertices = max(3, int(max_vertices))
    while len(ring) > max_vertices and len(ring) > 3:
        candidates = [(_importance(ring, index), index) for index in range(len(ring))]
        _, remove_index = min(candidates, key=lambda item: (item[0], item[1]))
        ring = [point for index, point in enumerate(ring) if index != remove_index]
    return [[float(x), float(y)] for x, y in ring]


def _payload_components(payload: Dict[str, object]) -> List[Dict[str, object]]:
    polygons = payload.get("polygons_local")
    if polygons:
        return [dict(polygon) for polygon in polygons]
    return [{"outer_local": payload.get("outer_local", []), "holes_local": payload.get("holes_local", [])}]


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            if not polygon.is_empty:
                yield polygon
        return
    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def polygon_payload_to_shapely(payload: Dict[str, object]):
    polygons = []
    for component in _payload_components(payload):
        outer = _trim_ring(component.get("outer_local", []))
        holes = [_trim_ring(hole) for hole in component.get("holes_local", []) or []]
        holes = [hole for hole in holes if len(hole) >= 3]
        if len(outer) < 3:
            continue
        try:
            polygons.append(Polygon(outer, holes))
        except Exception:
            continue
    if not polygons:
        return Polygon()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def shapely_to_polygon_payload(geometry) -> Dict[str, object]:
    polygons = [orient(polygon, sign=1.0) for polygon in _iter_polygons(geometry)]
    polygons = [polygon for polygon in polygons if not polygon.is_empty and polygon.area > 0.0]
    polygons.sort(key=lambda item: -float(item.area))
    polygons_local = []
    for polygon in polygons:
        outer = [[float(x), float(y)] for x, y in _trim_ring(polygon.exterior.coords)]
        holes = [
            [[float(x), float(y)] for x, y in _trim_ring(interior.coords)]
            for interior in polygon.interiors
            if len(_trim_ring(interior.coords)) >= 3
        ]
        polygons_local.append({"outer_local": outer, "holes_local": holes})
    if not polygons_local:
        return {"outer_local": [], "holes_local": [], "polygons_local": []}
    return {
        "outer_local": polygons_local[0]["outer_local"],
        "holes_local": polygons_local[0]["holes_local"],
        "polygons_local": polygons_local,
    }


def _clean_payload_rings(payload: Dict[str, object], config: ManualTargetSimplifyConfig) -> Dict[str, object]:
    components = []
    for component in _payload_components(payload):
        outer = remove_near_collinear_points(component.get("outer_local", []), eps=float(config.remove_collinear_eps))
        holes = [
            remove_near_collinear_points(hole, eps=float(config.remove_collinear_eps))
            for hole in component.get("holes_local", []) or []
        ]
        holes = [hole for hole in holes if len(hole) >= 3]
        components.append({"outer_local": outer, "holes_local": holes})
    if not components:
        return {"outer_local": [], "holes_local": [], "polygons_local": []}
    return {"outer_local": components[0]["outer_local"], "holes_local": components[0]["holes_local"], "polygons_local": components}


def _limit_payload_rings(payload: Dict[str, object], config: ManualTargetSimplifyConfig) -> Dict[str, object]:
    components = []
    for component in _payload_components(payload):
        outer = limit_ring_vertices(component.get("outer_local", []), config.max_ring_vertices)
        holes = [limit_ring_vertices(hole, config.max_ring_vertices) for hole in component.get("holes_local", []) or []]
        holes = [hole for hole in holes if len(hole) >= 3]
        components.append({"outer_local": outer, "holes_local": holes})
    if not components:
        return {"outer_local": [], "holes_local": [], "polygons_local": []}
    return {"outer_local": components[0]["outer_local"], "holes_local": components[0]["holes_local"], "polygons_local": components}


def count_polygon_vertices(node_or_payload: Dict[str, object]) -> int:
    payload = node_or_payload.get("geometry", node_or_payload)
    stats = polygon_payload_stats(payload)
    return int(stats["polygon_vertex_count"])


def _local_tolerance(config: ManualTargetSimplifyConfig, frame: Dict[str, object]) -> float:
    tolerance = float(config.simplify_tolerance or 0.0)
    if tolerance <= 0.0:
        return 0.0
    if str(config.tolerance_units) == "pixel":
        scale = max(float(frame.get("scale", 1.0)), 1e-8)
        return float(tolerance / scale)
    if str(config.tolerance_units) == "local":
        return tolerance
    raise ValueError(f"Unknown tolerance units: {config.tolerance_units}")


def simplify_polygon_payload(
    payload: Dict[str, object],
    config: ManualTargetSimplifyConfig,
    *,
    frame: Dict[str, object] | None = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    config = resolve_simplify_config(config)
    frame = frame or {}
    original_payload = copy.deepcopy(payload)
    original_vertices = int(polygon_payload_stats(original_payload)["polygon_vertex_count"])
    original_geometry = polygon_payload_to_shapely(original_payload)
    if original_geometry.is_empty or float(original_geometry.area) <= float(config.min_valid_area):
        return original_payload, {
            "success": False,
            "failure_reason": "invalid_original_geometry",
            "original_vertex_count": original_vertices,
            "simplified_vertex_count": original_vertices,
            "area_error": 0.0,
            "area_error_ratio": 0.0,
        }

    cleaned_payload = _clean_payload_rings(original_payload, config)
    geometry = polygon_payload_to_shapely(cleaned_payload)
    if geometry.is_empty:
        return original_payload, {
            "success": False,
            "failure_reason": "empty_after_collinear_cleanup",
            "original_vertex_count": original_vertices,
            "simplified_vertex_count": original_vertices,
            "area_error": 0.0,
            "area_error_ratio": 0.0,
        }

    tolerance = _local_tolerance(config, frame)
    if tolerance > 0.0:
        geometry = geometry.simplify(tolerance, preserve_topology=bool(config.preserve_topology))
    simplified_payload = shapely_to_polygon_payload(geometry)
    simplified_payload = _limit_payload_rings(simplified_payload, config)
    simplified_geometry = polygon_payload_to_shapely(simplified_payload)
    if not simplified_geometry.is_valid:
        fixed = simplified_geometry.buffer(0)
        if fixed.is_empty or not fixed.is_valid:
            return original_payload, {
                "success": False,
                "failure_reason": "invalid_simplified_geometry",
                "original_vertex_count": original_vertices,
                "simplified_vertex_count": original_vertices,
                "area_error": 0.0,
                "area_error_ratio": 0.0,
            }
        simplified_geometry = fixed
        simplified_payload = shapely_to_polygon_payload(fixed)

    if simplified_geometry.is_empty or float(simplified_geometry.area) <= float(config.min_valid_area):
        return original_payload, {
            "success": False,
            "failure_reason": "empty_simplified_geometry",
            "original_vertex_count": original_vertices,
            "simplified_vertex_count": original_vertices,
            "area_error": 0.0,
            "area_error_ratio": 0.0,
        }

    simplified_vertices = int(polygon_payload_stats(simplified_payload)["polygon_vertex_count"])
    area_error = abs(float(original_geometry.area) - float(simplified_geometry.area))
    area_error_ratio = float(area_error / max(float(original_geometry.area), float(config.min_valid_area)))
    return simplified_payload, {
        "success": True,
        "failure_reason": None,
        "original_vertex_count": original_vertices,
        "simplified_vertex_count": simplified_vertices,
        "vertex_reduction": int(original_vertices - simplified_vertices),
        "area_error": float(area_error),
        "area_error_ratio": float(area_error_ratio),
        "local_tolerance": float(tolerance),
    }


def _should_simplify_node(node: Dict[str, object], config: ManualTargetSimplifyConfig) -> bool:
    if config.simplify_renderable_only and not bool(node.get("renderable", True)):
        return False
    if bool(node.get("is_reference_only", False)):
        return False
    if str(node.get("role")) == "insert_object_group":
        return False
    if config.simplify_polygon_code_only and str(node.get("geometry_model")) != "polygon_code":
        return False
    return str(node.get("geometry_model")) == "polygon_code"


def simplify_manual_generator_target(
    target: Dict[str, object],
    *,
    config: ManualTargetSimplifyConfig,
) -> tuple[Dict[str, object], Dict[str, object]]:
    config = resolve_simplify_config(config)
    simplified = copy.deepcopy(target)
    graph = simplified.setdefault("parse_graph", {})
    nodes = graph.setdefault("nodes", [])
    node_diagnostics: List[Dict[str, object]] = []
    original_vertex_count = 0
    simplified_vertex_count = 0
    simplified_node_count = 0
    failed_node_count = 0
    invalid_geometry_count = 0
    area_error_total = 0.0
    area_error_ratios: List[float] = []

    for node in nodes:
        original_vertices = count_polygon_vertices(node) if str(node.get("geometry_model")) == "polygon_code" else 0
        original_vertex_count += original_vertices
        if not _should_simplify_node(node, config):
            simplified_vertex_count += original_vertices
            continue
        new_payload, diag = simplify_polygon_payload(node.get("geometry", {}) or {}, config, frame=node.get("frame", {}) or {})
        node_diag = {
            "id": node.get("id"),
            "role": node.get("role"),
            "label": node.get("label"),
            **diag,
        }
        if bool(diag.get("success")):
            node["geometry"] = new_payload
            if int(diag["simplified_vertex_count"]) < int(diag["original_vertex_count"]):
                simplified_node_count += 1
            area_error_total += float(diag.get("area_error", 0.0))
            area_error_ratios.append(float(diag.get("area_error_ratio", 0.0)))
            simplified_vertex_count += int(diag["simplified_vertex_count"])
        else:
            failed_node_count += 1
            invalid_geometry_count += 1
            simplified_vertex_count += original_vertices
        node_diagnostics.append(node_diag)

    total_reduction = int(original_vertex_count - simplified_vertex_count)
    diagnostics = {
        "profile": config.profile,
        "config": asdict(config),
        "node_count": int(len(nodes)),
        "simplified_node_count": int(simplified_node_count),
        "failed_node_count": int(failed_node_count),
        "original_vertex_count": int(original_vertex_count),
        "simplified_vertex_count": int(simplified_vertex_count),
        "vertex_reduction": int(total_reduction),
        "vertex_reduction_ratio": float(total_reduction / original_vertex_count) if original_vertex_count else 0.0,
        "invalid_geometry_count": int(invalid_geometry_count),
        "area_error_total": float(area_error_total),
        "area_error_ratio_mean": float(sum(area_error_ratios) / len(area_error_ratios)) if area_error_ratios else 0.0,
        "max_node_area_error_ratio": float(max(area_error_ratios)) if area_error_ratios else 0.0,
        "node_diagnostics": node_diagnostics,
        "top_changed_nodes": sorted(
            [item for item in node_diagnostics if int(item.get("vertex_reduction", 0)) > 0],
            key=lambda item: int(item.get("vertex_reduction", 0)),
            reverse=True,
        )[:20],
    }
    metadata = simplified.setdefault("metadata", {})
    metadata["geometry_simplified"] = True
    metadata["simplify_profile"] = config.profile
    metadata["simplify_diagnostics"] = diagnostics
    return simplified, diagnostics
