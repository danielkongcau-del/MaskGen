from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union

from partition_gen.dual_graph import face_polygon, load_vertices


Point = Tuple[float, float]


@dataclass(frozen=True)
class PrimitiveRecord:
    primitive_id: int
    primitive_type: str
    vertices: Tuple[Tuple[float, float], ...]
    area: float


@dataclass(frozen=True)
class PrimitiveCompressionConfig:
    max_group_size: int = 8
    min_iou: float = 0.72
    error_weight: float = 3.5


@dataclass(frozen=True)
class StripCoverConfig:
    min_aspect_ratio: float = 2.0
    max_angle_delta_deg: float = 18.0
    normal_offset_scale: float = 1.35
    width_scale: float = 1.1
    min_precision: float = 0.72
    min_support_ratio: float = 0.72


@dataclass(frozen=True)
class StripRefineConfig:
    max_group_size: int = 4
    min_aspect_ratio: float = 2.0
    max_angle_delta_deg: float = 18.0
    normal_offset_scale: float = 1.75
    width_scale: float = 1.1
    merge_error_weight: float = 0.55
    min_candidate_quality: float = 0.6
    cap_line_offset_scale: float = 1.5
    cap_endpoint_margin_scale: float = 2.5
    max_cap_area_ratio: float = 0.35


@dataclass(frozen=True)
class CompositeGroupConfig:
    max_candidate_area_ratio: float = 0.5
    normal_offset_scale: float = 2.0
    endpoint_margin_scale: float = 3.0
    area_gain_weight: float = 1.0
    edge_cost_weight: float = 0.45
    connectivity_bonus_weight: float = 0.35
    fit_gain_weight: float = 0.5
    hole_invasion_weight: float = 2.5
    hole_loss_weight: float = 0.75
    min_score: float = 0.02
    max_fit_drop: float = 0.08


def _trim_ring(points: Sequence[Point], *, eps: float = 1e-6) -> List[Point]:
    ring = [tuple(float(value) for value in point) for point in points]
    if len(ring) >= 2:
        first = ring[0]
        last = ring[-1]
        if abs(first[0] - last[0]) <= eps and abs(first[1] - last[1]) <= eps:
            ring = ring[:-1]
    if len(ring) < 3:
        return ring

    changed = True
    while changed and len(ring) >= 3:
        changed = False
        kept: List[Point] = []
        for index, point in enumerate(ring):
            prev_point = ring[index - 1]
            next_point = ring[(index + 1) % len(ring)]
            cross = (point[0] - prev_point[0]) * (next_point[1] - point[1]) - (point[1] - prev_point[1]) * (
                next_point[0] - point[0]
            )
            if abs(cross) <= eps:
                changed = True
                continue
            kept.append(point)
        ring = kept
    return ring


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry.is_empty:
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


def _polygon_ring_vertices(polygon: Polygon) -> List[Point]:
    polygon = orient(polygon, sign=1.0)
    return _trim_ring(list(polygon.exterior.coords))


def _polygon_hole_vertices(polygon: Polygon) -> List[List[Point]]:
    polygon = orient(polygon, sign=1.0)
    holes: List[List[Point]] = []
    for interior in polygon.interiors:
        ring = _trim_ring(list(interior.coords))
        if len(ring) >= 3:
            holes.append(ring)
    return holes


def _primitive_type_from_polygon(polygon: Polygon) -> str | None:
    vertices = _polygon_ring_vertices(polygon)
    if len(vertices) == 3:
        return "triangle"
    if len(vertices) == 4:
        return "quad"
    return None


def _approx_iou(geometry_a: Polygon, geometry_b: Polygon) -> float:
    intersection = float(geometry_a.intersection(geometry_b).area)
    union = float(geometry_a.union(geometry_b).area)
    return intersection / max(union, 1e-6)


def primitive_polygon(primitive: Dict[str, object] | PrimitiveRecord) -> Polygon:
    vertices = primitive.vertices if isinstance(primitive, PrimitiveRecord) else primitive["vertices"]
    polygon = Polygon(vertices)
    return orient(polygon, sign=1.0)


def primitives_union_geometry(primitives: Sequence[Dict[str, object] | PrimitiveRecord]) -> Polygon:
    polygons = [primitive_polygon(primitive) for primitive in primitives]
    if not polygons:
        return Polygon()
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        return orient(merged, sign=1.0)
    geoms = [geom for geom in _iter_polygons(merged)]
    if not geoms:
        return Polygon()
    union_geometry = unary_union(geoms)
    if isinstance(union_geometry, Polygon):
        return orient(union_geometry, sign=1.0)
    geoms = [geom for geom in _iter_polygons(union_geometry)]
    return orient(unary_union(geoms), sign=1.0) if geoms else Polygon()


def geometry_iou(geometry_a: Polygon, geometry_b: Polygon) -> float:
    return _approx_iou(geometry_a, geometry_b)


def _angle_mod_pi(angle_rad: float) -> float:
    value = float(angle_rad) % math.pi
    return value if value >= 0.0 else value + math.pi


def _angle_delta_deg(angle_a: float, angle_b: float) -> float:
    delta = abs(_angle_mod_pi(angle_a) - _angle_mod_pi(angle_b))
    delta = min(delta, math.pi - delta)
    return math.degrees(delta)


def _weighted_axis_angle(angles: Sequence[float], weights: Sequence[float]) -> float:
    doubled = np.asarray([2.0 * _angle_mod_pi(angle) for angle in angles], dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    vector = np.array(
        [
            float(np.sum(np.cos(doubled) * weights_array)),
            float(np.sum(np.sin(doubled) * weights_array)),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(vector)) <= 1e-8:
        return _angle_mod_pi(float(angles[0]) if angles else 0.0)
    return _angle_mod_pi(0.5 * math.atan2(float(vector[1]), float(vector[0])))


def _edge_lengths(ring: Sequence[Point]) -> List[float]:
    lengths: List[float] = []
    for index in range(len(ring)):
        point_a = ring[index]
        point_b = ring[(index + 1) % len(ring)]
        lengths.append(float(math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])))
    return lengths


def _oriented_quad_from_axis(
    *,
    axis_angle: float,
    t_min: float,
    t_max: float,
    s_center: float,
    half_width: float,
) -> Polygon:
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    corners = [
        axis * t_min + normal * (s_center - half_width),
        axis * t_max + normal * (s_center - half_width),
        axis * t_max + normal * (s_center + half_width),
        axis * t_min + normal * (s_center + half_width),
    ]
    return orient(Polygon([(float(point[0]), float(point[1])) for point in corners]), sign=1.0)


def primitive_strip_stats(primitive: Dict[str, object] | PrimitiveRecord) -> Dict[str, object]:
    polygon = primitive_polygon(primitive)
    rectangle = orient(polygon.minimum_rotated_rectangle, sign=1.0)
    ring = _polygon_ring_vertices(rectangle)
    lengths = _edge_lengths(ring)
    if len(lengths) != 4:
        angle = 0.0
        long_edge = float(math.sqrt(max(float(polygon.area), 1e-6)))
        short_edge = long_edge
    else:
        max_index = int(np.argmax(np.asarray(lengths, dtype=np.float64)))
        point_a = ring[max_index]
        point_b = ring[(max_index + 1) % len(ring)]
        angle = _angle_mod_pi(math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0]))
        long_edge = float(max(lengths))
        short_edge = float(min(lengths))
    centroid = polygon.centroid
    return {
        "id": int(primitive.primitive_id if isinstance(primitive, PrimitiveRecord) else primitive["id"]),
        "polygon": polygon,
        "angle": float(angle),
        "length": float(long_edge),
        "width": float(max(short_edge, 1e-6)),
        "aspect_ratio": float(long_edge / max(short_edge, 1e-6)),
        "centroid": (float(centroid.x), float(centroid.y)),
        "area": float(polygon.area),
    }


def _similar_strip_primitives(
    stat_a: Dict[str, object],
    stat_b: Dict[str, object],
    *,
    config: StripCoverConfig,
) -> bool:
    if max(float(stat_a["aspect_ratio"]), float(stat_b["aspect_ratio"])) < float(config.min_aspect_ratio):
        return False
    angle_a = float(stat_a["angle"])
    angle_b = float(stat_b["angle"])
    if _angle_delta_deg(angle_a, angle_b) > float(config.max_angle_delta_deg):
        return False

    mean_angle = _weighted_axis_angle([angle_a, angle_b], [float(stat_a["area"]), float(stat_b["area"])])
    normal = np.array([-math.sin(mean_angle), math.cos(mean_angle)], dtype=np.float64)
    center_a = np.asarray(stat_a["centroid"], dtype=np.float64)
    center_b = np.asarray(stat_b["centroid"], dtype=np.float64)
    offset = abs(float(np.dot(center_b - center_a, normal)))
    width_limit = float(config.normal_offset_scale) * max(float(stat_a["width"]), float(stat_b["width"]))
    return offset <= width_limit


def _projection_interval(stat: Dict[str, object], axis_angle: float) -> Tuple[float, float]:
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    points = np.asarray(list(stat["polygon"].exterior.coords)[:-1], dtype=np.float64)
    values = points @ axis
    return float(np.min(values)), float(np.max(values))


def _attachable_strip_cap(
    stat_a: Dict[str, object],
    stat_b: Dict[str, object],
    *,
    config: StripRefineConfig,
) -> bool:
    main, other = (stat_a, stat_b) if float(stat_a["area"]) >= float(stat_b["area"]) else (stat_b, stat_a)
    if float(main["aspect_ratio"]) < float(config.min_aspect_ratio):
        return False
    if float(other["area"]) > float(main["area"]) * float(config.max_cap_area_ratio):
        return False

    axis_angle = float(main["angle"])
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    center_main = np.asarray(main["centroid"], dtype=np.float64)
    center_other = np.asarray(other["centroid"], dtype=np.float64)
    normal_offset = abs(float(np.dot(center_other - center_main, normal)))
    if normal_offset > float(config.cap_line_offset_scale) * max(float(main["width"]), float(other["width"])):
        return False

    main_min, main_max = _projection_interval(main, axis_angle)
    other_min, other_max = _projection_interval(other, axis_angle)
    endpoint_margin = float(config.cap_endpoint_margin_scale) * max(float(main["width"]), float(other["width"]))
    near_low = (main_min - endpoint_margin) <= other_max <= (main_min + endpoint_margin)
    near_high = (main_max - endpoint_margin) <= other_min <= (main_max + endpoint_margin)
    return bool(near_low or near_high)


def _connected_components(adjacency: Dict[int, List[int]]) -> List[Tuple[int, ...]]:
    remaining = set(adjacency.keys())
    components: List[Tuple[int, ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda item: (len(item), item))


def _harmonic_quality(precision: float, support_ratio: float) -> float:
    if precision <= 0.0 or support_ratio <= 0.0:
        return 0.0
    return float(2.0 * precision * support_ratio / max(precision + support_ratio, 1e-6))


def _primitive_source_ids(primitive: Dict[str, object]) -> List[int]:
    for key in ["source_atom_ids", "covered_primitive_ids", "residual_from", "merged_from"]:
        values = primitive.get(key)
        if values:
            return [int(value) for value in values]
    return [int(primitive["id"])]


def _polygon_components(geometry) -> List[Polygon]:
    return [polygon for polygon in _iter_polygons(geometry)]


def _geometry_component_count(geometry) -> int:
    return int(len(_polygon_components(geometry)))


def _geometry_vertex_count(geometry) -> int:
    count = 0
    for polygon in _polygon_components(geometry):
        count += len(_polygon_ring_vertices(polygon))
    return int(count)


def _geometry_hole_count(geometry) -> int:
    count = 0
    for polygon in _polygon_components(geometry):
        count += len(polygon.interiors)
    return int(count)


def _geometry_hole_area(geometry) -> float:
    area = 0.0
    for polygon in _polygon_components(geometry):
        for interior in polygon.interiors:
            area += abs(float(Polygon(interior).area))
    return float(area)


def _protected_hole_geometry(geometry) -> Polygon:
    hole_polygons: List[Polygon] = []
    for polygon in _polygon_components(geometry):
        for interior in polygon.interiors:
            hole_polygon = Polygon(interior)
            if not hole_polygon.is_empty and float(hole_polygon.area) > 1e-6:
                hole_polygons.append(orient(hole_polygon, sign=1.0))
    if not hole_polygons:
        return Polygon()
    merged = unary_union(hole_polygons)
    if isinstance(merged, Polygon):
        return orient(merged, sign=1.0)
    geoms = [geom for geom in _iter_polygons(merged)]
    return orient(unary_union(geoms), sign=1.0) if geoms else Polygon()


def _serialize_geometry_components(geometry) -> List[Dict[str, object]]:
    components: List[Dict[str, object]] = []
    for polygon in _polygon_components(geometry):
        components.append(
            {
                "outer": [[float(x), float(y)] for x, y in _polygon_ring_vertices(polygon)],
                "holes": [
                    [[float(x), float(y)] for x, y in hole_ring]
                    for hole_ring in _polygon_hole_vertices(polygon)
                ],
            }
        )
    return components


def _serialize_geometry_polygons(geometry) -> List[List[List[float]]]:
    return [component["outer"] for component in _serialize_geometry_components(geometry)]


def _geometry_centroid(geometry) -> Tuple[float, float]:
    if geometry.is_empty:
        return (0.0, 0.0)
    centroid = geometry.centroid
    return (float(centroid.x), float(centroid.y))


def _axis_strip_candidate(
    component_stats: Sequence[Dict[str, object]],
    *,
    width_scale: float,
) -> Polygon:
    axis_angle = _weighted_axis_angle(
        [float(item["angle"]) for item in component_stats],
        [float(item["area"]) for item in component_stats],
    )
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    all_vertices = []
    for item in component_stats:
        all_vertices.extend(list(item["polygon"].exterior.coords)[:-1])
    points = np.asarray(all_vertices, dtype=np.float64)
    t_values = points @ axis
    centroid_offsets = np.asarray([np.dot(np.asarray(item["centroid"], dtype=np.float64), normal) for item in component_stats])
    widths = np.asarray([float(item["width"]) for item in component_stats], dtype=np.float64)
    weights = np.asarray([float(item["area"]) for item in component_stats], dtype=np.float64)
    width = float(np.average(widths, weights=weights) * float(width_scale))
    s_center = float(np.average(centroid_offsets, weights=weights))
    return _oriented_quad_from_axis(
        axis_angle=axis_angle,
        t_min=float(np.min(t_values)),
        t_max=float(np.max(t_values)),
        s_center=s_center,
        half_width=max(width * 0.5, 1e-3),
    )


def _candidate_metrics(
    candidate: Polygon | None,
    *,
    reference_geometry: Polygon,
    support_geometry: Polygon,
) -> Dict[str, float] | None:
    if candidate is None or candidate.is_empty:
        return None
    ring = _polygon_ring_vertices(candidate)
    if len(ring) != 4:
        return None
    candidate = orient(Polygon(ring), sign=1.0)
    precision = float(candidate.intersection(reference_geometry).area / max(float(candidate.area), 1e-6))
    support_ratio = float(candidate.intersection(support_geometry).area / max(float(support_geometry.area), 1e-6))
    return {
        "precision": float(precision),
        "support_ratio": float(support_ratio),
        "quality": _harmonic_quality(precision, support_ratio),
    }


def _convex_primitive_type(ring: Sequence[Point]) -> str | None:
    count = len(ring)
    if count < 3:
        return None
    if count == 3:
        return "triangle"
    if count == 4:
        return "quad"
    return "convex"


def _best_convex_candidate(
    component_primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
) -> Tuple[Polygon | None, Dict[str, float] | None]:
    if not component_primitives:
        return None, None
    support_geometry = primitives_union_geometry(component_primitives)
    if support_geometry.is_empty:
        return None, None
    candidate = orient(support_geometry.convex_hull, sign=1.0)
    if not isinstance(candidate, Polygon) or candidate.is_empty:
        return None, None
    ring = _polygon_ring_vertices(candidate)
    primitive_type = _convex_primitive_type(ring)
    if primitive_type is None:
        return None, None
    candidate = orient(Polygon(ring), sign=1.0)
    metrics = _candidate_metrics(candidate, reference_geometry=reference_geometry, support_geometry=support_geometry)
    if metrics is None:
        return None, None
    return candidate, {**metrics, "primitive_type": primitive_type, "vertex_count": float(len(ring))}


def _best_strip_candidate(
    component_primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
    width_scale: float,
) -> Tuple[Polygon | None, Dict[str, float] | None]:
    if not component_primitives:
        return None, None
    component_stats = [primitive_strip_stats(primitive) for primitive in component_primitives]
    support_geometry = primitives_union_geometry(component_primitives)
    candidates: List[Tuple[Polygon, Dict[str, float]]] = []

    axis_candidate = _axis_strip_candidate(component_stats, width_scale=width_scale)
    metrics = _candidate_metrics(axis_candidate, reference_geometry=reference_geometry, support_geometry=support_geometry)
    if metrics is not None:
        candidates.append((axis_candidate, metrics))

    best_candidate, best_type, _ = best_low_complexity_fit(support_geometry)
    if best_candidate is not None and best_type == "quad":
        metrics = _candidate_metrics(best_candidate, reference_geometry=reference_geometry, support_geometry=support_geometry)
        if metrics is not None:
            candidates.append((best_candidate, metrics))

    rectangle = orient(support_geometry.minimum_rotated_rectangle, sign=1.0)
    metrics = _candidate_metrics(rectangle, reference_geometry=reference_geometry, support_geometry=support_geometry)
    if metrics is not None:
        candidates.append((rectangle, metrics))

    if not candidates:
        return None, None
    candidate, metrics = max(candidates, key=lambda item: (item[1]["quality"], item[1]["precision"], item[1]["support_ratio"]))
    return candidate, metrics


def _group_strip_quality(
    member_primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
    width_scale: float,
) -> float:
    _, metrics = _best_strip_candidate(
        member_primitives,
        reference_geometry=reference_geometry,
        width_scale=width_scale,
    )
    return float(metrics["quality"]) if metrics is not None else 0.0


def _candidate_near_group(
    *,
    seed_primitive: Dict[str, object],
    current_geometry,
    candidate_primitive: Dict[str, object],
    config: CompositeGroupConfig,
) -> bool:
    seed_stats = primitive_strip_stats(seed_primitive)
    candidate_stats = primitive_strip_stats(candidate_primitive)
    if float(candidate_stats["area"]) > float(seed_stats["area"]) * float(config.max_candidate_area_ratio):
        return False

    seed_polygon = primitive_polygon(seed_primitive)
    if seed_polygon.intersects(primitive_polygon(candidate_primitive)):
        return True

    if current_geometry.intersects(primitive_polygon(candidate_primitive)):
        return True

    axis_angle = float(seed_stats["angle"])
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    seed_center = np.asarray(seed_stats["centroid"], dtype=np.float64)
    candidate_center = np.asarray(candidate_stats["centroid"], dtype=np.float64)
    normal_offset = abs(float(np.dot(candidate_center - seed_center, normal)))
    if normal_offset > float(config.normal_offset_scale) * max(float(seed_stats["width"]), float(candidate_stats["width"])):
        return False

    seed_min, seed_max = _projection_interval(seed_stats, axis_angle)
    cand_min, cand_max = _projection_interval(candidate_stats, axis_angle)
    endpoint_margin = float(config.endpoint_margin_scale) * max(float(seed_stats["width"]), float(candidate_stats["width"]))
    near_low = (seed_min - endpoint_margin) <= cand_max <= (seed_min + endpoint_margin)
    near_high = (seed_max - endpoint_margin) <= cand_min <= (seed_max + endpoint_margin)
    return bool(near_low or near_high)


def _candidate_patch_score(
    *,
    current_geometry,
    new_geometry,
    current_quality: float,
    new_quality: float,
    config: CompositeGroupConfig,
    protected_hole_geometry=None,
) -> Dict[str, float]:
    current_area = max(float(current_geometry.area), 1e-6)
    added_area = max(float(new_geometry.area) - float(current_geometry.area), 0.0)
    area_gain_term = added_area / current_area

    current_vertices = max(_geometry_vertex_count(current_geometry), 1)
    new_vertices = _geometry_vertex_count(new_geometry)
    edge_cost_term = max(new_vertices - current_vertices, 0) / current_vertices

    current_components = max(_geometry_component_count(current_geometry), 1)
    new_components = _geometry_component_count(new_geometry)
    connectivity_term = max(current_components - new_components, 0) / current_components

    fit_gain_term = float(new_quality - current_quality)
    current_holes = _geometry_hole_count(current_geometry)
    new_holes = _geometry_hole_count(new_geometry)
    hole_loss_term = max(current_holes - new_holes, 0) / max(current_holes, 1) if current_holes > 0 else 0.0

    current_hole_overlap = 0.0
    new_hole_overlap = 0.0
    if protected_hole_geometry is not None and not protected_hole_geometry.is_empty:
        current_hole_overlap = float(current_geometry.intersection(protected_hole_geometry).area)
        new_hole_overlap = float(new_geometry.intersection(protected_hole_geometry).area)
    hole_invasion_term = max(new_hole_overlap - current_hole_overlap, 0.0) / current_area

    total_score = (
        float(config.area_gain_weight) * area_gain_term
        - float(config.edge_cost_weight) * edge_cost_term
        + float(config.connectivity_bonus_weight) * connectivity_term
        + float(config.fit_gain_weight) * fit_gain_term
        - float(config.hole_invasion_weight) * hole_invasion_term
        - float(config.hole_loss_weight) * hole_loss_term
    )
    return {
        "total_score": float(total_score),
        "area_gain_term": float(area_gain_term),
        "edge_cost_term": float(edge_cost_term),
        "connectivity_term": float(connectivity_term),
        "fit_gain_term": float(fit_gain_term),
        "hole_invasion_term": float(hole_invasion_term),
        "hole_loss_term": float(hole_loss_term),
        "current_area": float(current_area),
        "new_area": float(new_geometry.area),
        "added_area": float(added_area),
        "current_vertices": float(current_vertices),
        "new_vertices": float(new_vertices),
        "current_components": float(current_components),
        "new_components": float(new_components),
        "current_quality": float(current_quality),
        "new_quality": float(new_quality),
        "current_holes": float(current_holes),
        "new_holes": float(new_holes),
        "current_hole_overlap": float(current_hole_overlap),
        "new_hole_overlap": float(new_hole_overlap),
    }


def build_strip_cover(
    primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
    config: StripCoverConfig | None = None,
) -> Dict[str, object]:
    config = config or StripCoverConfig()
    if not primitives:
        return {
            "primitive_count": 0,
            "triangle_count": 0,
            "quad_count": 0,
            "approx_area": 0.0,
            "approx_iou": 0.0,
            "primitives": [],
        }

    active = [{key: value for key, value in primitive.items()} for primitive in primitives]

    while True:
        polygons = [primitive_polygon(primitive) for primitive in active]
        adjacency = build_primitive_adjacency(polygons)
        best_merge = None
        best_gain = 0.0

        for index_a in range(len(active)):
            for index_b in adjacency[index_a]:
                if index_b <= index_a:
                    continue
                candidate, metrics = _best_convex_candidate(
                    [active[index_a], active[index_b]],
                    reference_geometry=reference_geometry,
                )
                if candidate is None or metrics is None:
                    continue
                precision = float(metrics["precision"])
                support_ratio = float(metrics["support_ratio"])
                if precision < float(config.min_precision) or support_ratio < float(config.min_support_ratio):
                    continue
                baseline_cost = 2.0
                merge_cost = 1.0 + (1.0 - float(metrics["quality"])) * baseline_cost
                gain = baseline_cost - merge_cost
                if gain > best_gain:
                    best_gain = gain
                    best_merge = (index_a, index_b, candidate, metrics)

        if best_merge is None:
            break

        index_a, index_b, candidate, metrics = best_merge
        remaining = [primitive for index, primitive in enumerate(active) if index not in {index_a, index_b}]
        merged_sources = sorted(set(_primitive_source_ids(active[index_a])) | set(_primitive_source_ids(active[index_b])))
        ring = _polygon_ring_vertices(candidate)
        remaining.append(
            {
                "id": max((int(item["id"]) for item in remaining), default=-1) + 1,
                "type": str(metrics["primitive_type"]),
                "vertices": [[float(x), float(y)] for x, y in ring],
                "area": float(candidate.area),
                "cover_precision": float(metrics["precision"]),
                "support_ratio": float(metrics["support_ratio"]),
                "merge_quality": float(metrics["quality"]),
                "covered_primitive_ids": merged_sources,
            }
        )
        active = remaining

    approx_geometry = primitives_union_geometry(active)
    triangle_count = sum(1 for primitive in active if primitive["type"] == "triangle")
    quad_count = sum(1 for primitive in active if primitive["type"] == "quad")
    convex_count = sum(1 for primitive in active if primitive["type"] == "convex")
    return {
        "primitive_count": int(len(active)),
        "triangle_count": int(triangle_count),
        "quad_count": int(quad_count),
        "convex_count": int(convex_count),
        "approx_area": float(approx_geometry.area) if not approx_geometry.is_empty else 0.0,
        "approx_iou": float(geometry_iou(reference_geometry, approx_geometry)) if not approx_geometry.is_empty else 0.0,
        "primitives": active,
    }


def refine_strip_cover(
    primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
    config: StripRefineConfig | None = None,
) -> Dict[str, object]:
    config = config or StripRefineConfig()
    active = [
        {
            key: value
            for key, value in primitive.items()
        }
        for primitive in primitives
    ]
    if not active:
        return {
            "primitive_count": 0,
            "triangle_count": 0,
            "quad_count": 0,
            "approx_area": 0.0,
            "approx_iou": 0.0,
            "primitives": [],
        }

    while True:
        best_merge = None
        best_gain = 0.0

        polygons = [primitive_polygon(primitive) for primitive in active]
        adjacency = build_primitive_adjacency(polygons)
        for index_a in range(len(active)):
            for index_b in adjacency[index_a]:
                if index_b <= index_a:
                    continue
                candidate, metrics = _best_convex_candidate(
                    [active[index_a], active[index_b]],
                    reference_geometry=reference_geometry,
                )
                if candidate is None or metrics is None:
                    continue
                quality = float(metrics["quality"])
                if quality < float(config.min_candidate_quality):
                    continue
                baseline_cost = 2.0
                merge_cost = 1.0 + float(config.merge_error_weight) * (1.0 - quality) * baseline_cost
                gain = baseline_cost - merge_cost
                if gain > best_gain:
                    best_gain = gain
                    best_merge = ((index_a, index_b), candidate, metrics)

        if best_merge is None:
            break

        group, candidate, metrics = best_merge
        remaining = [primitive for index, primitive in enumerate(active) if index not in group]
        merged_sources: List[int] = []
        for index in group:
            merged_sources.extend(_primitive_source_ids(active[index]))
        merged_sources = sorted(set(merged_sources))
        remaining.append(
            {
                "id": max((int(item["id"]) for item in remaining), default=-1) + 1,
                "type": str(metrics["primitive_type"]),
                "vertices": [[float(x), float(y)] for x, y in _polygon_ring_vertices(candidate)],
                "area": float(candidate.area),
                "covered_primitive_ids": merged_sources,
                "cover_precision": float(metrics["precision"]),
                "support_ratio": float(metrics["support_ratio"]),
                "merge_quality": float(metrics["quality"]),
            }
        )
        active = remaining

    approx_geometry = primitives_union_geometry(active)
    triangle_count = sum(1 for primitive in active if primitive["type"] == "triangle")
    quad_count = sum(1 for primitive in active if primitive["type"] == "quad")
    convex_count = sum(1 for primitive in active if primitive["type"] == "convex")
    return {
        "primitive_count": int(len(active)),
        "triangle_count": int(triangle_count),
        "quad_count": int(quad_count),
        "convex_count": int(convex_count),
        "approx_area": float(approx_geometry.area) if not approx_geometry.is_empty else 0.0,
        "approx_iou": float(geometry_iou(reference_geometry, approx_geometry)) if not approx_geometry.is_empty else 0.0,
        "primitives": active,
    }


def build_composite_groups(
    seed_groups: Sequence[Dict[str, object]],
    source_primitives: Sequence[Dict[str, object]],
    atom_primitives: Sequence[Dict[str, object]],
    *,
    reference_geometry: Polygon,
    config: CompositeGroupConfig | None = None,
) -> Dict[str, object]:
    config = config or CompositeGroupConfig()
    source_by_id = {int(primitive["id"]): primitive for primitive in source_primitives}
    atoms_by_id = {int(primitive["id"]): primitive for primitive in atom_primitives}
    protected_hole_geometry = _protected_hole_geometry(reference_geometry)
    groups: List[Dict[str, object]] = []

    for seed in seed_groups:
        source_ids = _primitive_source_ids(seed)
        seed_atom_ids: List[int] = []
        for source_id in source_ids:
            source_primitive = source_by_id.get(int(source_id))
            if source_primitive is None:
                continue
            seed_atom_ids.extend(_primitive_source_ids(source_primitive))
        seed_atom_ids = sorted(set(seed_atom_ids))
        seed_atom_primitives = [atoms_by_id[atom_id] for atom_id in seed_atom_ids if atom_id in atoms_by_id]
        if not seed_atom_primitives:
            continue

        member_ids = list(seed_atom_ids)
        current_geometry = primitives_union_geometry(seed_atom_primitives)
        current_quality = _group_strip_quality(
            seed_atom_primitives,
            reference_geometry=reference_geometry,
            width_scale=1.1,
        )
        patch_history: List[Dict[str, object]] = []

        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_score = float(config.min_score)
            candidate_rows: List[Dict[str, object]] = []

            for atom_id, atom in atoms_by_id.items():
                if atom_id in member_ids:
                    continue
                near_group = _candidate_near_group(
                    seed_primitive=seed,
                    current_geometry=current_geometry,
                    candidate_primitive=atom,
                    config=config,
                )
                if not near_group:
                    continue

                member_primitives = [atoms_by_id[index] for index in member_ids] + [atom]
                new_geometry = primitives_union_geometry(member_primitives)
                new_quality = _group_strip_quality(
                    member_primitives,
                    reference_geometry=reference_geometry,
                    width_scale=1.1,
                )
                if new_quality < current_quality - float(config.max_fit_drop):
                    continue

                score_breakdown = _candidate_patch_score(
                    current_geometry=current_geometry,
                    new_geometry=new_geometry,
                    current_quality=current_quality,
                    new_quality=new_quality,
                    config=config,
                    protected_hole_geometry=protected_hole_geometry,
                )
                candidate_rows.append(
                    {
                        "atom_id": int(atom_id),
                        "atom_type": str(atom["type"]),
                        "atom_area": float(atom["area"]),
                        **score_breakdown,
                    }
                )
                score = float(score_breakdown["total_score"])
                if score > best_score:
                    best_score = score
                    best_candidate = (atom_id, new_geometry, new_quality, score_breakdown)

            candidate_rows = sorted(candidate_rows, key=lambda item: item["total_score"], reverse=True)
            if best_candidate is None:
                if candidate_rows:
                    patch_history.append(
                        {
                            "status": "stop",
                            "current_atom_ids": list(member_ids),
                            "accepted": None,
                            "top_rejected": candidate_rows[:5],
                        }
                    )
                continue

            atom_id, new_geometry, new_quality, accepted_breakdown = best_candidate
            patch_history.append(
                {
                    "status": "accept",
                    "current_atom_ids": list(member_ids),
                    "accepted": {
                        "atom_id": int(atom_id),
                        **accepted_breakdown,
                    },
                    "top_rejected": [row for row in candidate_rows if int(row["atom_id"]) != int(atom_id)][:5],
                }
            )
            member_ids.append(int(atom_id))
            member_ids = sorted(set(member_ids))
            current_geometry = new_geometry
            current_quality = new_quality
            improved = True

        components = _serialize_geometry_components(current_geometry)
        polygons = [component["outer"] for component in components]
        groups.append(
            {
                "id": int(seed["id"]),
                "seed_primitive_id": int(seed["id"]),
                "seed_atom_ids": seed_atom_ids,
                "atom_ids": member_ids,
                "added_atom_ids": [atom_id for atom_id in member_ids if atom_id not in seed_atom_ids],
                "atom_count": int(len(member_ids)),
                "component_count": int(_geometry_component_count(current_geometry)),
                "vertex_count": int(_geometry_vertex_count(current_geometry)),
                "hole_count": int(_geometry_hole_count(current_geometry)),
                "hole_area": float(_geometry_hole_area(current_geometry)),
                "area": float(current_geometry.area),
                "strip_quality": float(current_quality),
                "seed_vertices": seed["vertices"],
                "components": components,
                "polygons": polygons,
                "centroid": [float(value) for value in _geometry_centroid(current_geometry)],
                "patch_history": patch_history,
            }
        )

    return {
        "group_count": int(len(groups)),
        "mean_atom_count": float(np.mean([group["atom_count"] for group in groups])) if groups else 0.0,
        "mean_vertex_count": float(np.mean([group["vertex_count"] for group in groups])) if groups else 0.0,
        "mean_hole_count": float(np.mean([group["hole_count"] for group in groups])) if groups else 0.0,
        "groups": groups,
    }


def _fit_simplified_candidate(geometry: Polygon, *, max_vertices: int) -> Polygon | None:
    if geometry.is_empty:
        return None
    target = orient(geometry, sign=1.0)
    bbox = target.bounds
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    for factor in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2]:
        simplified = orient(target.simplify(scale * factor, preserve_topology=True), sign=1.0)
        if not isinstance(simplified, Polygon):
            continue
        if len(simplified.interiors) > 0:
            continue
        ring = _polygon_ring_vertices(simplified)
        if 3 <= len(ring) <= max_vertices:
            return Polygon(ring)
    return None


def best_low_complexity_fit(geometry: Polygon) -> tuple[Polygon | None, str | None, float]:
    candidates: List[tuple[Polygon, str]] = []
    hull = orient(geometry.convex_hull, sign=1.0)
    for source in [geometry, hull]:
        for max_vertices, primitive_type in [(3, "triangle"), (4, "quad")]:
            candidate = _fit_simplified_candidate(source, max_vertices=max_vertices)
            if candidate is not None:
                ring = _polygon_ring_vertices(candidate)
                if primitive_type == "triangle" and len(ring) != 3:
                    continue
                if primitive_type == "quad" and len(ring) != 4:
                    continue
                candidates.append((orient(candidate, sign=1.0), primitive_type))

    if not geometry.is_empty:
        rectangle = orient(geometry.minimum_rotated_rectangle, sign=1.0)
        ring = _polygon_ring_vertices(rectangle)
        if len(ring) == 4:
            candidates.append((rectangle, "quad"))

    if not candidates:
        return None, None, 0.0

    best_candidate = None
    best_type = None
    best_iou = -1.0
    for candidate, primitive_type in candidates:
        iou = _approx_iou(geometry, candidate)
        if iou > best_iou:
            best_candidate = candidate
            best_type = primitive_type
            best_iou = iou
    return best_candidate, best_type, float(best_iou)


def _shared_length(geometry_a: Polygon, geometry_b: Polygon) -> float:
    return float(geometry_a.boundary.intersection(geometry_b.boundary).length)


def build_primitive_adjacency(polygons: Sequence[Polygon], *, eps: float = 1e-6) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(polygons))}
    for index_a in range(len(polygons)):
        for index_b in range(index_a + 1, len(polygons)):
            if _shared_length(polygons[index_a], polygons[index_b]) <= eps:
                continue
            adjacency[index_a].append(index_b)
            adjacency[index_b].append(index_a)
    return adjacency


def _enumerate_connected_groups(adjacency: Dict[int, List[int]], max_group_size: int) -> List[Tuple[int, ...]]:
    groups = set()
    for start in adjacency:
        frontier = [tuple([start])]
        while frontier:
            group = frontier.pop()
            if len(group) > max_group_size:
                continue
            key = tuple(sorted(group))
            if len(key) >= 2:
                groups.add(key)
            if len(key) == max_group_size:
                continue
            neighbors = set()
            for node in key:
                neighbors.update(adjacency[node])
            for neighbor in sorted(neighbors):
                if neighbor in key:
                    continue
                expanded = tuple(sorted((*key, neighbor)))
                if expanded in groups:
                    continue
                frontier.append(expanded)
    return sorted(groups, key=lambda item: (len(item), item))


def compress_primitives(
    primitives: Sequence[Dict[str, object]],
    *,
    config: PrimitiveCompressionConfig | None = None,
) -> Dict[str, object]:
    config = config or PrimitiveCompressionConfig()
    active = [
        {
            "id": int(primitive["id"]),
            "type": str(primitive["type"]),
            "vertices": primitive["vertices"],
            "area": float(primitive["area"]),
            "source_atom_ids": [int(primitive["id"])],
        }
        for primitive in primitives
    ]

    while True:
        polygons = [primitive_polygon(primitive) for primitive in active]
        adjacency = build_primitive_adjacency(polygons)
        groups = _enumerate_connected_groups(adjacency, max_group_size=config.max_group_size)
        best_merge = None
        best_gain = 0.0

        for group in groups:
            union_geometry = unary_union([polygons[index] for index in group])
            if not isinstance(union_geometry, Polygon):
                continue
            if len(union_geometry.interiors) > 0:
                continue
            candidate, primitive_type, iou = best_low_complexity_fit(union_geometry)
            if candidate is None or primitive_type is None or iou < config.min_iou:
                continue
            baseline_cost = float(len(group))
            merge_cost = 1.0 + float(config.error_weight) * (1.0 - iou) * float(len(group))
            gain = baseline_cost - merge_cost
            if gain > best_gain:
                best_gain = gain
                best_merge = (group, candidate, primitive_type, iou)

        if best_merge is None:
            break

        group, candidate, primitive_type, iou = best_merge
        merged_vertices = _polygon_ring_vertices(candidate)
        remaining = [primitive for index, primitive in enumerate(active) if index not in group]
        remaining.append(
            {
                "id": max((int(item["id"]) for item in remaining), default=-1) + 1,
                "type": primitive_type,
                "vertices": [[float(x), float(y)] for x, y in merged_vertices],
                "area": float(candidate.area),
                "merged_from": [int(active[index]["id"]) for index in group],
                "source_atom_ids": sorted(
                    set(source_atom_id for index in group for source_atom_id in _primitive_source_ids(active[index]))
                ),
                "fit_iou": float(iou),
            }
        )
        active = remaining

    combined = primitives_union_geometry(active)
    triangle_count = sum(1 for primitive in active if primitive["type"] == "triangle")
    quad_count = sum(1 for primitive in active if primitive["type"] == "quad")
    return {
        "primitive_count": int(len(active)),
        "triangle_count": int(triangle_count),
        "quad_count": int(quad_count),
        "approx_area": float(combined.area) if not combined.is_empty else 0.0,
        "primitives": active,
    }


def _queue_triangulate(geometry: Polygon, *, area_epsilon: float = 1e-3) -> List[Polygon]:
    queue: List[Polygon] = [geometry]
    pieces: List[Polygon] = []

    while queue:
        current = queue.pop()
        if current.is_empty or float(current.area) <= area_epsilon:
            continue

        primitive_type = _primitive_type_from_polygon(current)
        if primitive_type is not None and len(current.interiors) == 0:
            pieces.append(current)
            continue

        triangulated = triangulate(current)
        if not triangulated:
            pieces.append(current)
            continue

        progressed = False
        for triangle in triangulated:
            clipped = triangle.intersection(current)
            for polygon in _iter_polygons(clipped):
                if polygon.is_empty or float(polygon.area) <= area_epsilon:
                    continue
                if polygon.equals(current):
                    continue
                queue.append(orient(polygon, sign=1.0))
                progressed = True
        if not progressed:
            pieces.append(current)

    return pieces


def _can_merge_to_quad(polygon_a: Polygon, polygon_b: Polygon, *, area_epsilon: float = 1e-3) -> Polygon | None:
    union = polygon_a.union(polygon_b)
    if not isinstance(union, Polygon):
        return None
    if len(union.interiors) > 0:
        return None
    if abs(float(union.area) - float(polygon_a.area) - float(polygon_b.area)) > area_epsilon:
        return None
    primitive_type = _primitive_type_from_polygon(union)
    if primitive_type != "quad":
        return None
    return orient(union, sign=1.0)


def _shared_boundary_length(polygon_a: Polygon, polygon_b: Polygon) -> float:
    return float(polygon_a.boundary.intersection(polygon_b.boundary).length)


def merge_triangles_to_quads(polygons: Sequence[Polygon], *, area_epsilon: float = 1e-3) -> List[Polygon]:
    active = [orient(polygon, sign=1.0) for polygon in polygons if float(polygon.area) > area_epsilon]
    triangle_indices = [
        index for index, polygon in enumerate(active) if _primitive_type_from_polygon(polygon) == "triangle"
    ]
    candidates: List[Tuple[int, int, Polygon, float]] = []

    for left_offset, index_a in enumerate(triangle_indices):
        polygon_a = active[index_a]
        for index_b in triangle_indices[left_offset + 1 :]:
            polygon_b = active[index_b]
            shared = _shared_boundary_length(polygon_a, polygon_b)
            if shared <= area_epsilon:
                continue
            merged = _can_merge_to_quad(polygon_a, polygon_b, area_epsilon=area_epsilon)
            if merged is None:
                continue
            score = shared + float(merged.area)
            candidates.append((index_a, index_b, merged, score))

    candidates.sort(key=lambda item: (-float(item[3]), int(item[0]), int(item[1])))
    used_indices: set[int] = set()
    merged_pieces: List[Polygon] = []
    for index_a, index_b, merged, _ in candidates:
        if index_a in used_indices or index_b in used_indices:
            continue
        used_indices.add(index_a)
        used_indices.add(index_b)
        merged_pieces.append(merged)

    kept = [polygon for index, polygon in enumerate(active) if index not in used_indices]
    return kept + merged_pieces


def face_geometry(graph_data: Dict[str, object], face_data: Dict[str, object]) -> Polygon:
    vertices = load_vertices(graph_data)
    return orient(face_polygon(face_data, vertices), sign=1.0)


def decompose_face_geometry(
    geometry: Polygon,
    *,
    simplify_tolerance: float = 1.5,
    area_epsilon: float = 1e-3,
) -> Dict[str, object]:
    original = orient(geometry, sign=1.0)
    simplified = orient(original.simplify(simplify_tolerance, preserve_topology=True), sign=1.0)
    if simplified.is_empty:
        simplified = original

    raw_pieces = _queue_triangulate(simplified, area_epsilon=area_epsilon)
    merged_pieces = merge_triangles_to_quads(raw_pieces, area_epsilon=area_epsilon)

    primitive_records: List[PrimitiveRecord] = []
    for primitive_id, polygon in enumerate(sorted(merged_pieces, key=lambda item: (-float(item.area), item.centroid.x, item.centroid.y))):
        primitive_type = _primitive_type_from_polygon(polygon)
        if primitive_type is None:
            extra_pieces = _queue_triangulate(polygon, area_epsilon=area_epsilon)
            for extra in merge_triangles_to_quads(extra_pieces, area_epsilon=area_epsilon):
                extra_type = _primitive_type_from_polygon(extra)
                if extra_type is None:
                    continue
                primitive_records.append(
                    PrimitiveRecord(
                        primitive_id=len(primitive_records),
                        primitive_type=extra_type,
                        vertices=tuple(_polygon_ring_vertices(extra)),
                        area=float(extra.area),
                    )
                )
            continue
        primitive_records.append(
            PrimitiveRecord(
                primitive_id=len(primitive_records),
                primitive_type=primitive_type,
                vertices=tuple(_polygon_ring_vertices(polygon)),
                area=float(polygon.area),
            )
        )

    approx_geometry = None
    for primitive in primitive_records:
        polygon = Polygon(primitive.vertices)
        approx_geometry = polygon if approx_geometry is None else approx_geometry.union(polygon)
    if approx_geometry is None:
        approx_geometry = Polygon()

    tri_count = sum(1 for primitive in primitive_records if primitive.primitive_type == "triangle")
    quad_count = sum(1 for primitive in primitive_records if primitive.primitive_type == "quad")

    return {
        "original_area": float(original.area),
        "simplified_area": float(simplified.area),
        "approx_area": float(approx_geometry.area) if not approx_geometry.is_empty else 0.0,
        "approx_iou": _approx_iou(original, approx_geometry) if not approx_geometry.is_empty else 0.0,
        "primitive_count": int(len(primitive_records)),
        "triangle_count": int(tri_count),
        "quad_count": int(quad_count),
        "primitives": [
            {
                "id": int(primitive.primitive_id),
                "type": primitive.primitive_type,
                "vertices": [[float(x), float(y)] for x, y in primitive.vertices],
                "area": float(primitive.area),
            }
            for primitive in primitive_records
        ],
    }


def decompose_partition_face(
    graph_data: Dict[str, object],
    face_data: Dict[str, object],
    *,
    simplify_tolerance: float = 1.5,
    area_epsilon: float = 1e-3,
) -> Dict[str, object]:
    geometry = face_geometry(graph_data, face_data)
    result = decompose_face_geometry(
        geometry,
        simplify_tolerance=simplify_tolerance,
        area_epsilon=area_epsilon,
    )
    result.update(
        {
            "face_id": int(face_data["id"]),
            "label": int(face_data["label"]),
            "bbox": [int(value) for value in face_data["bbox"]],
            "hole_count": int(len(face_data["holes"])),
        }
    )
    return result
