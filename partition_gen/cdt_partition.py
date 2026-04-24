from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely import simplify
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from partition_gen.dual_graph import face_polygon, load_vertices


Point = Tuple[float, float]


@dataclass(frozen=True)
class CdtSimplifyConfig:
    tolerances: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    min_iou: float = 0.995
    trim_collinear_eps: float = 1e-9


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


def _trim_ring(points: Sequence[Point], *, eps: float) -> List[Point]:
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


def _normalize_polygon(polygon: Polygon, *, eps: float) -> Polygon:
    polygon = orient(polygon, sign=1.0)
    outer = _trim_ring(list(polygon.exterior.coords), eps=eps)
    holes: List[List[Point]] = []
    for interior in polygon.interiors:
        ring = _trim_ring(list(interior.coords), eps=eps)
        if len(ring) >= 3:
            holes.append(ring)
    return orient(Polygon(outer, holes), sign=1.0)


def _polygon_iou(left: Polygon, right: Polygon) -> float:
    union = left.union(right)
    if union.is_empty or union.area <= 0:
        return 0.0
    return float(left.intersection(right).area / union.area)


def _vertex_count(polygon: Polygon, *, eps: float) -> int:
    total = len(_trim_ring(list(orient(polygon, sign=1.0).exterior.coords), eps=eps))
    for interior in orient(polygon, sign=1.0).interiors:
        total += len(_trim_ring(list(interior.coords), eps=eps))
    return int(total)


def face_geometry(graph_data: Dict[str, object], face_data: Dict[str, object]) -> Polygon:
    vertices = load_vertices(graph_data)
    return orient(face_polygon(face_data, vertices), sign=1.0)


def simplify_face_polygon(
    polygon: Polygon,
    *,
    config: CdtSimplifyConfig | None = None,
) -> Dict[str, object]:
    config = config or CdtSimplifyConfig()
    original = _normalize_polygon(polygon, eps=config.trim_collinear_eps)
    original_holes = len(original.interiors)
    original_vertices = _vertex_count(original, eps=config.trim_collinear_eps)

    best = original
    best_iou = 1.0
    best_tolerance = 0.0
    best_vertices = original_vertices

    for tolerance in config.tolerances:
        candidate = simplify(original, tolerance=tolerance, preserve_topology=True)
        if not isinstance(candidate, Polygon) or candidate.is_empty:
            continue
        candidate = _normalize_polygon(candidate, eps=config.trim_collinear_eps)
        if candidate.is_empty or not isinstance(candidate, Polygon):
            continue
        if len(candidate.interiors) != original_holes:
            continue
        iou = _polygon_iou(original, candidate)
        if iou < config.min_iou:
            continue
        candidate_vertices = _vertex_count(candidate, eps=config.trim_collinear_eps)
        if candidate_vertices < best_vertices or (candidate_vertices == best_vertices and iou > best_iou):
            best = candidate
            best_iou = iou
            best_tolerance = float(tolerance)
            best_vertices = candidate_vertices

    return {
        "polygon": best,
        "original_vertex_count": int(original_vertices),
        "simplified_vertex_count": int(best_vertices),
        "hole_count": int(len(best.interiors)),
        "iou": float(best_iou),
        "tolerance": float(best_tolerance),
    }


def polygon_payload(polygon: Polygon, *, eps: float) -> Dict[str, object]:
    polygon = _normalize_polygon(polygon, eps=eps)
    outer = _trim_ring(list(polygon.exterior.coords), eps=eps)
    holes = [_trim_ring(list(interior.coords), eps=eps) for interior in polygon.interiors]
    return {
        "outer": [[float(x), float(y)] for x, y in outer],
        "holes": [[[float(x), float(y)] for x, y in ring] for ring in holes],
    }
