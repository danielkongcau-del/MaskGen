from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


Point = Tuple[float, float]


def trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1e-9:
        ring = ring[:-1]
    return ring


def polygon_from_face(face: Dict[str, object]) -> Polygon:
    geometry = face.get("geometry") or {}
    outer = trim_ring(geometry.get("outer", []))
    holes = [trim_ring(ring) for ring in geometry.get("holes", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        polygon = Polygon(outer, holes)
    except Exception:
        return Polygon()
    fixed = polygon if polygon.is_valid else polygon.buffer(0)
    if isinstance(fixed, Polygon):
        return orient(fixed, sign=1.0)
    polygons = list(iter_polygons(fixed))
    if not polygons:
        return Polygon()
    return orient(max(polygons, key=lambda item: item.area), sign=1.0)


def iter_polygons(geometry) -> Iterable[Polygon]:
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
            yield from iter_polygons(item)


def union_face_polygons(faces: Sequence[Dict[str, object]]):
    polygons = [polygon_from_face(face) for face in faces]
    polygons = [polygon for polygon in polygons if not polygon.is_empty and polygon.area > 0.0]
    if not polygons:
        return Polygon()
    return unary_union(polygons)


def largest_polygon(geometry) -> Polygon:
    polygons = list(iter_polygons(geometry))
    if not polygons:
        return Polygon()
    return orient(max(polygons, key=lambda item: item.area), sign=1.0)


def frame_from_geometry(geometry, *, eps: float = 1e-8) -> Dict[str, object]:
    polygon = largest_polygon(geometry)
    if polygon.is_empty:
        return {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0}
    centroid = polygon.centroid
    rectangle = polygon.minimum_rotated_rectangle
    coords = trim_ring(rectangle.exterior.coords)
    orientation = 0.0
    minx, miny, maxx, maxy = polygon.bounds
    scale = max(maxx - minx, maxy - miny, 1.0)
    if len(coords) >= 2:
        edges = []
        for index in range(len(coords)):
            p = coords[index]
            q = coords[(index + 1) % len(coords)]
            length = math.hypot(q[0] - p[0], q[1] - p[1])
            edges.append((length, p, q))
        length, p, q = max(edges, key=lambda item: item[0])
        if length > eps:
            orientation = math.atan2(q[1] - p[1], q[0] - p[0])
            scale = max(float(length), eps)
    return {
        "origin": [float(centroid.x), float(centroid.y)],
        "scale": float(scale),
        "orientation": float(orientation),
    }


def to_local(point: Sequence[float], frame: Dict[str, object]) -> List[float]:
    cx, cy = [float(value) for value in frame["origin"]]
    scale = max(float(frame["scale"]), 1e-8)
    theta = float(frame["orientation"])
    x = float(point[0]) - cx
    y = float(point[1]) - cy
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    return [float((x * cos_t - y * sin_t) / scale), float((x * sin_t + y * cos_t) / scale)]


def ring_to_local(ring: Sequence[Sequence[float]], frame: Dict[str, object]) -> List[List[float]]:
    return [to_local(point, frame) for point in ring]


def polygon_to_local_payload(geometry, *, eps: float = 1e-8) -> Dict[str, object]:
    frame = frame_from_geometry(geometry, eps=eps)
    polygons = list(iter_polygons(geometry))
    polygons.sort(key=lambda item: -item.area)
    if not polygons:
        return {"frame": frame, "geometry": {"outer_local": [], "holes_local": [], "polygons_local": []}}
    polygons_local = []
    for polygon in polygons:
        polygons_local.append(
            {
                "outer_local": ring_to_local(trim_ring(polygon.exterior.coords), frame),
                "holes_local": [ring_to_local(trim_ring(interior.coords), frame) for interior in polygon.interiors],
            }
        )
    geometry_payload = {
        "outer_local": polygons_local[0]["outer_local"],
        "holes_local": polygons_local[0]["holes_local"],
        "polygons_local": polygons_local,
    }
    return {"frame": frame, "geometry": geometry_payload}


def atom_to_local(atom: Dict[str, object], frame: Dict[str, object]) -> Dict[str, object]:
    return {
        "type": atom.get("type", "convex"),
        "outer_local": ring_to_local(atom.get("outer", []), frame),
        "vertex_count": int(atom.get("vertex_count", len(atom.get("outer", [])))),
        "area": float(atom.get("area", 0.0)),
    }


def vertex_count_for_geometry(geometry) -> int:
    count = 0
    for polygon in iter_polygons(geometry):
        count += len(trim_ring(polygon.exterior.coords))
        count += sum(len(trim_ring(interior.coords)) for interior in polygon.interiors)
    return int(count)


def vertex_count_for_face(face: Dict[str, object]) -> int:
    geometry = face.get("geometry") or {}
    return int(len(geometry.get("outer", [])) + sum(len(ring) for ring in geometry.get("holes", [])))
