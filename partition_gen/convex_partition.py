from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely import constrained_delaunay_triangles
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from partition_gen.dual_graph import face_polygon, load_json, load_vertices


Point = Tuple[float, float]


@dataclass(frozen=True)
class ConvexMergeConfig:
    convex_rel_eps: float = 1e-6
    convex_abs_eps: float = 1e-8
    shared_edge_eps: float = 1e-6
    area_eps: float = 1e-8
    vertex_round_digits: int = 8


@dataclass
class _ActivePrimitive:
    primitive_id: int
    polygon: Polygon
    source_triangle_ids: Tuple[int, ...]
    neighbors: set[int]
    active: bool = True
    version: int = 0


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


def _trim_ring(points: Sequence[Point], *, eps: float = 1e-9) -> List[Point]:
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


def _polygon_outer_vertices(polygon: Polygon) -> List[Point]:
    return _trim_ring(list(orient(polygon, sign=1.0).exterior.coords))


def _polygon_holes(polygon: Polygon) -> List[List[Point]]:
    holes: List[List[Point]] = []
    for interior in orient(polygon, sign=1.0).interiors:
        ring = _trim_ring(list(interior.coords))
        if len(ring) >= 3:
            holes.append(ring)
    return holes


def _primitive_type(polygon: Polygon) -> str:
    vertex_count = len(_polygon_outer_vertices(polygon))
    if vertex_count == 3:
        return "triangle"
    if vertex_count == 4:
        return "quad"
    return "convex"


def _edge_key(point_a: Point, point_b: Point, *, digits: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    qa = (round(float(point_a[0]), digits), round(float(point_a[1]), digits))
    qb = (round(float(point_b[0]), digits), round(float(point_b[1]), digits))
    return (qa, qb) if qa <= qb else (qb, qa)


def _shared_boundary_length(polygon_a: Polygon, polygon_b: Polygon) -> float:
    return float(polygon_a.boundary.intersection(polygon_b.boundary).length)


def _is_convex_polygon(polygon: Polygon, *, rel_eps: float, abs_eps: float) -> bool:
    if polygon.is_empty or not isinstance(polygon, Polygon):
        return False
    if len(polygon.interiors) > 0:
        return False
    hull = polygon.convex_hull
    return abs(float(hull.area) - float(polygon.area)) <= rel_eps * float(hull.area) + abs_eps


def _single_polygon_union(polygon_a: Polygon, polygon_b: Polygon) -> Polygon | None:
    union = polygon_a.union(polygon_b)
    if not isinstance(union, Polygon):
        return None
    if union.is_empty or len(union.interiors) > 0:
        return None
    return orient(union, sign=1.0)


def face_geometry_from_graph(graph_data: Dict[str, object], face_id: int) -> tuple[Dict[str, object], Polygon]:
    face_data = next(face for face in graph_data["faces"] if int(face["id"]) == int(face_id))
    if str(graph_data.get("format", "")) == "cdt_partition_v1":
        geometry = Polygon(face_data["outer"], face_data.get("holes", []))
        return face_data, orient(geometry, sign=1.0)
    vertices = load_vertices(graph_data)
    return face_data, orient(face_polygon(face_data, vertices), sign=1.0)


def triangulate_face_geometry(
    geometry: Polygon,
    *,
    config: ConvexMergeConfig | None = None,
) -> List[Polygon]:
    config = config or ConvexMergeConfig()
    triangles = constrained_delaunay_triangles(geometry)
    pieces: List[Polygon] = []
    for polygon in _iter_polygons(triangles):
        polygon = orient(polygon, sign=1.0)
        if polygon.area <= config.area_eps:
            continue
        if len(polygon.interiors) > 0:
            continue
        ring = _polygon_outer_vertices(polygon)
        if len(ring) != 3:
            continue
        pieces.append(Polygon(ring))
    return pieces


def _initial_triangle_adjacency(
    triangles: Sequence[Polygon],
    *,
    config: ConvexMergeConfig,
) -> tuple[Dict[int, set[int]], Dict[Tuple[int, int], float]]:
    edge_to_triangles: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List[Tuple[int, float]]] = {}
    for triangle_id, triangle in enumerate(triangles):
        ring = _polygon_outer_vertices(triangle)
        for index, point_a in enumerate(ring):
            point_b = ring[(index + 1) % len(ring)]
            key = _edge_key(point_a, point_b, digits=config.vertex_round_digits)
            length = float(math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))
            edge_to_triangles.setdefault(key, []).append((triangle_id, length))

    adjacency: Dict[int, set[int]] = {triangle_id: set() for triangle_id in range(len(triangles))}
    shared_lengths: Dict[Tuple[int, int], float] = {}
    for items in edge_to_triangles.values():
        if len(items) != 2:
            continue
        (triangle_a, length_a), (triangle_b, length_b) = items
        shared_length = min(length_a, length_b)
        if shared_length <= config.shared_edge_eps:
            continue
        left, right = sorted((triangle_a, triangle_b))
        adjacency[left].add(right)
        adjacency[right].add(left)
        shared_lengths[(left, right)] = shared_length
    return adjacency, shared_lengths


def _candidate_merge(
    left: _ActivePrimitive,
    right: _ActivePrimitive,
    *,
    config: ConvexMergeConfig,
    shared_length: float,
) -> tuple[Polygon, float, float] | None:
    union = _single_polygon_union(left.polygon, right.polygon)
    if union is None:
        return None
    if not _is_convex_polygon(
        union,
        rel_eps=config.convex_rel_eps,
        abs_eps=config.convex_abs_eps,
    ):
        return None
    merged_area = float(union.area)
    return union, float(shared_length), merged_area


def _primitive_payload(primitive: _ActivePrimitive) -> Dict[str, object]:
    polygon = orient(primitive.polygon, sign=1.0)
    outer = _polygon_outer_vertices(polygon)
    holes = _polygon_holes(polygon)
    centroid = polygon.centroid
    return {
        "id": int(primitive.primitive_id),
        "type": _primitive_type(polygon),
        "outer": [[float(x), float(y)] for x, y in outer],
        "holes": [[[float(x), float(y)] for x, y in ring] for ring in holes],
        "vertex_count": int(len(outer)),
        "area": float(polygon.area),
        "centroid": [float(centroid.x), float(centroid.y)],
        "source_triangle_ids": [int(value) for value in primitive.source_triangle_ids],
    }


def greedy_convex_merge(
    geometry: Polygon,
    *,
    config: ConvexMergeConfig | None = None,
) -> Dict[str, object]:
    config = config or ConvexMergeConfig()
    triangles = triangulate_face_geometry(geometry, config=config)
    adjacency, shared_lengths = _initial_triangle_adjacency(triangles, config=config)

    primitives: Dict[int, _ActivePrimitive] = {}
    for triangle_id, triangle in enumerate(triangles):
        primitives[triangle_id] = _ActivePrimitive(
            primitive_id=triangle_id,
            polygon=triangle,
            source_triangle_ids=(triangle_id,),
            neighbors=set(adjacency.get(triangle_id, set())),
        )

    next_id = len(triangles)
    heap: List[Tuple[float, float, int, int, int, int]] = []

    def push_candidate(left_id: int, right_id: int) -> None:
        left_id, right_id = sorted((left_id, right_id))
        left = primitives.get(left_id)
        right = primitives.get(right_id)
        if left is None or right is None or not left.active or not right.active:
            return
        shared_length = _shared_boundary_length(left.polygon, right.polygon)
        if shared_length <= config.shared_edge_eps:
            return
        candidate = _candidate_merge(left, right, config=config, shared_length=shared_length)
        if candidate is None:
            return
        _, candidate_shared, merged_area = candidate
        heapq.heappush(
            heap,
            (-float(candidate_shared), -float(merged_area), left_id, right_id, left.version, right.version),
        )

    for left_id, neighbors in adjacency.items():
        for right_id in neighbors:
            if right_id <= left_id:
                continue
            push_candidate(left_id, right_id)

    merge_history: List[Dict[str, object]] = []
    step = 0
    while heap:
        _, _, left_id, right_id, left_version, right_version = heapq.heappop(heap)
        left = primitives.get(left_id)
        right = primitives.get(right_id)
        if left is None or right is None or not left.active or not right.active:
            continue
        if left.version != left_version or right.version != right_version:
            continue

        shared_length = _shared_boundary_length(left.polygon, right.polygon)
        candidate = _candidate_merge(left, right, config=config, shared_length=shared_length)
        if candidate is None:
            continue
        merged_polygon, candidate_shared, merged_area = candidate

        neighbor_ids = (left.neighbors | right.neighbors) - {left_id, right_id}
        merged_id = next_id
        next_id += 1
        merged_primitive = _ActivePrimitive(
            primitive_id=merged_id,
            polygon=merged_polygon,
            source_triangle_ids=tuple(sorted(left.source_triangle_ids + right.source_triangle_ids)),
            neighbors=set(),
        )

        left.active = False
        right.active = False
        left.version += 1
        right.version += 1

        for neighbor_id in neighbor_ids:
            neighbor = primitives.get(neighbor_id)
            if neighbor is None or not neighbor.active:
                continue
            neighbor.neighbors.discard(left_id)
            neighbor.neighbors.discard(right_id)
            shared_with_merged = _shared_boundary_length(merged_polygon, neighbor.polygon)
            if shared_with_merged <= config.shared_edge_eps:
                continue
            merged_primitive.neighbors.add(neighbor_id)
            neighbor.neighbors.add(merged_id)

        primitives[merged_id] = merged_primitive
        for neighbor_id in list(merged_primitive.neighbors):
            push_candidate(merged_id, neighbor_id)

        merge_history.append(
            {
                "step": int(step),
                "left_id": int(left_id),
                "right_id": int(right_id),
                "merged_id": int(merged_id),
                "shared_edge_length": float(candidate_shared),
                "merged_area": float(merged_area),
                "source_triangle_ids": [int(value) for value in merged_primitive.source_triangle_ids],
                "vertex_count": int(len(_polygon_outer_vertices(merged_polygon))),
            }
        )
        step += 1

    active_primitives = [primitive for primitive in primitives.values() if primitive.active]
    active_primitives.sort(key=lambda item: (-float(item.polygon.area), item.polygon.centroid.x, item.polygon.centroid.y))
    output_primitives = [_primitive_payload(primitive) for primitive in active_primitives]

    merged_geometry = None
    for primitive in active_primitives:
        merged_geometry = primitive.polygon if merged_geometry is None else merged_geometry.union(primitive.polygon)
    if merged_geometry is None:
        merged_geometry = Polygon()
    merged_geometry = orient(merged_geometry, sign=1.0) if isinstance(merged_geometry, Polygon) else merged_geometry

    triangle_payloads = []
    for triangle_id, triangle in enumerate(triangles):
        centroid = triangle.centroid
        triangle_payloads.append(
            {
                "id": int(triangle_id),
                "outer": [[float(x), float(y)] for x, y in _polygon_outer_vertices(triangle)],
                "area": float(triangle.area),
                "centroid": [float(centroid.x), float(centroid.y)],
                "neighbors": sorted(int(value) for value in adjacency.get(triangle_id, set())),
            }
        )

    return {
        "triangle_count": int(len(triangle_payloads)),
        "final_primitive_count": int(len(output_primitives)),
        "final_convex_count": int(len(output_primitives)),
        "original_area": float(geometry.area),
        "covered_area": float(merged_geometry.area) if not merged_geometry.is_empty else 0.0,
        "approx_iou": float(geometry.intersection(merged_geometry).area / geometry.union(merged_geometry).area)
        if not merged_geometry.is_empty and geometry.union(merged_geometry).area > 0
        else 0.0,
        "triangles": triangle_payloads,
        "primitives": output_primitives,
        "merge_history": merge_history,
    }


def build_face_convex_partition(
    partition_graph_path,
    *,
    face_id: int,
    config: ConvexMergeConfig | None = None,
) -> Dict[str, object]:
    graph_data = load_json(partition_graph_path)
    face_data, geometry = face_geometry_from_graph(graph_data, face_id)
    payload = greedy_convex_merge(geometry, config=config)
    payload.update(
        {
            "source_partition_graph": str(partition_graph_path),
            "face_id": int(face_data["id"]),
            "label": int(face_data["label"]),
            "bbox": [int(value) for value in face_data["bbox"]],
            "hole_count": int(len(face_data.get("holes", []))),
            "outer": [[float(x), float(y)] for x, y in _polygon_outer_vertices(geometry)],
            "holes": [[[float(x), float(y)] for x, y in ring] for ring in _polygon_holes(geometry)],
        }
    )
    return payload


def build_convex_partition_from_geometry_payload(
    geometry_payload: Dict[str, object],
    *,
    config: ConvexMergeConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    approx_geometry = geometry_payload.get("approx_geometry") or {}
    geometry = orient(
        Polygon(
            approx_geometry.get("outer", []),
            approx_geometry.get("holes", []),
        ),
        sign=1.0,
    )
    payload = greedy_convex_merge(geometry, config=config)
    payload.update(
        {
            "source_partition_graph": source_tag or geometry_payload.get("source_partition_graph"),
            "face_id": int(geometry_payload["face_id"]),
            "label": int(geometry_payload["label"]),
            "bbox": [int(value) for value in geometry_payload["bbox"]],
            "hole_count": int(len(approx_geometry.get("holes", []))),
            "outer": [[float(x), float(y)] for x, y in _polygon_outer_vertices(geometry)],
            "holes": [[[float(x), float(y)] for x, y in ring] for ring in _polygon_holes(geometry)],
        }
    )
    return payload
