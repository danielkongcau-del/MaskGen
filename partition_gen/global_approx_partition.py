from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point as ShapelyPoint, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from partition_gen.dual_graph import face_polygon, load_json, load_vertices
from partition_gen.geometry_approximator import GeometryApproximationConfig, approximate_face_from_partition_graph


Point = Tuple[float, float]
VertexId = int
EdgeKey = Tuple[int, int]
IncidentKey = Tuple[int, ...]


EXTERIOR_FACE_ID = -1


@dataclass(frozen=True)
class GlobalApproxConfig:
    simplify_tolerance: float = 1.0
    simplify_backoff: float = 0.5
    max_simplify_attempts: int = 8
    face_simplify_tolerance: float = 1.5
    face_area_epsilon: float = 1e-3
    area_eps: float = 1e-8
    validity_eps: float = 1e-6


@dataclass(frozen=True)
class _SmallEdge:
    edge_id: int
    vertices: EdgeKey
    faces: IncidentKey
    length: float


def _edge_key(left: int, right: int) -> EdgeKey:
    return (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))


def _incident_key(faces: Sequence[int]) -> IncidentKey:
    values = sorted(int(face_id) for face_id in faces)
    if len(values) == 1:
        return (EXTERIOR_FACE_ID, values[0])
    return tuple(values)


def _point_key(point: Point, digits: int = 8) -> Tuple[float, float]:
    return (round(float(point[0]), digits), round(float(point[1]), digits))


def _points_close(left: Point, right: Point, *, eps: float) -> bool:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])) <= eps


def _trim_ring(points: Sequence[Point], *, eps: float = 1e-9) -> List[Point]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and _points_close(ring[0], ring[-1], eps=eps):
        ring = ring[:-1]
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


def _polygon_vertices(polygon: Polygon) -> List[Point]:
    return _trim_ring(list(orient(polygon, sign=1.0).exterior.coords))


def _polygon_holes(polygon: Polygon) -> List[List[Point]]:
    holes: List[List[Point]] = []
    for interior in orient(polygon, sign=1.0).interiors:
        ring = _trim_ring(list(interior.coords))
        if len(ring) >= 3:
            holes.append(ring)
    return holes


def _line_length(points: Sequence[Point]) -> float:
    return float(
        sum(
            math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
            for index in range(len(points) - 1)
        )
    )


def _polygon_from_payload(payload: Dict[str, object]) -> Polygon:
    geometry = payload.get("approx_geometry") or {}
    polygon = Polygon(geometry.get("outer", []), geometry.get("holes", []))
    return orient(polygon, sign=1.0) if not polygon.is_empty else polygon


def _face_approximation_payloads(
    graph_data: Dict[str, object],
    *,
    config: GlobalApproxConfig,
) -> Dict[int, Dict[str, object]]:
    approx_config = GeometryApproximationConfig(
        simplify_tolerance=float(config.face_simplify_tolerance),
        area_epsilon=float(config.face_area_epsilon),
    )
    payloads: Dict[int, Dict[str, object]] = {}
    for face_data in graph_data.get("faces", []):
        payloads[int(face_data["id"])] = approximate_face_from_partition_graph(
            graph_data,
            face_data,
            config=approx_config,
        )
    return payloads


def _small_edges(graph_data: Dict[str, object]) -> Dict[EdgeKey, _SmallEdge]:
    output: Dict[EdgeKey, _SmallEdge] = {}
    for edge in graph_data.get("edges", []):
        left, right = (int(value) for value in edge["vertices"])
        key = _edge_key(left, right)
        output[key] = _SmallEdge(
            edge_id=int(edge["id"]),
            vertices=key,
            faces=_incident_key(edge.get("faces", [])),
            length=float(edge.get("length", 0.0)),
        )
    return output


def _ring_small_edge_keys(ring: Sequence[int]) -> List[Tuple[int, int, EdgeKey]]:
    output: List[Tuple[int, int, EdgeKey]] = []
    for index, start in enumerate(ring):
        end = int(ring[(index + 1) % len(ring)])
        start = int(start)
        output.append((start, end, _edge_key(start, end)))
    return output


def _build_incident_graphs(edges: Dict[EdgeKey, _SmallEdge]) -> Dict[IncidentKey, Dict[int, List[Tuple[int, EdgeKey]]]]:
    graphs: Dict[IncidentKey, Dict[int, List[Tuple[int, EdgeKey]]]] = {}
    for key, edge in edges.items():
        left, right = key
        graph = graphs.setdefault(edge.faces, {})
        graph.setdefault(left, []).append((right, key))
        graph.setdefault(right, []).append((left, key))
    for graph in graphs.values():
        for items in graph.values():
            items.sort(key=lambda item: (item[0], item[1]))
    return graphs


def _walk_chain(
    graph: Dict[int, List[Tuple[int, EdgeKey]]],
    start: int,
    first_neighbor: int,
    first_key: EdgeKey,
    visited: set[EdgeKey],
) -> Tuple[List[int], List[EdgeKey]]:
    vertices = [int(start), int(first_neighbor)]
    edge_keys = [first_key]
    visited.add(first_key)

    previous = int(start)
    current = int(first_neighbor)
    while len(graph.get(current, [])) == 2:
        next_items = [(neighbor, key) for neighbor, key in graph[current] if key not in visited]
        if not next_items:
            break
        neighbor, key = next_items[0]
        if neighbor == previous and len(next_items) > 1:
            neighbor, key = next_items[1]
        visited.add(key)
        vertices.append(int(neighbor))
        edge_keys.append(key)
        previous, current = current, int(neighbor)
        if current == start:
            break
    return vertices, edge_keys


def _extract_maximal_chains(
    graph_data: Dict[str, object],
) -> List[Dict[str, object]]:
    edges = _small_edges(graph_data)
    incident_graphs = _build_incident_graphs(edges)
    vertices = load_vertices(graph_data)
    chains: List[Dict[str, object]] = []

    for incident_faces, graph in sorted(incident_graphs.items(), key=lambda item: item[0]):
        visited: set[EdgeKey] = set()
        endpoint_vertices = [vertex for vertex, neighbors in graph.items() if len(neighbors) != 2]

        for start in sorted(endpoint_vertices):
            for neighbor, edge_key in list(graph[start]):
                if edge_key in visited:
                    continue
                chain_vertices, chain_edge_keys = _walk_chain(graph, start, neighbor, edge_key, visited)
                chains.append(
                    {
                        "incident_faces": incident_faces,
                        "source_vertex_ids": chain_vertices,
                        "source_edge_keys": chain_edge_keys,
                    }
                )

        remaining = [edge_key for items in graph.values() for _, edge_key in items if edge_key not in visited]
        for edge_key in sorted(set(remaining)):
            if edge_key in visited:
                continue
            start, first_neighbor = edge_key
            chain_vertices, chain_edge_keys = _walk_chain(graph, start, first_neighbor, edge_key, visited)
            chains.append(
                {
                    "incident_faces": incident_faces,
                    "source_vertex_ids": chain_vertices,
                    "source_edge_keys": chain_edge_keys,
                }
            )

    for chain_id, chain in enumerate(chains):
        points = [(float(vertices[vertex_id][0]), float(vertices[vertex_id][1])) for vertex_id in chain["source_vertex_ids"]]
        source_edge_ids = [int(edges[key].edge_id) for key in chain["source_edge_keys"]]
        chain.update(
            {
                "id": int(chain_id),
                "source_edge_ids": source_edge_ids,
                "original_points": points,
                "points": points,
                "original_vertex_count": int(len(points)),
                "vertex_count": int(len(points)),
                "original_length": _line_length(points),
                "length": _line_length(points),
            }
        )
    return chains


def _simplify_chain_points(points: Sequence[Point], *, tolerance: float, eps: float) -> List[Point]:
    if tolerance <= 0.0 or len(points) <= 2:
        return [(float(x), float(y)) for x, y in points]
    is_closed = len(points) >= 2 and _points_close(points[0], points[-1], eps=eps)
    simplified = list(LineString(points).simplify(tolerance, preserve_topology=False).coords)
    simplified = [(float(x), float(y)) for x, y in simplified]
    if not simplified:
        return [(float(x), float(y)) for x, y in points]

    simplified[0] = (float(points[0][0]), float(points[0][1]))
    simplified[-1] = (float(points[-1][0]), float(points[-1][1]))
    if is_closed and not _points_close(simplified[0], simplified[-1], eps=eps):
        simplified.append(simplified[0])
    deduped: List[Point] = []
    for point in simplified:
        if deduped and _points_close(deduped[-1], point, eps=eps):
            continue
        deduped.append(point)
    if len(deduped) < 2:
        return [(float(x), float(y)) for x, y in points]
    return deduped


def _ring_cumulative_distances(points: Sequence[Point]) -> List[float]:
    distances = [0.0]
    for index in range(len(points) - 1):
        distances.append(distances[-1] + math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1]))
    return distances


def _interpolate_on_polyline(points: Sequence[Point], distances: Sequence[float], distance: float) -> Point:
    if distance <= 0.0:
        return (float(points[0][0]), float(points[0][1]))
    if distance >= distances[-1]:
        return (float(points[-1][0]), float(points[-1][1]))
    for index in range(len(distances) - 1):
        left = distances[index]
        right = distances[index + 1]
        if left <= distance <= right:
            span = max(right - left, 1e-12)
            t = (distance - left) / span
            return (
                float(points[index][0] + t * (points[index + 1][0] - points[index][0])),
                float(points[index][1] + t * (points[index + 1][1] - points[index][1])),
            )
    return (float(points[-1][0]), float(points[-1][1]))


def _extract_open_subline(points: Sequence[Point], start_distance: float, end_distance: float) -> List[Point]:
    distances = _ring_cumulative_distances(points)
    start_distance = max(0.0, min(float(start_distance), distances[-1]))
    end_distance = max(0.0, min(float(end_distance), distances[-1]))
    if end_distance < start_distance:
        start_distance, end_distance = end_distance, start_distance

    output = [_interpolate_on_polyline(points, distances, start_distance)]
    for index in range(1, len(points) - 1):
        if start_distance < distances[index] < end_distance:
            output.append((float(points[index][0]), float(points[index][1])))
    output.append(_interpolate_on_polyline(points, distances, end_distance))
    return output


def _extract_closed_ring_path(ring: Sequence[Point], start_distance: float, end_distance: float) -> List[Point]:
    closed = list(ring) + [ring[0]]
    distances = _ring_cumulative_distances(closed)
    total = distances[-1]
    start_distance = float(start_distance) % total
    end_distance = float(end_distance) % total
    if end_distance >= start_distance:
        return _extract_open_subline(closed, start_distance, end_distance)
    first = _extract_open_subline(closed, start_distance, total)
    second = _extract_open_subline(closed, 0.0, end_distance)
    return first[:-1] + second


def _boundary_rings(polygon: Polygon) -> List[List[Point]]:
    rings = [_polygon_vertices(polygon)]
    rings.extend(_polygon_holes(polygon))
    return [ring for ring in rings if len(ring) >= 3]


def _candidate_distance(raw_points: Sequence[Point], candidate_points: Sequence[Point]) -> float:
    if len(raw_points) < 2 or len(candidate_points) < 2:
        return float("inf")
    raw_line = LineString(raw_points)
    candidate_line = LineString(candidate_points)
    return float(raw_line.hausdorff_distance(candidate_line) + 0.25 * candidate_line.hausdorff_distance(raw_line))


def _owner_segment_from_approx_polygon(
    approx_polygon: Polygon,
    raw_points: Sequence[Point],
    *,
    eps: float,
) -> Tuple[List[Point], float] | None:
    if approx_polygon.is_empty or len(raw_points) < 2:
        return None

    raw_points = [(float(x), float(y)) for x, y in raw_points]
    raw_closed = _points_close(raw_points[0], raw_points[-1], eps=eps)
    best: Tuple[float, List[Point]] | None = None

    for ring in _boundary_rings(approx_polygon):
        if raw_closed:
            closed = list(ring) + [ring[0]]
            score = _candidate_distance(raw_points, closed)
            if best is None or score < best[0]:
                best = (score, closed)
            continue

        closed_ring = list(ring) + [ring[0]]
        ring_line = LineString(closed_ring)
        total = float(ring_line.length)
        if total <= eps:
            continue
        start_distance = float(ring_line.project(ShapelyPoint(raw_points[0])))
        end_distance = float(ring_line.project(ShapelyPoint(raw_points[-1])))

        forward = _extract_closed_ring_path(ring, start_distance, end_distance)
        backward = list(reversed(_extract_closed_ring_path(ring, end_distance, start_distance)))
        for candidate in (forward, backward):
            if len(candidate) < 2:
                continue
            candidate = [raw_points[0], *candidate[1:-1], raw_points[-1]]
            deduped: List[Point] = []
            for point in candidate:
                if deduped and _points_close(deduped[-1], point, eps=eps):
                    continue
                deduped.append(point)
            if len(deduped) < 2:
                continue
            score = _candidate_distance(raw_points, deduped)
            if best is None or score < best[0]:
                best = (score, deduped)

    if best is None:
        return None
    return best[1], float(best[0])


def _choose_owner_face(
    incident_faces: Sequence[int],
    face_payloads: Dict[int, Dict[str, object]],
) -> int | None:
    faces = [int(face_id) for face_id in incident_faces if int(face_id) != EXTERIOR_FACE_ID]
    if len(faces) != 2:
        return None
    return sorted(
        faces,
        key=lambda face_id: (
            int(face_payloads[face_id].get("approx_vertex_count", 10**9)),
            float(face_payloads[face_id].get("approx_area", 10**18)),
            int(face_id),
        ),
    )[0]


def _base_arcs_from_chains(chains: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    arcs: List[Dict[str, object]] = []
    for chain in chains:
        points = [(float(x), float(y)) for x, y in chain["original_points"]]
        arcs.append(
            {
                "id": int(chain["id"]),
                "incident_faces": [int(value) for value in chain["incident_faces"]],
                "source_vertex_ids": [int(value) for value in chain["source_vertex_ids"]],
                "source_edge_ids": [int(value) for value in chain["source_edge_ids"]],
                "original_points": [[float(x), float(y)] for x, y in points],
                "points": [[float(x), float(y)] for x, y in points],
                "original_vertex_count": int(chain["original_vertex_count"]),
                "vertex_count": int(len(points)),
                "original_length": float(chain["original_length"]),
                "length": float(_line_length(points)),
                "method": "raw_exact",
                "owner_face_id": None,
                "simplified": False,
                "accepted": True,
            }
        )
    return arcs


def _owner_arc_candidates(
    chains: Sequence[Dict[str, object]],
    face_payloads: Dict[int, Dict[str, object]],
    *,
    config: GlobalApproxConfig,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for chain in chains:
        owner_face_id = _choose_owner_face(chain["incident_faces"], face_payloads)
        if owner_face_id is None:
            continue
        approx_polygon = _polygon_from_payload(face_payloads[owner_face_id])
        result = _owner_segment_from_approx_polygon(
            approx_polygon,
            chain["original_points"],
            eps=config.validity_eps,
        )
        if result is None:
            continue
        candidate_points, distance = result
        if len(candidate_points) < 2:
            continue
        original_vertex_count = int(chain["original_vertex_count"])
        candidate_vertex_count = int(len(candidate_points))
        benefit = original_vertex_count - candidate_vertex_count
        if benefit <= 0:
            continue
        candidates.append(
            {
                "arc_id": int(chain["id"]),
                "owner_face_id": int(owner_face_id),
                "points": [[float(x), float(y)] for x, y in candidate_points],
                "vertex_count": candidate_vertex_count,
                "length": float(_line_length(candidate_points)),
                "distance": float(distance),
                "vertex_reduction": int(benefit),
                "method": "owner_approx_boundary_segment",
            }
        )
    candidates.sort(key=lambda item: (-int(item["vertex_reduction"]), float(item["distance"]), int(item["arc_id"])))
    return candidates


def _build_directed_edge_to_arc(arcs: Sequence[Dict[str, object]]) -> Dict[Tuple[int, int], Tuple[int, bool]]:
    mapping: Dict[Tuple[int, int], Tuple[int, bool]] = {}
    for arc in arcs:
        arc_id = int(arc["id"])
        vertex_ids = [int(value) for value in arc["source_vertex_ids"]]
        for index in range(len(vertex_ids) - 1):
            left = vertex_ids[index]
            right = vertex_ids[index + 1]
            mapping[(left, right)] = (arc_id, False)
            mapping[(right, left)] = (arc_id, True)
    return mapping


def _collapse_ring_to_arc_refs(
    ring: Sequence[int],
    directed_edge_to_arc: Dict[Tuple[int, int], Tuple[int, bool]],
) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    for start, end, _ in _ring_small_edge_keys(ring):
        arc_id, reversed_ref = directed_edge_to_arc[(start, end)]
        if refs and int(refs[-1]["arc_id"]) == arc_id and bool(refs[-1]["reversed"]) == reversed_ref:
            continue
        refs.append({"arc_id": int(arc_id), "reversed": bool(reversed_ref)})
    if len(refs) > 1 and refs[0] == refs[-1]:
        refs.pop()
    return refs


def _points_from_arc_refs(arc_refs: Sequence[Dict[str, object]], arcs_by_id: Dict[int, Dict[str, object]]) -> List[Point]:
    ring: List[Point] = []
    for ref in arc_refs:
        points = [(float(x), float(y)) for x, y in arcs_by_id[int(ref["arc_id"])]["points"]]
        if bool(ref.get("reversed", False)):
            points = list(reversed(points))
        for point in points:
            if ring and _points_close(ring[-1], point, eps=1e-9):
                continue
            ring.append(point)
    if len(ring) > 1 and _points_close(ring[0], ring[-1], eps=1e-9):
        ring.pop()
    return ring


def _face_payloads_from_arcs(
    graph_data: Dict[str, object],
    arcs: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    directed_edge_to_arc = _build_directed_edge_to_arc(arcs)
    arcs_by_id = {int(arc["id"]): arc for arc in arcs}
    faces: List[Dict[str, object]] = []

    for face_data in graph_data.get("faces", []):
        faces.append(_face_payload_from_arcs(face_data, directed_edge_to_arc, arcs_by_id))
    return faces


def _face_payload_from_arcs(
    face_data: Dict[str, object],
    directed_edge_to_arc: Dict[Tuple[int, int], Tuple[int, bool]],
    arcs_by_id: Dict[int, Dict[str, object]],
) -> Dict[str, object]:
    outer_arc_refs = _collapse_ring_to_arc_refs(face_data["outer"], directed_edge_to_arc)
    outer = _points_from_arc_refs(outer_arc_refs, arcs_by_id)
    holes = []
    hole_refs = []
    for hole in face_data.get("holes", []):
        refs = _collapse_ring_to_arc_refs(hole, directed_edge_to_arc)
        ring = _points_from_arc_refs(refs, arcs_by_id)
        if len(ring) >= 3:
            holes.append(ring)
            hole_refs.append(refs)
    polygon = Polygon()
    is_valid = False
    if len(outer) >= 3:
        try:
            polygon = orient(Polygon(outer, holes), sign=1.0)
            is_valid = bool(polygon.is_valid and not polygon.is_empty)
        except ValueError:
            polygon = Polygon()
            is_valid = False
    centroid = polygon.centroid if not polygon.is_empty else None
    return {
        "id": int(face_data["id"]),
        "label": int(face_data["label"]),
        "area": int(face_data.get("area", 0)),
        "bbox": [int(value) for value in face_data.get("bbox", [])],
        "outer_arc_refs": outer_arc_refs,
        "hole_arc_refs": hole_refs,
        "outer": [[float(x), float(y)] for x, y in (_polygon_vertices(polygon) if is_valid else outer)],
        "holes": [[[float(x), float(y)] for x, y in ring] for ring in (_polygon_holes(polygon) if is_valid else holes)],
        "approx_area": float(polygon.area),
        "centroid": [float(centroid.x), float(centroid.y)] if centroid is not None else [0.0, 0.0],
        "is_valid": bool(is_valid),
    }


def _face_polygons_from_payload(faces: Sequence[Dict[str, object]]) -> Dict[int, Polygon]:
    output: Dict[int, Polygon] = {}
    for face in faces:
        if len(face.get("outer", [])) < 3:
            output[int(face["id"])] = Polygon()
            continue
        try:
            output[int(face["id"])] = orient(Polygon(face["outer"], face.get("holes", [])), sign=1.0)
        except ValueError:
            output[int(face["id"])] = Polygon()
    return output


def _safe_polygon_for_union(polygon: Polygon) -> Polygon:
    if polygon.is_empty:
        return Polygon()
    if polygon.is_valid:
        return polygon
    fixed = polygon.buffer(0)
    fixed_polygons = [item for item in _iter_polygons(fixed) if item.area > 0.0]
    if not fixed_polygons:
        return Polygon()
    merged = unary_union(fixed_polygons)
    if isinstance(merged, Polygon):
        return merged
    merged_polygons = [item for item in _iter_polygons(merged) if item.area > 0.0]
    return unary_union(merged_polygons) if merged_polygons else Polygon()


def _safe_unary_union(polygons: Sequence[Polygon]) -> Polygon:
    safe_polygons = [_safe_polygon_for_union(polygon) for polygon in polygons]
    safe_polygons = [polygon for polygon in safe_polygons if not polygon.is_empty]
    if not safe_polygons:
        return Polygon()
    try:
        return unary_union(safe_polygons)
    except Exception:
        buffered = [polygon.buffer(0) for polygon in safe_polygons]
        buffered = [polygon for polygon in buffered if not polygon.is_empty]
        return unary_union(buffered) if buffered else Polygon()


def _original_face_polygons(graph_data: Dict[str, object]) -> Dict[int, Polygon]:
    vertices = load_vertices(graph_data)
    output: Dict[int, Polygon] = {}
    for face in graph_data.get("faces", []):
        output[int(face["id"])] = orient(face_polygon(face, vertices), sign=1.0)
    return output


def _adjacency_pairs_from_graph(graph_data: Dict[str, object]) -> set[Tuple[int, int]]:
    pairs = set()
    for item in graph_data.get("adjacency", []):
        left, right = sorted(int(value) for value in item["faces"])
        pairs.add((left, right))
    return pairs


def _adjacency_pairs_from_arcs(arcs: Sequence[Dict[str, object]]) -> set[Tuple[int, int]]:
    pairs = set()
    for arc in arcs:
        faces = [int(value) for value in arc["incident_faces"] if int(value) != EXTERIOR_FACE_ID]
        if len(faces) == 2:
            left, right = sorted(faces)
            pairs.add((left, right))
    return pairs


def _junction_count(arcs: Sequence[Dict[str, object]]) -> int:
    endpoint_counts: Dict[Tuple[float, float], int] = {}
    for arc in arcs:
        points = [(float(x), float(y)) for x, y in arc["points"]]
        if len(points) < 2:
            continue
        for endpoint in (points[0], points[-1]):
            key = _point_key(endpoint)
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
    return sum(1 for count in endpoint_counts.values() if count != 2)


def validate_global_approx_partition(
    graph_data: Dict[str, object],
    arcs: Sequence[Dict[str, object]],
    faces: Sequence[Dict[str, object]],
    *,
    config: GlobalApproxConfig | None = None,
) -> Dict[str, object]:
    config = config or GlobalApproxConfig()
    polygons = _face_polygons_from_payload(faces)
    original_polygons = _original_face_polygons(graph_data)
    valid_faces = all(polygon.is_valid and not polygon.is_empty for polygon in polygons.values())

    approx_union = _safe_unary_union(list(polygons.values())) if polygons else Polygon()
    original_union = _safe_unary_union(list(original_polygons.values())) if original_polygons else Polygon()
    approx_union_area = float(approx_union.area) if not approx_union.is_empty else 0.0
    original_union_area = float(original_union.area) if not original_union.is_empty else 0.0
    total_face_area = float(sum(polygon.area for polygon in polygons.values()))
    overlap_area = max(0.0, total_face_area - approx_union_area)
    union_area = float(approx_union.union(original_union).area)
    iou = float(approx_union.intersection(original_union).area / union_area) if union_area > 0 else 0.0

    original_adjacency = _adjacency_pairs_from_graph(graph_data)
    approx_adjacency = _adjacency_pairs_from_arcs(arcs)
    missing_adjacency = sorted(list(original_adjacency - approx_adjacency))
    extra_adjacency = sorted(list(approx_adjacency - original_adjacency))
    all_rings_closed = all(len(face.get("outer", [])) >= 3 for face in faces)
    shared_arc_count = sum(
        1
        for arc in arcs
        if len([face_id for face_id in arc["incident_faces"] if int(face_id) != EXTERIOR_FACE_ID]) == 2
    )

    is_valid = bool(
        valid_faces
        and all_rings_closed
        and overlap_area <= config.validity_eps
        and iou >= 1.0 - config.validity_eps
        and not missing_adjacency
        and not extra_adjacency
    )

    return {
        "is_valid": is_valid,
        "all_faces_valid": bool(valid_faces),
        "all_face_rings_closed": bool(all_rings_closed),
        "original_union_area": original_union_area,
        "approx_union_area": approx_union_area,
        "total_face_area": total_face_area,
        "overlap_area": float(overlap_area),
        "union_iou": float(iou),
        "original_adjacency_count": int(len(original_adjacency)),
        "approx_adjacency_count": int(len(approx_adjacency)),
        "missing_adjacency": [[int(a), int(b)] for a, b in missing_adjacency],
        "extra_adjacency": [[int(a), int(b)] for a, b in extra_adjacency],
        "arc_count": int(len(arcs)),
        "shared_arc_count": int(shared_arc_count),
        "junction_count": int(_junction_count(arcs)),
        "face_count": int(len(faces)),
    }


def _with_simplified_arcs(
    chains: Sequence[Dict[str, object]],
    *,
    tolerance: float,
    config: GlobalApproxConfig,
) -> List[Dict[str, object]]:
    arcs: List[Dict[str, object]] = []
    for chain in chains:
        points = _simplify_chain_points(
            chain["original_points"],
            tolerance=tolerance,
            eps=config.validity_eps,
        )
        arc = dict(chain)
        arc["incident_faces"] = [int(value) for value in chain["incident_faces"]]
        arc["source_vertex_ids"] = [int(value) for value in chain["source_vertex_ids"]]
        arc["source_edge_ids"] = [int(value) for value in chain["source_edge_ids"]]
        arc["original_points"] = [[float(x), float(y)] for x, y in chain["original_points"]]
        arc["points"] = [[float(x), float(y)] for x, y in points]
        arc["vertex_count"] = int(len(points))
        arc["length"] = float(_line_length(points))
        arc["simplified"] = bool(len(points) < int(chain["original_vertex_count"]))
        arcs.append(arc)
    return arcs


def _candidate_applied_arcs(arcs: Sequence[Dict[str, object]], candidate: Dict[str, object]) -> List[Dict[str, object]]:
    next_arcs = [dict(arc) for arc in arcs]
    arc_id = int(candidate["arc_id"])
    next_arcs[arc_id] = _candidate_applied_arc(next_arcs[arc_id], candidate)
    return next_arcs


def _candidate_applied_arc(arc: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    output = dict(arc)
    points = [(float(x), float(y)) for x, y in candidate["points"]]
    output.update(
        {
            "points": [[float(x), float(y)] for x, y in points],
            "vertex_count": int(len(points)),
            "length": float(_line_length(points)),
            "method": str(candidate["method"]),
            "owner_face_id": int(candidate["owner_face_id"]),
            "owner_distance": float(candidate["distance"]),
            "vertex_reduction": int(candidate["vertex_reduction"]),
            "simplified": True,
            "accepted": True,
        }
    )
    return output


def _affected_face_ids_for_arc(arc: Dict[str, object]) -> List[int]:
    return [int(face_id) for face_id in arc["incident_faces"] if int(face_id) != EXTERIOR_FACE_ID]


def _local_candidate_validation(
    face_data_by_id: Dict[int, Dict[str, object]],
    directed_edge_to_arc: Dict[Tuple[int, int], Tuple[int, bool]],
    arcs_by_id: Dict[int, Dict[str, object]],
    current_polygons_by_id: Dict[int, Polygon],
    affected_face_ids: Sequence[int],
    *,
    config: GlobalApproxConfig,
) -> Tuple[bool, Dict[int, Dict[str, object]], Dict[int, Polygon], Dict[str, object]]:
    new_faces_by_id: Dict[int, Dict[str, object]] = {}
    for face_id in affected_face_ids:
        face_data = face_data_by_id[int(face_id)]
        new_faces_by_id[int(face_id)] = _face_payload_from_arcs(face_data, directed_edge_to_arc, arcs_by_id)

    new_polygons_by_id = _face_polygons_from_payload(list(new_faces_by_id.values()))
    all_faces_valid = all(
        new_faces_by_id[face_id].get("is_valid", False)
        and new_polygons_by_id[face_id].is_valid
        and not new_polygons_by_id[face_id].is_empty
        for face_id in new_faces_by_id
    )
    if not all_faces_valid:
        return False, new_faces_by_id, new_polygons_by_id, {
            "reason": "local affected face invalid",
            "local_union_iou": 0.0,
            "local_overlap_area": 0.0,
        }

    previous_polygons = [current_polygons_by_id[int(face_id)] for face_id in affected_face_ids]
    next_polygons = [new_polygons_by_id[int(face_id)] for face_id in affected_face_ids]
    previous_union = _safe_unary_union(previous_polygons)
    next_union = _safe_unary_union(next_polygons)
    next_union_area = float(next_union.area) if not next_union.is_empty else 0.0
    total_next_area = float(sum(polygon.area for polygon in next_polygons))
    overlap_area = max(0.0, total_next_area - next_union_area)

    try:
        union_area = float(previous_union.union(next_union).area)
        local_iou = float(previous_union.intersection(next_union).area / union_area) if union_area > 0 else 0.0
    except Exception:
        return False, new_faces_by_id, new_polygons_by_id, {
            "reason": "local union operation failed",
            "local_union_iou": 0.0,
            "local_overlap_area": float(overlap_area),
        }

    is_valid = bool(overlap_area <= config.validity_eps and local_iou >= 1.0 - config.validity_eps)
    return is_valid, new_faces_by_id, new_polygons_by_id, {
        "reason": "local validation failed" if not is_valid else None,
        "local_union_iou": float(local_iou),
        "local_overlap_area": float(overlap_area),
    }


def _build_faces_and_validation(
    graph_data: Dict[str, object],
    arcs: Sequence[Dict[str, object]],
    *,
    config: GlobalApproxConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    faces = _face_payloads_from_arcs(graph_data, arcs)
    validation = validate_global_approx_partition(graph_data, arcs, faces, config=config)
    return faces, validation


def _reconcile_owner_approx_arcs_full_validation(
    graph_data: Dict[str, object],
    chains: Sequence[Dict[str, object]],
    face_payloads: Dict[int, Dict[str, object]],
    *,
    config: GlobalApproxConfig,
    candidates: Sequence[Dict[str, object]] | None = None,
    fallback_reason: str | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    arcs = _base_arcs_from_chains(chains)
    faces, validation = _build_faces_and_validation(graph_data, arcs, config=config)
    if not validation["is_valid"]:
        return arcs, faces, validation, {
            "policy": "greedy_owner_boundary_transfer_v1",
            "candidate_validation": "full_global",
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "fallback_reason": fallback_reason,
            "rejections": [{"reason": "raw exact arc map is invalid"}],
        }

    if candidates is None:
        candidates = _owner_arc_candidates(chains, face_payloads, config=config)
    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    for candidate in candidates:
        next_arcs = _candidate_applied_arcs(arcs, candidate)
        next_faces, next_validation = _build_faces_and_validation(graph_data, next_arcs, config=config)
        if next_validation["is_valid"]:
            arcs = next_arcs
            faces = next_faces
            validation = next_validation
            accepted.append(
                {
                    "arc_id": int(candidate["arc_id"]),
                    "owner_face_id": int(candidate["owner_face_id"]),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                }
            )
        else:
            rejected.append(
                {
                    "arc_id": int(candidate["arc_id"]),
                    "owner_face_id": int(candidate["owner_face_id"]),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                    "reason": "global validation failed",
                    "union_iou": float(next_validation["union_iou"]),
                    "overlap_area": float(next_validation["overlap_area"]),
                    "missing_adjacency": next_validation["missing_adjacency"],
                    "extra_adjacency": next_validation["extra_adjacency"],
                }
            )

    reconciliation = {
        "policy": "greedy_owner_boundary_transfer_v1",
        "candidate_validation": "full_global",
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "accepted": accepted,
        "rejected": rejected[:64],
    }
    if fallback_reason:
        reconciliation["fallback_reason"] = fallback_reason
    return arcs, faces, validation, reconciliation


def _reconcile_owner_approx_arcs(
    graph_data: Dict[str, object],
    chains: Sequence[Dict[str, object]],
    face_payloads: Dict[int, Dict[str, object]],
    *,
    config: GlobalApproxConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    arcs = _base_arcs_from_chains(chains)
    faces, validation = _build_faces_and_validation(graph_data, arcs, config=config)
    if not validation["is_valid"]:
        return arcs, faces, validation, {
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejections": [{"reason": "raw exact arc map is invalid"}],
        }

    candidates = _owner_arc_candidates(chains, face_payloads, config=config)
    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    face_data_by_id = {int(face_data["id"]): face_data for face_data in graph_data.get("faces", [])}
    face_order = [int(face_data["id"]) for face_data in graph_data.get("faces", [])]
    directed_edge_to_arc = _build_directed_edge_to_arc(arcs)
    arcs_by_id = {int(arc["id"]): arc for arc in arcs}
    faces_by_id = {int(face["id"]): face for face in faces}
    polygons_by_id = _face_polygons_from_payload(faces)

    for candidate in candidates:
        arc_id = int(candidate["arc_id"])
        old_arc = arcs_by_id[arc_id]
        new_arc = _candidate_applied_arc(old_arc, candidate)
        affected_face_ids = _affected_face_ids_for_arc(old_arc)
        arcs_by_id[arc_id] = new_arc
        locally_valid, new_faces_by_id, new_polygons_by_id, local_validation = _local_candidate_validation(
            face_data_by_id,
            directed_edge_to_arc,
            arcs_by_id,
            polygons_by_id,
            affected_face_ids,
            config=config,
        )
        if locally_valid:
            arcs[arc_id] = new_arc
            for face_id, face in new_faces_by_id.items():
                faces_by_id[int(face_id)] = face
            for face_id, polygon in new_polygons_by_id.items():
                polygons_by_id[int(face_id)] = polygon
            accepted.append(
                {
                    "arc_id": int(candidate["arc_id"]),
                    "owner_face_id": int(candidate["owner_face_id"]),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                }
            )
        else:
            arcs_by_id[arc_id] = old_arc
            rejected.append(
                {
                    "arc_id": int(candidate["arc_id"]),
                    "owner_face_id": int(candidate["owner_face_id"]),
                    "vertex_reduction": int(candidate["vertex_reduction"]),
                    "distance": float(candidate["distance"]),
                    "reason": str(local_validation["reason"]),
                    "local_union_iou": float(local_validation["local_union_iou"]),
                    "local_overlap_area": float(local_validation["local_overlap_area"]),
                }
            )

    faces = [faces_by_id[face_id] for face_id in face_order]
    validation = validate_global_approx_partition(graph_data, arcs, faces, config=config)
    if not validation["is_valid"]:
        return _reconcile_owner_approx_arcs_full_validation(
            graph_data,
            chains,
            face_payloads,
            config=config,
            candidates=candidates,
            fallback_reason="local candidate validation final global validation failed",
        )

    reconciliation = {
        "policy": "greedy_owner_boundary_transfer_v1",
        "candidate_validation": "local_pair_union_then_final_global",
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "accepted": accepted,
        "rejected": rejected[:64],
    }
    return arcs, faces, validation, reconciliation


def build_global_approx_partition_payload(
    graph_data: Dict[str, object],
    *,
    config: GlobalApproxConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or GlobalApproxConfig()
    chains = _extract_maximal_chains(graph_data)
    face_payloads = _face_approximation_payloads(graph_data, config=config)
    arcs, faces, validation, reconciliation = _reconcile_owner_approx_arcs(
        graph_data,
        chains,
        face_payloads,
        config=config,
    )
    face_approx_summary = [
        {
            "face_id": int(face_id),
            "label": int(payload["label"]),
            "original_vertex_count": int(payload["original_vertex_count"]),
            "approx_vertex_count": int(payload["approx_vertex_count"]),
            "approx_iou": float(payload["approx_iou"]),
            "base_primitive_count": int(payload["base_primitive_count"]),
        }
        for face_id, payload in sorted(face_payloads.items())
    ]
    return {
        "format": "global_owner_approx_partition_v1",
        "source_partition_graph": source_tag,
        "source_mask": graph_data.get("source_mask"),
        "size": [int(value) for value in graph_data["size"]],
        "config": asdict(config),
        "arcs": arcs,
        "faces": faces,
        "face_approximations": face_approx_summary,
        "reconciliation": reconciliation,
        "validation": validation,
    }


def build_global_approx_partition_from_path(
    partition_graph_path: str | Path,
    *,
    config: GlobalApproxConfig | None = None,
) -> Dict[str, object]:
    path = Path(partition_graph_path)
    graph_data = load_json(path)
    return build_global_approx_partition_payload(graph_data, config=config, source_tag=str(path.as_posix()))
