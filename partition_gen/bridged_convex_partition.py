from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from partition_gen.convex_partition import (
    ConvexMergeConfig,
    _is_convex_polygon,
    _polygon_holes,
    _polygon_outer_vertices,
    _primitive_type,
    _shared_boundary_length,
    _single_polygon_union,
    greedy_convex_merge,
)


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class BridgeCandidate:
    id: int
    component_a: int
    vertex_a: int
    component_b: int
    vertex_b: int
    p: Point2D
    q: Point2D
    length: float


@dataclass(frozen=True)
class BridgeSet:
    bridge_ids: Tuple[int, ...]
    total_length: float


@dataclass(frozen=True)
class BridgedPartitionConfig:
    max_bridge_sets: int = 256
    vertex_round_digits: int = 8
    area_eps: float = 1e-8
    validity_eps: float = 1e-7
    backend: str = "auto"
    cgal_cli: str | None = None
    cut_slit_scale: float = 1e-6


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def _point_key(point: Point2D, digits: int) -> Tuple[float, float]:
    return (round(float(point[0]), digits), round(float(point[1]), digits))


def _segment_length(p: Point2D, q: Point2D) -> float:
    return float(math.hypot(float(q[0]) - float(p[0]), float(q[1]) - float(p[1])))


def _flatten_points(geometry) -> List[Point2D]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [(float(geometry.x), float(geometry.y))]
    if isinstance(geometry, MultiPoint):
        return [(float(point.x), float(point.y)) for point in geometry.geoms]
    if isinstance(geometry, GeometryCollection):
        points: List[Point2D] = []
        for item in geometry.geoms:
            points.extend(_flatten_points(item))
        return points
    return []


def _has_line_overlap(geometry, *, eps: float) -> bool:
    if geometry.is_empty:
        return False
    if isinstance(geometry, LineString):
        return geometry.length > eps
    if isinstance(geometry, MultiLineString):
        return any(item.length > eps for item in geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return any(_has_line_overlap(item, eps=eps) for item in geometry.geoms)
    return False


def _is_endpoint(point: Point2D, endpoint_keys: set[Tuple[float, float]], *, digits: int) -> bool:
    return _point_key(point, digits) in endpoint_keys


def _is_bridge_visible(
    polygon: Polygon,
    p: Point2D,
    q: Point2D,
    endpoint_keys: set[Tuple[float, float]],
    *,
    digits: int,
    eps: float,
) -> bool:
    if _segment_length(p, q) <= eps:
        return False

    line = LineString([p, q])
    if not polygon.buffer(eps).covers(line):
        return False

    boundary_intersection = line.intersection(polygon.boundary)
    if _has_line_overlap(boundary_intersection, eps=eps):
        return False
    for point in _flatten_points(boundary_intersection):
        if not _is_endpoint(point, endpoint_keys, digits=digits):
            return False

    for interior in polygon.interiors:
        hole_polygon = Polygon(interior)
        if line.crosses(hole_polygon) or line.within(hole_polygon):
            return False
        hole_intersection = line.intersection(hole_polygon.boundary)
        if _has_line_overlap(hole_intersection, eps=eps):
            return False
        for point in _flatten_points(hole_intersection):
            if not _is_endpoint(point, endpoint_keys, digits=digits):
                return False
    return True


def _component_rings(polygon: Polygon) -> List[List[Point2D]]:
    return [_polygon_outer_vertices(polygon), *_polygon_holes(polygon)]


def generate_bridge_candidates(polygon: Polygon, *, config: BridgedPartitionConfig) -> List[BridgeCandidate]:
    rings = _component_rings(polygon)
    candidates: List[BridgeCandidate] = []
    next_id = 0
    for component_a, ring_a in enumerate(rings):
        for component_b in range(component_a + 1, len(rings)):
            ring_b = rings[component_b]
            for vertex_a, p in enumerate(ring_a):
                for vertex_b, q in enumerate(ring_b):
                    endpoint_keys = {
                        _point_key(p, config.vertex_round_digits),
                        _point_key(q, config.vertex_round_digits),
                    }
                    if not _is_bridge_visible(
                        polygon,
                        p,
                        q,
                        endpoint_keys,
                        digits=config.vertex_round_digits,
                        eps=config.validity_eps,
                    ):
                        continue
                    candidates.append(
                        BridgeCandidate(
                            id=next_id,
                            component_a=int(component_a),
                            vertex_a=int(vertex_a),
                            component_b=int(component_b),
                            vertex_b=int(vertex_b),
                            p=(float(p[0]), float(p[1])),
                            q=(float(q[0]), float(q[1])),
                            length=_segment_length(p, q),
                        )
                    )
                    next_id += 1
    candidates.sort(key=lambda item: (item.length, item.component_a, item.component_b, item.vertex_a, item.vertex_b))
    return [BridgeCandidate(id=index, **{key: value for key, value in asdict(candidate).items() if key != "id"}) for index, candidate in enumerate(candidates)]


def _bridge_lines_cross(left: BridgeCandidate, right: BridgeCandidate, *, digits: int, eps: float) -> bool:
    left_line = LineString([left.p, left.q])
    right_line = LineString([right.p, right.q])
    intersection = left_line.intersection(right_line)
    if intersection.is_empty:
        return False
    if _has_line_overlap(intersection, eps=eps):
        return True
    left_endpoints = {_point_key(left.p, digits), _point_key(left.q, digits)}
    right_endpoints = {_point_key(right.p, digits), _point_key(right.q, digits)}
    allowed = left_endpoints | right_endpoints
    for point in _flatten_points(intersection):
        key = _point_key(point, digits)
        if key not in allowed:
            return True
    return False


def enumerate_outer_star_bridge_sets(
    candidates: Sequence[BridgeCandidate],
    *,
    hole_count: int,
    config: BridgedPartitionConfig,
) -> List[BridgeSet]:
    if hole_count == 0:
        return [BridgeSet(bridge_ids=(), total_length=0.0)]

    by_hole: Dict[int, List[BridgeCandidate]] = {hole_id: [] for hole_id in range(1, hole_count + 1)}
    for candidate in candidates:
        if {candidate.component_a, candidate.component_b} == {0, max(candidate.component_a, candidate.component_b)}:
            hole_id = candidate.component_b if candidate.component_a == 0 else candidate.component_a
            if 1 <= hole_id <= hole_count:
                by_hole[hole_id].append(candidate)

    ordered_holes = sorted(by_hole, key=lambda hole_id: len(by_hole[hole_id]))
    bridge_sets: List[BridgeSet] = []

    def dfs(hole_index: int, selected: List[BridgeCandidate]) -> None:
        if len(bridge_sets) >= config.max_bridge_sets:
            return
        if hole_index == len(ordered_holes):
            bridge_sets.append(
                BridgeSet(
                    bridge_ids=tuple(sorted(candidate.id for candidate in selected)),
                    total_length=float(sum(candidate.length for candidate in selected)),
                )
            )
            return
        hole_id = ordered_holes[hole_index]
        for candidate in by_hole.get(hole_id, []):
            if any(
                _bridge_lines_cross(candidate, existing, digits=config.vertex_round_digits, eps=config.validity_eps)
                for existing in selected
            ):
                continue
            selected.append(candidate)
            dfs(hole_index + 1, selected)
            selected.pop()

    dfs(0, [])
    bridge_sets.sort(key=lambda item: (item.total_length, item.bridge_ids))
    return bridge_sets[: config.max_bridge_sets]


def enumerate_tree_bridge_sets(
    candidates: Sequence[BridgeCandidate],
    *,
    component_count: int,
    config: BridgedPartitionConfig,
) -> List[BridgeSet]:
    required_edges = max(0, component_count - 1)
    if required_edges == 0:
        return [BridgeSet(bridge_ids=(), total_length=0.0)]

    bridge_sets: List[BridgeSet] = []

    def dfs(start: int, selected: List[BridgeCandidate], uf: _UnionFind) -> None:
        if len(bridge_sets) >= config.max_bridge_sets:
            return
        if len(selected) == required_edges:
            roots = {uf.find(component) for component in range(component_count)}
            if len(roots) == 1:
                bridge_sets.append(
                    BridgeSet(
                        bridge_ids=tuple(sorted(candidate.id for candidate in selected)),
                        total_length=float(sum(candidate.length for candidate in selected)),
                    )
                )
            return
        for index in range(start, len(candidates)):
            candidate = candidates[index]
            if uf.find(candidate.component_a) == uf.find(candidate.component_b):
                continue
            if any(
                _bridge_lines_cross(candidate, existing, digits=config.vertex_round_digits, eps=config.validity_eps)
                for existing in selected
            ):
                continue
            next_uf = _UnionFind(component_count)
            next_uf.parent = uf.parent[:]
            next_uf.union(candidate.component_a, candidate.component_b)
            selected.append(candidate)
            dfs(index + 1, selected, next_uf)
            selected.pop()

    dfs(0, [], _UnionFind(component_count))
    bridge_sets.sort(key=lambda item: (item.total_length, item.bridge_ids))
    return bridge_sets[: config.max_bridge_sets]


def build_outer_star_boundary_walk(
    polygon: Polygon,
    bridge_set: BridgeSet,
    candidates_by_id: Dict[int, BridgeCandidate],
) -> List[Point2D]:
    rings = _component_rings(polygon)
    outer = rings[0]
    bridges_by_outer_vertex: Dict[int, List[BridgeCandidate]] = {}
    for bridge_id in bridge_set.bridge_ids:
        candidate = candidates_by_id[bridge_id]
        if candidate.component_a == 0:
            outer_vertex = candidate.vertex_a
        elif candidate.component_b == 0:
            outer_vertex = candidate.vertex_b
        else:
            continue
        bridges_by_outer_vertex.setdefault(outer_vertex, []).append(candidate)

    for items in bridges_by_outer_vertex.values():
        items.sort(key=lambda item: item.length)

    walk: List[Point2D] = []
    for outer_index, outer_point in enumerate(outer):
        walk.append(outer_point)
        for bridge in bridges_by_outer_vertex.get(outer_index, []):
            if bridge.component_a == 0:
                hole_component = bridge.component_b
                hole_vertex = bridge.vertex_b
                hole_point = bridge.q
            else:
                hole_component = bridge.component_a
                hole_vertex = bridge.vertex_a
                hole_point = bridge.p
            hole_ring = rings[hole_component]
            walk.append(hole_point)
            for offset in range(1, len(hole_ring) + 1):
                walk.append(hole_ring[(hole_vertex + offset) % len(hole_ring)])
            walk.append(outer_point)
    return [(float(x), float(y)) for x, y in walk]


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for item in geometry.geoms:
            yield item
        return
    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def _bridge_candidates_for_set(
    bridge_set: BridgeSet,
    candidates_by_id: Dict[int, BridgeCandidate],
) -> List[BridgeCandidate]:
    return [candidates_by_id[bridge_id] for bridge_id in bridge_set.bridge_ids]


def _slit_distance(geometry: Polygon, *, config: BridgedPartitionConfig) -> float:
    minx, miny, maxx, maxy = geometry.bounds
    scale = max(float(maxx - minx), float(maxy - miny), 1.0)
    return max(config.validity_eps * 100.0, scale * config.cut_slit_scale)


def _extended_bridge_line(candidate: BridgeCandidate, distance: float) -> LineString:
    px, py = candidate.p
    qx, qy = candidate.q
    length = _segment_length(candidate.p, candidate.q)
    if length <= 0:
        return LineString([candidate.p, candidate.q])
    ux = (qx - px) / length
    uy = (qy - py) / length
    extension = max(distance * 8.0, 1e-9)
    return LineString(
        [
            (float(px - ux * extension), float(py - uy * extension)),
            (float(qx + ux * extension), float(qy + uy * extension)),
        ]
    )


def _cut_open_geometry_with_slits(
    geometry: Polygon,
    bridge_set: BridgeSet,
    candidates_by_id: Dict[int, BridgeCandidate],
    *,
    config: BridgedPartitionConfig,
) -> Tuple[Polygon | None, Dict[str, object]]:
    bridges = _bridge_candidates_for_set(bridge_set, candidates_by_id)
    distance = _slit_distance(geometry, config=config)
    if not bridges:
        return orient(geometry, sign=1.0), {"cut_slit_distance": 0.0, "cut_slit_area": 0.0}

    slit_buffers = [
        _extended_bridge_line(candidate, distance).buffer(distance, cap_style=2, join_style=2)
        for candidate in bridges
    ]
    slit_union = unary_union(slit_buffers)
    cut_geometry = geometry.difference(slit_union).buffer(0)
    polygons = [polygon for polygon in _iter_polygons(cut_geometry) if polygon.area > config.area_eps]
    metadata = {
        "cut_slit_distance": float(distance),
        "cut_slit_area": float(geometry.intersection(slit_union).area),
        "cut_component_count": int(len(polygons)),
    }
    if len(polygons) != 1:
        metadata["reject_reason"] = "cut geometry is not a single polygon"
        return None, metadata
    cut_polygon = orient(polygons[0], sign=1.0)
    metadata["cut_hole_count"] = int(len(cut_polygon.interiors))
    metadata["cut_area"] = float(cut_polygon.area)
    if len(cut_polygon.interiors) != 0:
        metadata["reject_reason"] = "cut geometry still has holes"
        return None, metadata
    if not cut_polygon.is_valid:
        metadata["reject_reason"] = "cut geometry is invalid"
        return None, metadata
    return cut_polygon, metadata


def _project_point_to_segment(point: Point2D, p: Point2D, q: Point2D) -> Tuple[Point2D, float, float]:
    px, py = p
    qx, qy = q
    vx = qx - px
    vy = qy - py
    wx = point[0] - px
    wy = point[1] - py
    denom = vx * vx + vy * vy
    if denom <= 0:
        return p, _segment_length(point, p), 0.0
    t = (wx * vx + wy * vy) / denom
    clamped = min(1.0, max(0.0, t))
    projected = (float(px + clamped * vx), float(py + clamped * vy))
    return projected, _segment_length(point, projected), float(t)


def _snap_point_to_bridges(
    point: Point2D,
    bridges: Sequence[BridgeCandidate],
    *,
    snap_eps: float,
) -> Point2D:
    best_point = point
    best_distance = snap_eps
    for bridge in bridges:
        projected, distance, t = _project_point_to_segment(point, bridge.p, bridge.q)
        if -0.05 <= t <= 1.05 and distance <= best_distance:
            if _segment_length(projected, bridge.p) <= snap_eps * 4.0:
                projected = bridge.p
            elif _segment_length(projected, bridge.q) <= snap_eps * 4.0:
                projected = bridge.q
            best_point = projected
            best_distance = distance
    return (float(best_point[0]), float(best_point[1]))


def _remove_duplicate_and_collinear(points: Sequence[Point2D], *, eps: float) -> List[Point2D]:
    cleaned: List[Point2D] = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if cleaned and _segment_length(cleaned[-1], point) <= eps:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and _segment_length(cleaned[0], cleaned[-1]) <= eps:
        cleaned.pop()

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_points: List[Point2D] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            prev_point = cleaned[(index - 1) % count]
            next_point = cleaned[(index + 1) % count]
            prev_length = _segment_length(prev_point, point)
            next_length = _segment_length(point, next_point)
            if prev_length <= eps or next_length <= eps:
                changed = True
                continue

            ax = point[0] - prev_point[0]
            ay = point[1] - prev_point[1]
            bx = next_point[0] - point[0]
            by = next_point[1] - point[1]
            cross = abs(ax * by - ay * bx)
            base_length = _segment_length(prev_point, next_point)
            line_distance = cross / max(base_length, eps)
            dot = (point[0] - prev_point[0]) * (point[0] - next_point[0]) + (
                point[1] - prev_point[1]
            ) * (point[1] - next_point[1])
            if line_distance <= eps and dot <= eps * max(base_length, 1.0):
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return cleaned


def _cluster_close_points(points: Sequence[Point2D], *, eps: float) -> Dict[Point2D, Point2D]:
    clusters: List[List[Point2D]] = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        matched_cluster: List[Point2D] | None = None
        for cluster in clusters:
            if any(_segment_length(point, existing) <= eps for existing in cluster):
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([point])
        else:
            matched_cluster.append(point)

    mapping: Dict[Point2D, Point2D] = {}
    for cluster in clusters:
        representative = (
            float(sum(point[0] for point in cluster) / len(cluster)),
            float(sum(point[1] for point in cluster) / len(cluster)),
        )
        for point in cluster:
            mapping[point] = representative
    return mapping


def _cleanup_polygon_after_snap(
    polygon: Polygon,
    *,
    cleanup_eps: float,
    config: BridgedPartitionConfig,
    point_mapping: Dict[Point2D, Point2D] | None = None,
) -> Polygon | None:
    if polygon.is_empty or polygon.area <= config.area_eps:
        return None
    ring = []
    for point in _polygon_outer_vertices(polygon):
        point = (float(point[0]), float(point[1]))
        ring.append(point_mapping.get(point, point) if point_mapping is not None else point)
    ring = _remove_duplicate_and_collinear(ring, eps=cleanup_eps)
    if len(ring) < 3:
        return None
    cleaned = Polygon(ring)
    if cleaned.is_empty or cleaned.area <= config.area_eps:
        return None
    if not cleaned.is_valid:
        fixed = cleaned.buffer(0)
        fixed_polygons = [item for item in _iter_polygons(fixed) if item.area > config.area_eps]
        if len(fixed_polygons) != 1:
            return None
        cleaned = fixed_polygons[0]
    if len(cleaned.interiors) > 0:
        return None
    return orient(cleaned, sign=1.0)


def _cleanup_snapped_pieces(
    pieces: Sequence[Polygon],
    *,
    cleanup_eps: float,
    config: BridgedPartitionConfig,
) -> List[Polygon]:
    all_points: List[Point2D] = []
    for piece in pieces:
        all_points.extend((float(x), float(y)) for x, y in _polygon_outer_vertices(piece))
    point_mapping = _cluster_close_points(all_points, eps=cleanup_eps)
    cleaned: List[Polygon] = []
    for piece in pieces:
        cleaned_piece = _cleanup_polygon_after_snap(
            piece,
            cleanup_eps=cleanup_eps,
            config=config,
            point_mapping=point_mapping,
        )
        if cleaned_piece is not None:
            cleaned.append(cleaned_piece)
    return cleaned


def _post_snap_cleanup_and_convex_merge(
    pieces: Sequence[Polygon],
    *,
    cleanup_eps: float,
    config: BridgedPartitionConfig,
) -> Tuple[List[Polygon], Dict[str, object]]:
    active = _cleanup_snapped_pieces(pieces, cleanup_eps=cleanup_eps, config=config)
    merge_history: List[Dict[str, object]] = []
    shared_eps = max(config.validity_eps, cleanup_eps * 0.25)

    while True:
        best: Tuple[float, float, int, int, Polygon] | None = None
        for left_index in range(len(active)):
            for right_index in range(left_index + 1, len(active)):
                left = active[left_index]
                right = active[right_index]
                shared_length = _shared_boundary_length(left, right)
                if shared_length <= shared_eps:
                    continue
                union = _single_polygon_union(left, right)
                if union is None:
                    continue
                merged = _cleanup_polygon_after_snap(
                    union,
                    cleanup_eps=cleanup_eps,
                    config=config,
                )
                if merged is None:
                    continue
                if not _is_convex_polygon(merged, rel_eps=config.validity_eps, abs_eps=config.area_eps):
                    continue
                candidate = (float(shared_length), float(merged.area), left_index, right_index, merged)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate

        if best is None:
            break

        shared_length, merged_area, left_index, right_index, merged = best
        merge_history.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "shared_edge_length": float(shared_length),
                "merged_area": float(merged_area),
                "vertex_count": int(len(_polygon_outer_vertices(merged))),
            }
        )
        next_active = [
            piece
            for index, piece in enumerate(active)
            if index not in {left_index, right_index}
        ]
        next_active.append(merged)
        next_active.sort(key=lambda item: (-float(item.area), item.centroid.x, item.centroid.y))
        active = next_active

    return active, {
        "post_snap_cleanup_eps": float(cleanup_eps),
        "post_snap_merge_count": int(len(merge_history)),
        "post_snap_merge_history": merge_history,
        "post_snap_final_piece_count": int(len(active)),
    }


def _snap_piece_to_bridge_lines(
    piece: Polygon,
    bridges: Sequence[BridgeCandidate],
    *,
    snap_eps: float,
    config: BridgedPartitionConfig,
) -> Polygon | None:
    if piece.is_empty or piece.area <= config.area_eps:
        return None
    ring = _polygon_outer_vertices(piece)
    snapped = [
        _snap_point_to_bridges((float(x), float(y)), bridges, snap_eps=snap_eps)
        for x, y in ring
    ]
    snapped = _remove_duplicate_and_collinear(snapped, eps=config.validity_eps)
    if len(snapped) < 3:
        return None
    polygon = Polygon(snapped)
    if polygon.is_empty or polygon.area <= config.area_eps:
        return None
    if not polygon.is_valid:
        fixed = polygon.buffer(0)
        fixed_polygons = [item for item in _iter_polygons(fixed) if item.area > config.area_eps]
        if len(fixed_polygons) != 1:
            return None
        polygon = fixed_polygons[0]
    return orient(polygon, sign=1.0)


def _sum_piece_vertices(pieces: Sequence[Polygon]) -> int:
    return sum(len(_polygon_outer_vertices(piece)) for piece in pieces)


def _polygon_from_primitive_payload(primitive: Dict[str, object]) -> Polygon:
    return orient(Polygon(primitive["outer"], primitive.get("holes", [])), sign=1.0)


def _primitive_payload(primitive_id: int, polygon: Polygon) -> Dict[str, object]:
    polygon = orient(polygon, sign=1.0)
    outer = _polygon_outer_vertices(polygon)
    centroid = polygon.centroid
    return {
        "id": int(primitive_id),
        "type": _primitive_type(polygon),
        "outer": [[float(x), float(y)] for x, y in outer],
        "holes": [],
        "vertex_count": int(len(outer)),
        "area": float(polygon.area),
        "centroid": [float(centroid.x), float(centroid.y)],
    }


def _pieces_from_fallback(geometry: Polygon, *, config: BridgedPartitionConfig) -> Tuple[List[Polygon], Dict[str, object]]:
    merge_config = ConvexMergeConfig(
        convex_rel_eps=config.validity_eps,
        convex_abs_eps=config.area_eps,
        shared_edge_eps=config.validity_eps,
        area_eps=config.area_eps,
        vertex_round_digits=config.vertex_round_digits,
    )
    payload = greedy_convex_merge(geometry, config=merge_config)
    pieces = [_polygon_from_primitive_payload(primitive) for primitive in payload["primitives"]]
    return pieces, {
        "backend": "fallback_cdt_greedy",
        "optimal": False,
        "simple_polygon_optimal": False,
        "global_optimal": False,
        "optimal_scope": "fallback_cdt_greedy",
        "reason": "CGAL optimal convex partition CLI is not available.",
        "triangle_count": int(payload["triangle_count"]),
        "baseline_final_primitive_count": int(payload["final_primitive_count"]),
    }


def _find_cgal_cli(config: BridgedPartitionConfig) -> str | None:
    if config.cgal_cli:
        return str(config.cgal_cli)
    path_cli = shutil.which("optimal_convex_partition_cli")
    if path_cli:
        return path_cli

    executable_names = ["optimal_convex_partition_cli.exe", "optimal_convex_partition_cli"]
    repo_root = Path(__file__).resolve().parents[1]
    candidate_dirs = [
        repo_root / "build" / "cgal_tools" / "Release",
        repo_root / "build" / "cgal_tools" / "Debug",
        repo_root / "build" / "Release",
        repo_root / "build" / "Debug",
        repo_root / "tools" / "build" / "Release",
        repo_root / "tools" / "build" / "Debug",
        repo_root / "tools",
    ]
    for directory in candidate_dirs:
        for executable_name in executable_names:
            candidate = directory / executable_name
            if candidate.exists():
                return str(candidate)
    return None


def _run_cgal_cli(simple_ring: Sequence[Point2D], *, config: BridgedPartitionConfig) -> Tuple[List[Polygon], Dict[str, object]] | None:
    cli_path = _find_cgal_cli(config)
    if cli_path is None:
        return None
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        input_path.write_text(
            __import__("json").dumps({"outer": [[float(x), float(y)] for x, y in simple_ring]}),
            encoding="utf-8",
        )
        result = subprocess.run(
            [cli_path, str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CGAL CLI failed: {result.stderr.strip() or result.stdout.strip()}")
        payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
        if not payload.get("success", False):
            raise RuntimeError(f"CGAL CLI returned success=false: {payload.get('error', 'unknown error')}")
        pieces = [orient(Polygon(ring), sign=1.0) for ring in payload.get("pieces", [])]
    backend_info = dict(payload.get("backend_info") or {})
    backend_info.setdefault("backend", "cgal")
    backend_info.setdefault("optimal", True)
    backend_info.setdefault("simple_polygon_optimal", True)
    backend_info.setdefault("global_optimal", True)
    backend_info.setdefault("optimal_scope", "simple_polygon")
    backend_info["cli_path"] = str(cli_path)
    backend_info["piece_count"] = int(len(pieces))
    return pieces, backend_info


def _run_cgal_cut_open_bridge_sets(
    geometry: Polygon,
    bridge_sets: Sequence[BridgeSet],
    candidates_by_id: Dict[int, BridgeCandidate],
    *,
    config: BridgedPartitionConfig,
) -> Tuple[List[Polygon], Dict[str, object], BridgeSet, List[Point2D]] | None:
    if not bridge_sets:
        return None
    if _find_cgal_cli(config) is None:
        return None

    best: Tuple[Tuple[object, ...], List[Polygon], Dict[str, object], BridgeSet, List[Point2D]] | None = None
    rejected: List[Dict[str, object]] = []
    attempts = 0
    for bridge_set in bridge_sets[: config.max_bridge_sets]:
        attempts += 1
        cut_geometry, cut_metadata = _cut_open_geometry_with_slits(
            geometry,
            bridge_set,
            candidates_by_id,
            config=config,
        )
        if cut_geometry is None:
            rejected.append(
                {
                    "bridge_ids": list(bridge_set.bridge_ids),
                    "reason": cut_metadata.get("reject_reason", "invalid cut geometry"),
                }
            )
            continue

        simple_ring = _polygon_outer_vertices(cut_geometry)
        try:
            cgal_result = _run_cgal_cli(simple_ring, config=config)
        except RuntimeError as error:
            rejected.append({"bridge_ids": list(bridge_set.bridge_ids), "reason": str(error)})
            continue
        if cgal_result is None:
            return None

        cut_pieces, backend_info = cgal_result
        bridges = _bridge_candidates_for_set(bridge_set, candidates_by_id)
        snap_eps = max(float(cut_metadata["cut_slit_distance"]) * 4.0, config.validity_eps * 10.0)
        snapped_pieces = [
            snapped
            for piece in cut_pieces
            if (snapped := _snap_piece_to_bridge_lines(piece, bridges, snap_eps=snap_eps, config=config)) is not None
        ]
        cleanup_eps = max(snap_eps * 4.0, config.validity_eps)
        final_pieces, post_snap_info = _post_snap_cleanup_and_convex_merge(
            snapped_pieces,
            cleanup_eps=cleanup_eps,
            config=config,
        )
        validation = validate_bridged_partition(geometry, final_pieces, config=config)
        score = (
            0 if validation["is_valid"] else 1,
            int(len(final_pieces)),
            float(bridge_set.total_length),
            int(_sum_piece_vertices(final_pieces)),
            -float(validation["iou"]),
        )
        candidate_backend = dict(backend_info)
        candidate_backend.update(
            {
                "backend": "cgal_bridge_cut",
                "optimal": bool(validation["is_valid"]),
                "simple_polygon_backend": "cgal",
                "simple_polygon_optimal": True,
                "optimal_scope": "selected_bridge_cut_simple_polygon",
                "global_optimal": False,
                "bridge_cut_mode": "epsilon_slit_snap_v1",
                "bridge_policy": "outer_star_v1",
                "cut_metadata": cut_metadata,
                "snap_eps": float(snap_eps),
                "pre_snap_piece_count": int(len(cut_pieces)),
                "post_snap_piece_count": int(len(snapped_pieces)),
                **post_snap_info,
                "selection_score": [float(value) if isinstance(value, float) else int(value) for value in score],
            }
        )
        if rejected:
            candidate_backend["rejected_bridge_sets"] = rejected[:16]

        if best is None or score < best[0]:
            best = (score, final_pieces, candidate_backend, bridge_set, simple_ring)
            if validation["is_valid"] and len(final_pieces) == 1:
                break

    if best is None:
        return None
    _, pieces, backend_info, bridge_set, simple_ring = best
    backend_info["evaluated_bridge_set_count"] = int(attempts)
    backend_info["rejected_bridge_set_count"] = int(len(rejected))
    if rejected:
        backend_info.setdefault("rejected_bridge_sets", rejected[:16])
    return pieces, backend_info, bridge_set, simple_ring


def run_simple_polygon_convex_partition(
    simple_ring: Sequence[Point2D],
    *,
    original_geometry: Polygon,
    config: BridgedPartitionConfig,
) -> Tuple[List[Polygon], Dict[str, object]]:
    if config.backend not in {"auto", "cgal", "fallback_hm", "fallback_cdt_greedy"}:
        raise ValueError(f"Unsupported backend: {config.backend}")

    has_holes = len(original_geometry.interiors) > 0
    if config.backend in {"auto", "cgal"} and not has_holes:
        cgal_result = _run_cgal_cli(simple_ring, config=config)
        if cgal_result is not None:
            return cgal_result
        if config.backend == "cgal":
            raise RuntimeError("CGAL backend requested, but optimal_convex_partition_cli was not found.")

    pieces, backend_info = _pieces_from_fallback(original_geometry, config=config)
    if config.backend == "fallback_hm":
        backend_info["requested_backend"] = "fallback_hm"
        backend_info["reason"] = "Hertel-Mehlhorn fallback CLI is not implemented; using CDT greedy fallback."
    return pieces, backend_info


def validate_bridged_partition(original_geometry: Polygon, pieces: Sequence[Polygon], *, config: BridgedPartitionConfig) -> Dict[str, object]:
    union = None
    overlap_area = 0.0
    for index, piece in enumerate(pieces):
        for other in pieces[index + 1 :]:
            overlap_area += float(piece.intersection(other).area)
        union = piece if union is None else union.union(piece)

    if union is None:
        union = Polygon()

    union_area = float(union.area) if not union.is_empty else 0.0
    total_union = original_geometry.union(union)
    iou = (
        float(original_geometry.intersection(union).area / total_union.area)
        if not total_union.is_empty and total_union.area > 0
        else 0.0
    )
    all_convex = all(
        _is_convex_polygon(piece, rel_eps=config.validity_eps, abs_eps=config.area_eps)
        for piece in pieces
    )
    is_valid = bool(iou >= 1.0 - 1e-6 and overlap_area <= config.validity_eps and all_convex)
    return {
        "is_valid": is_valid,
        "iou": float(iou),
        "covered_area": union_area,
        "original_area": float(original_geometry.area),
        "overlap_area": float(overlap_area),
        "all_convex": bool(all_convex),
        "piece_count": int(len(pieces)),
    }


def _bridge_candidate_payload(candidate: BridgeCandidate) -> Dict[str, object]:
    payload = asdict(candidate)
    payload["p"] = [float(candidate.p[0]), float(candidate.p[1])]
    payload["q"] = [float(candidate.q[0]), float(candidate.q[1])]
    return payload


def _bridge_set_payload(bridge_set: BridgeSet | None) -> Dict[str, object] | None:
    if bridge_set is None:
        return None
    return {
        "bridge_ids": [int(value) for value in bridge_set.bridge_ids],
        "total_length": float(bridge_set.total_length),
    }


def bridged_optimal_convex_partition(
    geometry: Polygon,
    *,
    config: BridgedPartitionConfig | None = None,
) -> Dict[str, object]:
    config = config or BridgedPartitionConfig()
    geometry = orient(geometry, sign=1.0)
    hole_count = len(geometry.interiors)
    candidates = generate_bridge_candidates(geometry, config=config)
    bridge_sets = enumerate_outer_star_bridge_sets(candidates, hole_count=hole_count, config=config)
    candidates_by_id = {candidate.id: candidate for candidate in candidates}

    selected_bridge_set: BridgeSet | None = None
    simple_ring = _polygon_outer_vertices(geometry)
    pieces: List[Polygon] | None = None
    backend_info: Dict[str, object] | None = None

    if hole_count > 0 and config.backend in {"auto", "cgal"}:
        cut_result = _run_cgal_cut_open_bridge_sets(
            geometry,
            bridge_sets,
            candidates_by_id,
            config=config,
        )
        if cut_result is not None:
            pieces, backend_info, selected_bridge_set, simple_ring = cut_result
        elif config.backend == "cgal":
            raise RuntimeError("CGAL bridge cut backend failed for all bridge sets.")

    if pieces is None or backend_info is None:
        if not bridge_sets:
            selected_bridge_set = None
            simple_ring = _polygon_outer_vertices(geometry)
        else:
            selected_bridge_set = bridge_sets[0]
            simple_ring = (
                _polygon_outer_vertices(geometry)
                if hole_count == 0
                else build_outer_star_boundary_walk(geometry, selected_bridge_set, candidates_by_id)
            )

        pieces, backend_info = run_simple_polygon_convex_partition(
            simple_ring,
            original_geometry=geometry,
            config=config,
        )
    validation = validate_bridged_partition(geometry, pieces, config=config)
    primitives = [_primitive_payload(index, piece) for index, piece in enumerate(pieces)]

    return {
        "method": "bridged_optimal_convex_partition",
        "bridge_policy": "outer_star_v1",
        "backend_info": backend_info,
        "hole_count": int(hole_count),
        "bridge_candidates": [_bridge_candidate_payload(candidate) for candidate in candidates],
        "selected_bridge_set": _bridge_set_payload(selected_bridge_set),
        "available_bridge_set_count": int(len(bridge_sets)),
        "simple_polygon_vertex_count": int(len(simple_ring)),
        "simple_polygon_boundary_walk": [[float(x), float(y)] for x, y in simple_ring],
        "final_primitive_count": int(len(primitives)),
        "final_convex_count": int(len(primitives)),
        "primitives": primitives,
        "validation": validation,
    }


def build_bridged_convex_partition_from_geometry_payload(
    geometry_payload: Dict[str, object],
    *,
    config: BridgedPartitionConfig | None = None,
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
    payload = bridged_optimal_convex_partition(geometry, config=config)
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
