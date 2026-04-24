from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.geometry.polygon import orient

from partition_gen.convex_partition import (
    ConvexMergeConfig,
    _is_convex_polygon,
    _polygon_holes,
    _polygon_outer_vertices,
    _primitive_type,
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
        "reason": "CGAL optimal convex partition CLI is not available.",
        "triangle_count": int(payload["triangle_count"]),
        "baseline_final_primitive_count": int(payload["final_primitive_count"]),
    }


def _find_cgal_cli(config: BridgedPartitionConfig) -> str | None:
    if config.cgal_cli:
        return str(config.cgal_cli)
    return shutil.which("optimal_convex_partition_cli")


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
    return pieces, {"backend": "cgal", "optimal": True}


def run_simple_polygon_convex_partition(
    simple_ring: Sequence[Point2D],
    *,
    original_geometry: Polygon,
    config: BridgedPartitionConfig,
) -> Tuple[List[Polygon], Dict[str, object]]:
    if config.backend not in {"auto", "cgal", "fallback_hm", "fallback_cdt_greedy"}:
        raise ValueError(f"Unsupported backend: {config.backend}")

    has_holes = len(original_geometry.interiors) > 0
    if config.backend == "cgal" and has_holes:
        raise RuntimeError("CGAL backend for cut-open polygons with holes is not implemented in outer_star_v1.")

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
