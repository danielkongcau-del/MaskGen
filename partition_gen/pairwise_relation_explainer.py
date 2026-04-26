from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from partition_gen.bridged_convex_partition import (
    BridgedPartitionConfig,
    bridged_optimal_convex_partition,
)


@dataclass(frozen=True)
class PairwiseRelationConfig:
    convex_backend: str = "fallback_cdt_greedy"
    convex_cgal_cli: str | None = None
    convex_max_bridge_sets: int = 128
    convex_cut_slit_scale: float = 1e-6
    area_eps: float = 1e-8
    validity_eps: float = 1e-7
    min_shared_length: float = 1e-6
    cost_piece: float = 1.0
    cost_vertex: float = 0.25
    cost_object: float = 2.0
    cost_relation: float = 1.0
    cost_uncovered_area: float = 0.01
    cost_fill_expansion_area: float = 0.0105
    cost_false_cover_area: float = 0.2
    max_false_cover_ratio: float = 0.12
    divider_min_aspect: float = 2.0
    divider_shape_penalty: float = 8.0
    divider_context_bonus: float = 12.0
    divider_min_context_score: float = 0.6
    divider_low_context_penalty: float = 20.0
    divider_area_ratio_penalty: float = 20.0
    divider_fragment_penalty: float = 0.75


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1e-9:
        ring = ring[:-1]
    return ring


def _polygon_from_face(face: Dict[str, object]) -> Polygon:
    geometry = face.get("geometry") or {}
    outer = _trim_ring(geometry.get("outer", []))
    holes = [_trim_ring(ring) for ring in geometry.get("holes", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        polygon = Polygon(outer, holes)
    except Exception:
        return Polygon()
    fixed = polygon if polygon.is_valid else polygon.buffer(0)
    return fixed if isinstance(fixed, Polygon) else unary_union([item for item in _iter_polygons(fixed)])


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


def _union_polygons(polygons: Sequence[Polygon]) -> Polygon | MultiPolygon:
    non_empty = [polygon for polygon in polygons if not polygon.is_empty and polygon.area > 0.0]
    if not non_empty:
        return Polygon()
    return unary_union(non_empty)


def _geometry_area(geometry) -> float:
    return float(geometry.area) if not geometry.is_empty else 0.0


def _geometry_vertex_count(geometry) -> int:
    count = 0
    for polygon in _iter_polygons(geometry):
        count += max(0, len(_trim_ring(polygon.exterior.coords)))
        count += sum(max(0, len(_trim_ring(ring.coords))) for ring in polygon.interiors)
    return int(count)


def _compactness(geometry) -> float:
    area = _geometry_area(geometry)
    perimeter = float(geometry.length) if not geometry.is_empty else 0.0
    if perimeter <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _oriented_aspect(geometry) -> float:
    polygons = [polygon for polygon in _iter_polygons(geometry)]
    if not polygons:
        return 0.0
    hull = unary_union(polygons).convex_hull
    if hull.is_empty:
        return 0.0
    rect = hull.minimum_rotated_rectangle
    coords = _trim_ring(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    lengths = [
        math.hypot(coords[(idx + 1) % len(coords)][0] - coords[idx][0], coords[(idx + 1) % len(coords)][1] - coords[idx][1])
        for idx in range(len(coords))
    ]
    short = min(lengths)
    long = max(lengths)
    if short <= 1e-9:
        return 0.0
    return float(long / short)


def _component_count(geometry) -> int:
    return int(len([polygon for polygon in _iter_polygons(geometry) if polygon.area > 0.0]))


def _largest_component_area(geometry) -> float:
    areas = [float(polygon.area) for polygon in _iter_polygons(geometry) if polygon.area > 0.0]
    return float(max(areas) if areas else 0.0)


def _weighted_component_aspect(geometry) -> float:
    polygons = [polygon for polygon in _iter_polygons(geometry) if polygon.area > 0.0]
    if not polygons:
        return 0.0
    total_area = sum(float(polygon.area) for polygon in polygons)
    if total_area <= 1e-9:
        return 0.0
    return float(sum(_oriented_aspect(polygon) * float(polygon.area) for polygon in polygons) / total_area)


def _geometry_cache_key(geometry) -> Tuple[float, float, int]:
    bounds = geometry.bounds if not geometry.is_empty else (0.0, 0.0, 0.0, 0.0)
    return (
        round(float(geometry.area), 6),
        round(float(geometry.length), 6),
        hash(tuple(round(float(value), 6) for value in bounds)),
    )


def _convex_partition_cost(
    geometry,
    *,
    config: PairwiseRelationConfig,
    cache: Dict[Tuple[float, float, int], Dict[str, object]] | None = None,
) -> Dict[str, object]:
    key = _geometry_cache_key(geometry)
    if cache is not None and key in cache:
        return dict(cache[key])
    polygons = [polygon for polygon in _iter_polygons(geometry) if polygon.area > config.area_eps]
    if not polygons:
        result = {
            "cost": 0.0,
            "piece_count": 0,
            "vertex_count": 0,
            "valid": True,
            "failure_reason": None,
        }
        if cache is not None:
            cache[key] = dict(result)
        return result
    total_cost = 0.0
    total_pieces = 0
    total_vertices = 0
    failures = []
    bridged_config = BridgedPartitionConfig(
        max_bridge_sets=config.convex_max_bridge_sets,
        area_eps=config.area_eps,
        validity_eps=config.validity_eps,
        backend=config.convex_backend,
        cgal_cli=config.convex_cgal_cli,
        cut_slit_scale=config.convex_cut_slit_scale,
    )
    for polygon in polygons:
        try:
            payload = bridged_optimal_convex_partition(polygon, config=bridged_config)
        except Exception as exc:
            failures.append(str(exc))
            total_pieces += 1
            total_vertices += _geometry_vertex_count(polygon)
            continue
        primitives = payload.get("primitives", [])
        total_pieces += int(len(primitives))
        total_vertices += int(sum(int(item.get("vertex_count", len(item.get("outer", [])))) for item in primitives))
        if not payload.get("validation", {}).get("is_valid", False):
            failures.append("invalid convex partition")
    total_cost = config.cost_piece * total_pieces + config.cost_vertex * total_vertices
    result = {
        "cost": float(total_cost),
        "piece_count": int(total_pieces),
        "vertex_count": int(total_vertices),
        "valid": bool(not failures),
        "failure_reason": "; ".join(failures) if failures else None,
    }
    if cache is not None:
        cache[key] = dict(result)
    return result


def _filled_support_geometry(support_geometry, blocker_geometry, *, fill_policy: str):
    if fill_policy == "no_fill":
        return support_geometry
    if fill_policy == "convex_hull_fill":
        return support_geometry.convex_hull if not support_geometry.is_empty else support_geometry
    raise ValueError(f"Unknown fill policy: {fill_policy}")


def _support_insert_candidate(
    support_label: int,
    insert_label: int,
    support_geometry,
    insert_geometry,
    *,
    fill_policy: str,
    config: PairwiseRelationConfig,
    cost_cache: Dict[Tuple[float, float, int], Dict[str, object]],
) -> Dict[str, object]:
    support_filled = _filled_support_geometry(support_geometry, insert_geometry, fill_policy=fill_policy)
    support_fill_expansion_area = float(max(0.0, _geometry_area(support_filled) - _geometry_area(support_geometry)))
    support_fill_expansion_cost = config.cost_fill_expansion_area * support_fill_expansion_area
    false_cover_area = float(max(0.0, support_filled.intersection(insert_geometry).area))
    support_area = max(_geometry_area(support_filled), config.area_eps)
    insert_area = _geometry_area(insert_geometry)
    largest_insert_area = _largest_component_area(insert_geometry)
    insert_support_ratio = float(insert_area / support_area)
    largest_insert_support_ratio = float(largest_insert_area / support_area)
    covered_insert_area = float(max(0.0, support_filled.intersection(insert_geometry).area))
    insert_coverage_ratio = float(covered_insert_area / max(insert_area, config.area_eps))
    uncovered_insert_area = float(max(0.0, insert_area - covered_insert_area))
    uncovered_insert_ratio = float(uncovered_insert_area / max(insert_area, config.area_eps))
    false_cover_ratio = float(false_cover_area / support_area)
    support_cost = _convex_partition_cost(support_filled, config=config, cache=cost_cache)
    insert_cost = _convex_partition_cost(insert_geometry, config=config, cache=cost_cache)
    insert_components = max(1, _component_count(insert_geometry))
    insert_group_bonus = -min(6.0, math.log1p(insert_components) * 2.0) if insert_support_ratio <= 1.0 else 0.0
    invalid = 0.0
    if uncovered_insert_ratio > config.max_false_cover_ratio:
        invalid += 1000.0 * (uncovered_insert_ratio - config.max_false_cover_ratio)
    if insert_support_ratio > 1.0:
        invalid += 1000.0 * (insert_support_ratio - 1.0)
    if insert_support_ratio > 0.2 and largest_insert_support_ratio > 0.1:
        invalid += 1000.0 * max(insert_support_ratio - 0.2, largest_insert_support_ratio - 0.1)
    total = (
        config.cost_object * 2.0
        + config.cost_relation
        + float(support_cost["cost"])
        + float(insert_cost["cost"])
        + insert_group_bonus
        + support_fill_expansion_cost
        + config.cost_uncovered_area * uncovered_insert_area
        + invalid
    )
    return {
        "template": "support_with_inserts",
        "fill_policy": fill_policy,
        "roles": {str(support_label): "support_region", str(insert_label): "insert_object"},
        "cost": float(total),
        "cost_breakdown": {
            "object": config.cost_object * 2.0,
            "relation": config.cost_relation,
            "support_geometry": support_cost,
            "insert_geometry": insert_cost,
            "insert_components": int(insert_components),
            "insert_support_ratio": insert_support_ratio,
            "largest_insert_support_ratio": largest_insert_support_ratio,
            "insert_coverage_ratio": insert_coverage_ratio,
            "uncovered_insert_area": uncovered_insert_area,
            "uncovered_insert_ratio": uncovered_insert_ratio,
            "insert_group_bonus": float(insert_group_bonus),
            "support_fill_expansion_area": support_fill_expansion_area,
            "support_fill_expansion_cost": support_fill_expansion_cost,
            "false_cover_area": false_cover_area,
            "false_cover_ratio": false_cover_ratio,
            "uncovered_insert_cost": config.cost_uncovered_area * uncovered_insert_area,
            "invalid": invalid,
        },
        "geometry_summary": {
            "support_area": _geometry_area(support_geometry),
            "support_filled_area": _geometry_area(support_filled),
            "insert_area": insert_area,
            "largest_insert_area": largest_insert_area,
            "support_compactness": _compactness(support_filled),
            "insert_compactness": _compactness(insert_geometry),
        },
    }


def _support_divider_candidate(
    support_label: int,
    divider_label: int,
    support_geometry,
    divider_geometry,
    *,
    fill_policy: str,
    divider_context_score: float,
    config: PairwiseRelationConfig,
    cost_cache: Dict[Tuple[float, float, int], Dict[str, object]],
) -> Dict[str, object]:
    support_filled = _filled_support_geometry(support_geometry, divider_geometry, fill_policy=fill_policy)
    support_fill_expansion_area = float(max(0.0, _geometry_area(support_filled) - _geometry_area(support_geometry)))
    support_fill_expansion_cost = config.cost_fill_expansion_area * support_fill_expansion_area
    false_cover_area = float(max(0.0, support_filled.intersection(divider_geometry).area))
    support_area = max(_geometry_area(support_filled), config.area_eps)
    divider_area = _geometry_area(divider_geometry)
    divider_support_ratio = float(divider_area / support_area)
    covered_divider_area = float(max(0.0, support_filled.intersection(divider_geometry).area))
    uncovered_divider_area = float(max(0.0, divider_area - covered_divider_area))
    uncovered_divider_ratio = float(uncovered_divider_area / max(divider_area, config.area_eps))
    false_cover_ratio = float(false_cover_area / support_area)
    support_cost = _convex_partition_cost(support_filled, config=config, cache=cost_cache)
    divider_cost = _convex_partition_cost(divider_geometry, config=config, cache=cost_cache)
    divider_aspect = _weighted_component_aspect(divider_geometry)
    divider_components = max(1, _component_count(divider_geometry))
    divider_bonus = -min(4.0, max(0.0, divider_aspect - 1.0)) * 0.75
    divider_shape_penalty = max(0.0, config.divider_min_aspect - divider_aspect) * config.divider_shape_penalty
    context_bonus = -float(divider_context_score) * config.divider_context_bonus
    low_context_penalty = max(0.0, config.divider_min_context_score - float(divider_context_score)) * config.divider_low_context_penalty
    divider_area_penalty = max(0.0, divider_support_ratio - 0.5) * config.divider_area_ratio_penalty
    divider_fragment_penalty = max(0.0, float(divider_components - 8)) * config.divider_fragment_penalty
    invalid = 0.0
    if uncovered_divider_ratio > config.max_false_cover_ratio:
        invalid += 1000.0 * (uncovered_divider_ratio - config.max_false_cover_ratio)
    total = (
        config.cost_object * 2.0
        + config.cost_relation
        + float(support_cost["cost"])
        + float(divider_cost["cost"])
        + support_fill_expansion_cost
        + config.cost_uncovered_area * uncovered_divider_area
        + divider_bonus
        + divider_shape_penalty
        + context_bonus
        + low_context_penalty
        + divider_area_penalty
        + divider_fragment_penalty
        + invalid
    )
    return {
        "template": "split_by_divider",
        "fill_policy": fill_policy,
        "roles": {str(support_label): "support_region", str(divider_label): "divider_region"},
        "cost": float(total),
        "cost_breakdown": {
            "object": config.cost_object * 2.0,
            "relation": config.cost_relation,
            "support_geometry": support_cost,
            "divider_geometry": divider_cost,
            "divider_aspect": divider_aspect,
            "divider_components": int(divider_components),
            "divider_support_ratio": divider_support_ratio,
            "divider_bonus": divider_bonus,
            "divider_shape_penalty": divider_shape_penalty,
            "divider_context_score": float(divider_context_score),
            "divider_context_bonus": context_bonus,
            "divider_low_context_penalty": low_context_penalty,
            "support_fill_expansion_area": support_fill_expansion_area,
            "support_fill_expansion_cost": support_fill_expansion_cost,
            "divider_area_penalty": divider_area_penalty,
            "divider_fragment_penalty": divider_fragment_penalty,
            "false_cover_area": false_cover_area,
            "false_cover_ratio": false_cover_ratio,
            "covered_divider_area": covered_divider_area,
            "uncovered_divider_area": uncovered_divider_area,
            "uncovered_divider_ratio": uncovered_divider_ratio,
            "uncovered_divider_cost": config.cost_uncovered_area * uncovered_divider_area,
            "invalid": invalid,
        },
        "geometry_summary": {
            "support_area": _geometry_area(support_geometry),
            "support_filled_area": _geometry_area(support_filled),
            "divider_area": divider_area,
            "support_compactness": _compactness(support_filled),
            "divider_compactness": _compactness(divider_geometry),
            "divider_aspect": divider_aspect,
            "divider_components": int(divider_components),
        },
    }


def _adjacent_supports_candidate(
    left_label: int,
    right_label: int,
    left_geometry,
    right_geometry,
    *,
    config: PairwiseRelationConfig,
    cost_cache: Dict[Tuple[float, float, int], Dict[str, object]],
) -> Dict[str, object]:
    left_cost = _convex_partition_cost(left_geometry, config=config, cache=cost_cache)
    right_cost = _convex_partition_cost(right_geometry, config=config, cache=cost_cache)
    total = config.cost_object * 2.0 + config.cost_relation + float(left_cost["cost"]) + float(right_cost["cost"])
    return {
        "template": "adjacent_supports",
        "fill_policy": "none",
        "roles": {str(left_label): "support_region", str(right_label): "support_region"},
        "cost": float(total),
        "cost_breakdown": {
            "object": config.cost_object * 2.0,
            "relation": config.cost_relation,
            "left_geometry": left_cost,
            "right_geometry": right_cost,
            "invalid": 0.0,
        },
        "geometry_summary": {
            "left_area": _geometry_area(left_geometry),
            "right_area": _geometry_area(right_geometry),
        },
    }


def _independent_candidate(
    left_label: int,
    right_label: int,
    left_geometry,
    right_geometry,
    *,
    config: PairwiseRelationConfig,
    cost_cache: Dict[Tuple[float, float, int], Dict[str, object]],
) -> Dict[str, object]:
    left_cost = _convex_partition_cost(left_geometry, config=config, cache=cost_cache)
    right_cost = _convex_partition_cost(right_geometry, config=config, cache=cost_cache)
    total = config.cost_object * 2.0 + 2.0 + float(left_cost["cost"]) + float(right_cost["cost"])
    return {
        "template": "independent_faces",
        "fill_policy": "none",
        "roles": {str(left_label): "residual_region", str(right_label): "residual_region"},
        "cost": float(total),
        "cost_breakdown": {
            "object": config.cost_object * 2.0,
            "residual": 2.0,
            "left_geometry": left_cost,
            "right_geometry": right_cost,
            "invalid": 0.0,
        },
        "geometry_summary": {
            "left_area": _geometry_area(left_geometry),
            "right_area": _geometry_area(right_geometry),
        },
    }


def _pair_shared_lengths(evidence_payload: Dict[str, object]) -> Dict[Tuple[int, int], float]:
    output: Dict[Tuple[int, int], float] = {}
    for edge in evidence_payload.get("adjacency", []):
        labels = [int(value) for value in edge.get("labels", [])]
        if len(labels) != 2 or labels[0] == labels[1]:
            continue
        key = (min(labels), max(labels))
        output[key] = float(output.get(key, 0.0) + float(edge.get("shared_length", 0.0)))
    return output


def build_pairwise_relation_payload(
    evidence_payload: Dict[str, object],
    *,
    config: PairwiseRelationConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or PairwiseRelationConfig()
    faces_by_label: Dict[int, List[Dict[str, object]]] = {}
    for face in evidence_payload.get("faces", []):
        faces_by_label.setdefault(int(face["label"]), []).append(face)

    geometry_by_label = {
        label: _union_polygons([_polygon_from_face(face) for face in faces])
        for label, faces in faces_by_label.items()
    }
    shared_by_pair = _pair_shared_lengths(evidence_payload)
    total_shared_by_label: Dict[int, float] = {}
    neighbor_labels_by_label: Dict[int, set[int]] = {}
    for (left, right), shared in shared_by_pair.items():
        total_shared_by_label[left] = float(total_shared_by_label.get(left, 0.0) + shared)
        total_shared_by_label[right] = float(total_shared_by_label.get(right, 0.0) + shared)
        neighbor_labels_by_label.setdefault(left, set()).add(right)
        neighbor_labels_by_label.setdefault(right, set()).add(left)
    max_neighbor_count = max([len(values) for values in neighbor_labels_by_label.values()] or [1])

    compactness_by_label = {label: _compactness(geometry) for label, geometry in geometry_by_label.items()}

    def divider_context_score(label: int) -> float:
        area = max(_geometry_area(geometry_by_label[label]), config.area_eps)
        shared_density = total_shared_by_label.get(label, 0.0) / area
        density_score = min(1.0, shared_density)
        neighbor_score = len(neighbor_labels_by_label.get(label, set())) / max(1, max_neighbor_count)
        compact_penalty = max(0.0, compactness_by_label.get(label, 0.0) - 0.25) * 1.5
        return float(max(0.0, 0.65 * density_score + 0.35 * neighbor_score - compact_penalty))
    pair_results: List[Dict[str, object]] = []
    cost_cache: Dict[Tuple[float, float, int], Dict[str, object]] = {}
    for left_label, right_label in sorted(shared_by_pair):
        shared_length = float(shared_by_pair[(left_label, right_label)])
        if shared_length <= config.min_shared_length:
            continue
        left_geometry = geometry_by_label[left_label]
        right_geometry = geometry_by_label[right_label]
        candidates: List[Dict[str, object]] = []
        for fill_policy in ("no_fill", "convex_hull_fill"):
            candidates.extend(
                [
                    _support_insert_candidate(left_label, right_label, left_geometry, right_geometry, fill_policy=fill_policy, config=config, cost_cache=cost_cache),
                    _support_insert_candidate(right_label, left_label, right_geometry, left_geometry, fill_policy=fill_policy, config=config, cost_cache=cost_cache),
                    _support_divider_candidate(
                        left_label,
                        right_label,
                        left_geometry,
                        right_geometry,
                        fill_policy=fill_policy,
                        divider_context_score=divider_context_score(right_label),
                        config=config,
                        cost_cache=cost_cache,
                    ),
                    _support_divider_candidate(
                        right_label,
                        left_label,
                        right_geometry,
                        left_geometry,
                        fill_policy=fill_policy,
                        divider_context_score=divider_context_score(left_label),
                        config=config,
                        cost_cache=cost_cache,
                    ),
                ]
            )
        candidates.append(_adjacent_supports_candidate(left_label, right_label, left_geometry, right_geometry, config=config, cost_cache=cost_cache))
        candidates.append(_independent_candidate(left_label, right_label, left_geometry, right_geometry, config=config, cost_cache=cost_cache))
        candidates.sort(key=lambda item: (float(item["cost"]), str(item["template"]), str(item["roles"])))
        selected = candidates[0]
        pair_results.append(
            {
                "labels": [int(left_label), int(right_label)],
                "shared_length": shared_length,
                "left_face_count": int(len(faces_by_label[left_label])),
                "right_face_count": int(len(faces_by_label[right_label])),
                "left_area": _geometry_area(left_geometry),
                "right_area": _geometry_area(right_geometry),
                "selected": selected,
                "candidates": candidates,
            }
        )

    preferred_roles: Dict[str, Dict[str, float]] = {}
    for pair in pair_results:
        for label, role in pair["selected"]["roles"].items():
            votes = preferred_roles.setdefault(str(label), {})
            votes[str(role)] = float(votes.get(str(role), 0.0) + float(pair["shared_length"]))
    preferred_role_by_label = {
        label: max(votes, key=lambda role: (votes[role], role))
        for label, votes in preferred_roles.items()
    }
    return {
        "format": "maskgen_pairwise_relation_analysis_v1",
        "source_evidence": source_tag,
        "size": evidence_payload.get("size", [0, 0]),
        "config": asdict(config),
        "pairs": pair_results,
        "preferred_role_votes_by_label": preferred_roles,
        "preferred_role_by_label": preferred_role_by_label,
        "statistics": {
            "label_count": int(len(faces_by_label)),
            "pair_count": int(len(pair_results)),
        },
    }
