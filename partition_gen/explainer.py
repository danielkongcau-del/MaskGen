from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from partition_gen.pairwise_relation_explainer import (
    PairwiseRelationConfig,
    build_pairwise_relation_payload,
)


Point = Tuple[float, float]


@dataclass(frozen=True)
class ExplainerConfig:
    max_role_candidates_per_face: int = 4
    residual_allowed: bool = True
    role_cost_margin: float = 0.0
    enable_pairwise_label_relations: bool = True
    enable_label_role_consistency: bool = True
    label_consistency_min_faces: int = 2
    label_consistency_penalty: float | None = None
    pair_relation_min_shared_length: float = 1e-6
    min_area_eps: float = 1e-8
    max_aspect_for_cost: float = 12.0
    cost_vertex: float = 0.35
    cost_relation: float = 1.0
    cost_object: float = 2.0
    cost_atom: float = 1.1
    cost_atom_vertex: float = 0.25


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
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
        return orient(Polygon(outer, holes), sign=1.0)
    except Exception:
        fixed = Polygon(outer, holes).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _normalise(value: float, maximum: float, *, eps: float) -> float:
    if maximum <= eps:
        return 0.0
    return float(max(0.0, min(1.0, value / maximum)))


def _safe_aspect(value: float, *, cap: float) -> float:
    if not math.isfinite(value):
        return cap
    return float(max(0.0, min(cap, value)))


def _role_costs(face: Dict[str, object], *, stats: Dict[str, float], config: ExplainerConfig) -> List[Dict[str, object]]:
    features = face["features"]
    convex = face.get("convex_partition", {})
    area = float(features.get("area", 0.0))
    area_norm = _normalise(area, stats["max_area"], eps=config.min_area_eps)
    degree_norm = _normalise(float(features.get("degree", 0)), stats["max_degree"], eps=config.min_area_eps)
    shared_density = float(features.get("shared_boundary_length", 0.0)) / max(area, 1.0)
    shared_density_norm = _normalise(shared_density, stats["max_shared_density"], eps=config.min_area_eps)
    aspect = _safe_aspect(float(features.get("oriented_aspect_ratio", 0.0)), cap=config.max_aspect_for_cost)
    aspect_norm = _normalise(aspect, config.max_aspect_for_cost, eps=config.min_area_eps)
    compactness = max(0.0, min(1.0, float(features.get("compactness", 0.0))))
    hole_count = int(features.get("hole_count", 0))
    vertex_count = len((face.get("geometry") or {}).get("outer", [])) + sum(
        len(ring) for ring in (face.get("geometry") or {}).get("holes", [])
    )
    atom_count = int(convex.get("piece_count", 0))
    atom_vertex_count = int(sum(int(atom.get("vertex_count", len(atom.get("outer", [])))) for atom in convex.get("atoms", [])))

    polygon_geometry_cost = config.cost_vertex * vertex_count + 1.25 * hole_count
    atoms_cost = config.cost_atom * atom_count + config.cost_atom_vertex * atom_vertex_count

    support_cost = (
        config.cost_object
        + polygon_geometry_cost
        - 4.0 * area_norm
        - 1.25 * degree_norm
        + 3.0 * aspect_norm
    )
    divider_cost = (
        config.cost_object
        + polygon_geometry_cost
        + 0.75 * area_norm
        - 4.5 * aspect_norm
        - 1.25 * degree_norm
        - 1.5 * shared_density_norm
    )
    insert_cost = (
        config.cost_object
        + polygon_geometry_cost
        + 3.0 * area_norm
        + 1.5 * aspect_norm
        - 2.5 * compactness
        + 0.75 * degree_norm
    )
    residual_cost = 1.5 * config.cost_object + atoms_cost

    candidates = [
        {
            "role": "support_region",
            "cost": float(support_cost),
            "cost_breakdown": {
                "object": config.cost_object,
                "geometry": float(polygon_geometry_cost),
                "area_bonus": float(-4.0 * area_norm),
                "degree_bonus": float(-1.25 * degree_norm),
                "thin_penalty": float(3.0 * aspect_norm),
            },
        },
        {
            "role": "divider_region",
            "cost": float(divider_cost),
            "cost_breakdown": {
                "object": config.cost_object,
                "geometry": float(polygon_geometry_cost),
                "area_penalty": float(0.75 * area_norm),
                "thin_bonus": float(-4.5 * aspect_norm),
                "degree_bonus": float(-1.25 * degree_norm),
                "shared_density_bonus": float(-1.5 * shared_density_norm),
            },
        },
        {
            "role": "insert_object",
            "cost": float(insert_cost),
            "cost_breakdown": {
                "object": config.cost_object,
                "geometry": float(polygon_geometry_cost),
                "area_penalty": float(3.0 * area_norm),
                "thin_penalty": float(1.5 * aspect_norm),
                "compactness_bonus": float(-2.5 * compactness),
                "degree_penalty": float(0.75 * degree_norm),
            },
        },
    ]
    if config.residual_allowed:
        candidates.append(
            {
                "role": "residual_region",
                "cost": float(residual_cost),
                "cost_breakdown": {
                    "object": 1.5 * config.cost_object,
                    "convex_atoms": float(atoms_cost),
                    "atom_count": int(atom_count),
                    "atom_vertex_count": int(atom_vertex_count),
                },
            }
        )
    candidates.sort(key=lambda item: (float(item["cost"]), str(item["role"])))
    return candidates


def _count_roles_by_label(faces: Sequence[Dict[str, object]], selected_roles: Dict[int, str]) -> Dict[str, Dict[str, int]]:
    output: Dict[str, Dict[str, int]] = {}
    for face in faces:
        label = str(int(face.get("label", -1)))
        role = str(selected_roles.get(int(face["id"]), "unassigned"))
        role_counts = output.setdefault(label, {})
        role_counts[role] = int(role_counts.get(role, 0)) + 1
    return output


def _label_aggregate_stats(
    faces: Sequence[Dict[str, object]],
    adjacency: Sequence[Dict[str, object]],
    *,
    config: ExplainerConfig,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    for face in faces:
        label = int(face.get("label", -1))
        features = face.get("features", {})
        area = float(features.get("area", 0.0))
        item = stats.setdefault(
            label,
            {
                "area": 0.0,
                "count": 0.0,
                "compactness_area": 0.0,
                "aspect_area": 0.0,
                "shared_length": 0.0,
            },
        )
        item["area"] += area
        item["count"] += 1.0
        item["compactness_area"] += float(features.get("compactness", 0.0)) * area
        item["aspect_area"] += _safe_aspect(float(features.get("oriented_aspect_ratio", 0.0)), cap=config.max_aspect_for_cost) * area
    for edge in adjacency:
        shared = float(edge.get("shared_length", 0.0))
        if shared <= config.pair_relation_min_shared_length:
            continue
        labels = [int(value) for value in edge.get("labels", [])]
        for label in labels:
            stats.setdefault(
                label,
                {
                    "area": 0.0,
                    "count": 0.0,
                    "compactness_area": 0.0,
                    "aspect_area": 0.0,
                    "shared_length": 0.0,
                },
            )["shared_length"] += shared
    for item in stats.values():
        area = max(float(item["area"]), config.min_area_eps)
        item["compactness"] = float(item["compactness_area"] / area)
        item["aspect"] = float(item["aspect_area"] / area)
        item["shared_density"] = float(item["shared_length"] / area)
    return stats


def _intrinsic_label_role_costs(
    label_stats: Dict[int, Dict[str, float]],
    *,
    config: ExplainerConfig,
) -> Dict[int, Dict[str, float]]:
    max_area = max([float(item["area"]) for item in label_stats.values()] or [0.0])
    max_shared_density = max([float(item["shared_density"]) for item in label_stats.values()] or [0.0])
    max_count = max([float(item["count"]) for item in label_stats.values()] or [1.0])
    costs: Dict[int, Dict[str, float]] = {}
    for label, item in label_stats.items():
        area_norm = _normalise(float(item["area"]), max_area, eps=config.min_area_eps)
        shared_norm = _normalise(float(item["shared_density"]), max_shared_density, eps=config.min_area_eps)
        count_norm = _normalise(math.log1p(float(item["count"])), math.log1p(max_count), eps=config.min_area_eps)
        compactness = max(0.0, min(1.0, float(item["compactness"])))
        aspect_norm = _normalise(_safe_aspect(float(item["aspect"]), cap=config.max_aspect_for_cost), config.max_aspect_for_cost, eps=config.min_area_eps)
        support = 2.0 + 2.5 * (1.0 - area_norm) + 0.5 * aspect_norm + 2.0 * compactness * count_norm
        divider = 2.0 + 2.0 * area_norm + 3.0 * (1.0 - shared_norm) + 0.75 * compactness - 0.5 * aspect_norm
        insert = 2.0 + 2.0 * area_norm + 2.0 * (1.0 - compactness) + 0.5 * aspect_norm - 1.25 * count_norm
        costs[label] = {
            "support_region": float(support),
            "divider_region": float(divider),
            "insert_object": float(insert),
            "residual_region": float(3.0 + 0.5 * area_norm),
        }
    return costs


def _select_label_pair_relations(
    faces: Sequence[Dict[str, object]],
    adjacency: Sequence[Dict[str, object]],
    *,
    config: ExplainerConfig,
) -> Dict[str, object]:
    if not config.enable_pairwise_label_relations:
        return {"enabled": False, "pairs": [], "preferred_roles_by_label": {}}
    pairwise_payload = build_pairwise_relation_payload(
        {"faces": list(faces), "adjacency": list(adjacency), "size": [0, 0]},
        config=PairwiseRelationConfig(
            min_shared_length=config.pair_relation_min_shared_length,
            area_eps=config.min_area_eps,
        ),
    )
    pair_summaries = [
        {
            "labels": pair["labels"],
            "shared_length": float(pair["shared_length"]),
            "selected_template": pair["selected"]["template"],
            "selected_fill_policy": pair["selected"]["fill_policy"],
            "selected_roles": pair["selected"]["roles"],
            "selected_cost": float(pair["selected"]["cost"]),
            "candidate_costs": [
                {
                    "template": candidate["template"],
                    "fill_policy": candidate["fill_policy"],
                    "roles": candidate["roles"],
                    "cost": float(candidate["cost"]),
                }
                for candidate in pair.get("candidates", [])
            ],
        }
        for pair in pairwise_payload.get("pairs", [])
    ]
    return {
        "enabled": True,
        "method": "binary_scene_completion_convex_cost",
        "pairs": pair_summaries,
        "role_votes_by_label": pairwise_payload.get("preferred_role_votes_by_label", {}),
        "preferred_roles_by_label": pairwise_payload.get("preferred_role_by_label", {}),
        "pairwise_statistics": pairwise_payload.get("statistics", {}),
        "pairwise_config": pairwise_payload.get("config", {}),
    }


def _select_roles_with_label_consistency(
    faces: Sequence[Dict[str, object]],
    role_candidates: Dict[int, List[Dict[str, object]]],
    *,
    config: ExplainerConfig,
    preferred_roles_by_label: Dict[str, str] | None = None,
) -> Tuple[Dict[int, str], Dict[str, object]]:
    preferred_roles_by_label = preferred_roles_by_label or {}
    local_roles = {
        int(face["id"]): str(role_candidates[int(face["id"])][0]["role"])
        for face in faces
        if role_candidates.get(int(face["id"]))
    }
    if not config.enable_label_role_consistency:
        return local_roles, {
            "enabled": False,
            "method": "local_argmin",
            "counts_before": _count_roles_by_label(faces, local_roles),
            "counts_after_consistency": _count_roles_by_label(faces, local_roles),
            "labels": {},
        }

    selected_roles = dict(local_roles)
    penalty = float(config.label_consistency_penalty if config.label_consistency_penalty is not None else config.cost_object * 4.0)
    faces_by_label: Dict[int, List[Dict[str, object]]] = {}
    for face in faces:
        faces_by_label.setdefault(int(face.get("label", -1)), []).append(face)

    labels_summary: Dict[str, object] = {}
    for label, label_faces in sorted(faces_by_label.items()):
        face_ids = [int(face["id"]) for face in label_faces]
        local_counts: Dict[str, int] = {}
        for face_id in face_ids:
            role = local_roles.get(face_id, "unassigned")
            local_counts[role] = int(local_counts.get(role, 0)) + 1

        if len(face_ids) < int(config.label_consistency_min_faces):
            labels_summary[str(label)] = {
                "dominant_role": None,
                "reason": "not_enough_faces",
                "face_count": int(len(face_ids)),
                "counts_before": local_counts,
                "counts_after_consistency": local_counts,
                "changed_face_ids": [],
                "override_face_ids": [],
            }
            continue

        role_names = sorted({str(candidate["role"]) for face_id in face_ids for candidate in role_candidates.get(face_id, [])})
        cost_by_role: Dict[str, float] = {}
        for role in role_names:
            total = 0.0
            valid = True
            for face_id in face_ids:
                candidate_costs = {str(candidate["role"]): float(candidate["cost"]) for candidate in role_candidates.get(face_id, [])}
                if role not in candidate_costs:
                    valid = False
                    break
                total += candidate_costs[role]
            if valid:
                cost_by_role[role] = float(total)
        if not cost_by_role:
            labels_summary[str(label)] = {
                "dominant_role": None,
                "reason": "no_shared_candidate_roles",
                "face_count": int(len(face_ids)),
                "counts_before": local_counts,
                "counts_after_consistency": local_counts,
                "changed_face_ids": [],
                "override_face_ids": [],
            }
            continue

        area_by_local_role: Dict[str, float] = {}
        for face in label_faces:
            face_id = int(face["id"])
            role = local_roles.get(face_id, "unassigned")
            area_by_local_role[role] = float(area_by_local_role.get(role, 0.0)) + float(face.get("features", {}).get("area", 0.0))
        preferred_role = preferred_roles_by_label.get(str(label))
        dominant_source = "pairwise_relation" if preferred_role in cost_by_role else "area_weighted_local_role"
        dominant_role = preferred_role if preferred_role in cost_by_role else max(
            area_by_local_role,
            key=lambda role: (
                area_by_local_role[role],
                -float(cost_by_role.get(role, float("inf"))),
                role,
            ),
        )
        if dominant_role not in cost_by_role:
            dominant_role = min(cost_by_role, key=lambda role: (cost_by_role[role], role))
        changed_face_ids: List[int] = []
        override_face_ids: List[int] = []
        after_counts: Dict[str, int] = {}
        for face_id in face_ids:
            adjusted = []
            for candidate in role_candidates.get(face_id, []):
                role = str(candidate["role"])
                adjusted_cost = float(candidate["cost"]) + (0.0 if role == dominant_role else penalty)
                adjusted.append((adjusted_cost, role))
            adjusted.sort(key=lambda item: (item[0], item[1]))
            role = adjusted[0][1]
            if role != local_roles.get(face_id):
                changed_face_ids.append(face_id)
            if role != dominant_role:
                override_face_ids.append(face_id)
            selected_roles[face_id] = role
            after_counts[role] = int(after_counts.get(role, 0)) + 1

        labels_summary[str(label)] = {
            "dominant_role": dominant_role,
            "method": "pairwise_preference_or_area_weighted_local_role_then_penalize_deviation",
            "dominant_source": dominant_source,
            "face_count": int(len(face_ids)),
            "penalty": penalty,
            "forced_total_cost_by_role": cost_by_role,
            "area_by_local_role": area_by_local_role,
            "counts_before": local_counts,
            "counts_after_consistency": after_counts,
            "changed_face_ids": changed_face_ids,
            "override_face_ids": override_face_ids,
            "override_count": int(len(override_face_ids)),
        }

    return selected_roles, {
        "enabled": True,
        "method": "image_level_label_role_consistency",
        "penalty": penalty,
        "min_faces": int(config.label_consistency_min_faces),
        "counts_before": _count_roles_by_label(faces, local_roles),
        "counts_after_consistency": _count_roles_by_label(faces, selected_roles),
        "labels": labels_summary,
    }


def _frame_from_polygon(polygon: Polygon, *, eps: float) -> Dict[str, float | List[float]]:
    if polygon.is_empty:
        return {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0}
    centroid = polygon.centroid
    rectangle = polygon.minimum_rotated_rectangle
    coords = _trim_ring(rectangle.exterior.coords)
    orientation = 0.0
    scale = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1], 1.0)
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


def _to_local(point: Sequence[float], frame: Dict[str, object]) -> List[float]:
    cx, cy = [float(value) for value in frame["origin"]]
    scale = max(float(frame["scale"]), 1e-8)
    theta = float(frame["orientation"])
    x = float(point[0]) - cx
    y = float(point[1]) - cy
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    return [float((x * cos_t - y * sin_t) / scale), float((x * sin_t + y * cos_t) / scale)]


def _ring_to_local(ring: Sequence[Sequence[float]], frame: Dict[str, object]) -> List[List[float]]:
    return [_to_local(point, frame) for point in ring]


def _holes_after_insert_fill(
    support_face: Dict[str, object],
    assigned_insert_faces: Sequence[Dict[str, object]],
) -> List[Sequence[Sequence[float]]]:
    holes = list((support_face.get("geometry") or {}).get("holes", []))
    if not holes or not assigned_insert_faces:
        return holes
    insert_polygons = [_polygon_from_face(face) for face in assigned_insert_faces]
    output = []
    for hole in holes:
        if len(hole) < 3:
            continue
        hole_polygon = Polygon(hole)
        matched_insert = any(
            not polygon.is_empty and hole_polygon.buffer(1e-7).covers(polygon.centroid)
            for polygon in insert_polygons
        )
        if not matched_insert:
            output.append(hole)
    return output


def _polygon_code_node_geometry(
    face: Dict[str, object],
    *,
    frame: Dict[str, object],
    holes_override: Sequence[Sequence[Sequence[float]]] | None = None,
) -> Dict[str, object]:
    geometry = face.get("geometry") or {}
    holes = holes_override if holes_override is not None else geometry.get("holes", [])
    return {
        "outer_local": _ring_to_local(geometry.get("outer", []), frame),
        "holes_local": [_ring_to_local(ring, frame) for ring in holes],
    }


def _residual_atoms(face: Dict[str, object], *, frame: Dict[str, object]) -> List[Dict[str, object]]:
    atoms = []
    for atom in face.get("convex_partition", {}).get("atoms", []):
        atoms.append(
            {
                "id": int(atom.get("id", len(atoms))),
                "type": atom.get("type", "convex"),
                "outer_local": _ring_to_local(atom.get("outer", []), frame),
                "vertex_count": int(atom.get("vertex_count", len(atom.get("outer", [])))),
                "area": float(atom.get("area", 0.0)),
            }
        )
    return atoms


def _adjacency_maps(evidence: Dict[str, object]) -> Tuple[Dict[int, List[Dict[str, object]]], Dict[Tuple[int, int], Dict[str, object]]]:
    by_face: Dict[int, List[Dict[str, object]]] = {}
    by_pair: Dict[Tuple[int, int], Dict[str, object]] = {}
    for item in evidence.get("adjacency", []):
        left, right = [int(value) for value in item.get("faces", [])]
        by_face.setdefault(left, []).append(item)
        by_face.setdefault(right, []).append(item)
        by_pair[(min(left, right), max(left, right))] = item
    return by_face, by_pair


def _best_adjacent_support(
    face_id: int,
    support_face_ids: set[int],
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
) -> int | None:
    candidates = []
    for item in adjacency_by_face.get(face_id, []):
        left, right = [int(value) for value in item["faces"]]
        other = right if left == face_id else left
        if other in support_face_ids:
            candidates.append((float(item.get("shared_length", 0.0)), other))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return int(candidates[0][1])


def _best_non_insert_role(candidates: Sequence[Dict[str, object]]) -> str:
    alternatives = [
        (float(candidate["cost"]), str(candidate["role"]))
        for candidate in candidates
        if str(candidate["role"]) != "insert_object"
    ]
    if not alternatives:
        return "residual_region"
    alternatives.sort(key=lambda item: (item[0], item[1]))
    return alternatives[0][1]


def _code_length(nodes: Sequence[Dict[str, object]], relations: Sequence[Dict[str, object]], residuals: Sequence[Dict[str, object]], *, config: ExplainerConfig) -> float:
    vertex_count = 0
    atom_count = 0
    for node in nodes:
        if node.get("geometry_model") == "polygon_code":
            geometry = node.get("geometry") or {}
            vertex_count += len(geometry.get("outer_local", []))
            vertex_count += sum(len(ring) for ring in geometry.get("holes_local", []))
        if node.get("geometry_model") == "convex_atoms":
            atoms = node.get("atoms", [])
            atom_count += len(atoms)
            vertex_count += sum(len(atom.get("outer_local", [])) for atom in atoms)
    return float(
        config.cost_object * len(nodes)
        + config.cost_relation * len(relations)
        + config.cost_atom * atom_count
        + config.cost_vertex * vertex_count
        + len(residuals)
    )


def build_explanation_payload(
    evidence_payload: Dict[str, object],
    *,
    config: ExplainerConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or ExplainerConfig()
    faces = sorted(evidence_payload.get("faces", []), key=lambda item: int(item["id"]))
    face_by_id = {int(face["id"]): face for face in faces}
    adjacency_by_face, _ = _adjacency_maps(evidence_payload)
    stats = {
        "max_area": max([float(face["features"].get("area", 0.0)) for face in faces] or [0.0]),
        "max_degree": max([float(face["features"].get("degree", 0.0)) for face in faces] or [0.0]),
        "max_shared_density": max(
            [
                float(face["features"].get("shared_boundary_length", 0.0)) / max(float(face["features"].get("area", 0.0)), 1.0)
                for face in faces
            ]
            or [0.0]
        ),
    }

    role_candidates: Dict[int, List[Dict[str, object]]] = {}
    for face in faces:
        face_id = int(face["id"])
        candidates = _role_costs(face, stats=stats, config=config)
        role_candidates[face_id] = candidates
    pairwise_relation_summary = _select_label_pair_relations(
        faces,
        evidence_payload.get("adjacency", []),
        config=config,
    )
    selected_roles, label_role_summary = _select_roles_with_label_consistency(
        faces,
        role_candidates,
        config=config,
        preferred_roles_by_label=pairwise_relation_summary.get("preferred_roles_by_label", {}),
    )
    label_role_summary["pairwise_relation_summary"] = pairwise_relation_summary

    for _ in range(3):
        support_face_ids = {face_id for face_id, role in selected_roles.items() if role == "support_region"}
        unparented_insert_ids = []
        for face_id, role in sorted(selected_roles.items()):
            if role != "insert_object":
                continue
            if _best_adjacent_support(face_id, support_face_ids, adjacency_by_face) is None:
                unparented_insert_ids.append(face_id)
        if not unparented_insert_ids:
            break
        for face_id in unparented_insert_ids:
            selected_roles[face_id] = _best_non_insert_role(role_candidates.get(face_id, []))

    support_face_ids = {face_id for face_id, role in selected_roles.items() if role == "support_region"}
    insert_face_ids = {face_id for face_id, role in selected_roles.items() if role == "insert_object"}
    divider_face_ids = {face_id for face_id, role in selected_roles.items() if role == "divider_region"}

    insert_parent: Dict[int, int] = {}
    for face_id in sorted(insert_face_ids):
        parent = _best_adjacent_support(face_id, support_face_ids, adjacency_by_face)
        if parent is not None:
            insert_parent[face_id] = parent
        else:
            selected_roles[face_id] = _best_non_insert_role(role_candidates.get(face_id, []))
    support_face_ids = {face_id for face_id, role in selected_roles.items() if role == "support_region"}
    insert_face_ids = set(insert_parent)
    divider_face_ids = {face_id for face_id, role in selected_roles.items() if role == "divider_region"}
    label_role_summary["counts_final"] = _count_roles_by_label(faces, selected_roles)
    for label, item in label_role_summary.get("labels", {}).items():
        if not isinstance(item, dict):
            continue
        item["counts_final"] = label_role_summary["counts_final"].get(label, {})

    assigned_inserts_by_support: Dict[int, List[Dict[str, object]]] = {}
    for insert_face_id, support_face_id in insert_parent.items():
        assigned_inserts_by_support.setdefault(support_face_id, []).append(face_by_id[insert_face_id])

    nodes: List[Dict[str, object]] = []
    relations: List[Dict[str, object]] = []
    residuals: List[Dict[str, object]] = []
    selected_explanations: List[Dict[str, object]] = []

    support_id_by_face: Dict[int, str] = {}
    for index, face_id in enumerate(sorted(support_face_ids, key=lambda item: (-face_by_id[item]["features"]["area"], item))):
        face = face_by_id[face_id]
        polygon = _polygon_from_face(face)
        frame = _frame_from_polygon(polygon, eps=config.min_area_eps)
        holes = _holes_after_insert_fill(face, assigned_inserts_by_support.get(face_id, []))
        node_id = f"support_{index}"
        support_id_by_face[face_id] = node_id
        nodes.append(
            {
                "id": node_id,
                "role": "support_region",
                "label": int(face["label"]),
                "frame": frame,
                "geometry_model": "polygon_code",
                "geometry": _polygon_code_node_geometry(face, frame=frame, holes_override=holes),
                "evidence": {"face_ids": [face_id], "arc_ids": _face_arc_ids(face)},
            }
        )

    divider_id_by_face: Dict[int, str] = {}
    for index, face_id in enumerate(sorted(divider_face_ids, key=lambda item: (-face_by_id[item]["features"]["shared_boundary_length"], item))):
        face = face_by_id[face_id]
        polygon = _polygon_from_face(face)
        frame = _frame_from_polygon(polygon, eps=config.min_area_eps)
        node_id = f"divider_{index}"
        divider_id_by_face[face_id] = node_id
        nodes.append(
            {
                "id": node_id,
                "role": "divider_region",
                "label": int(face["label"]),
                "frame": frame,
                "geometry_model": "polygon_code",
                "geometry": _polygon_code_node_geometry(face, frame=frame),
                "evidence": {"face_ids": [face_id], "arc_ids": _face_arc_ids(face)},
            }
        )
        support_neighbors = []
        for item in adjacency_by_face.get(face_id, []):
            left, right = [int(value) for value in item["faces"]]
            other = right if left == face_id else left
            if other in support_id_by_face:
                support_neighbors.append(other)
        for support_face_id in sorted(set(support_neighbors)):
            relations.append(
                {
                    "type": "divides",
                    "divider": node_id,
                    "support": support_id_by_face[support_face_id],
                    "evidence": {
                        "divider_face_id": face_id,
                        "support_face_id": support_face_id,
                    },
                }
            )

    insert_id_by_face: Dict[int, str] = {}
    group_key_to_children: Dict[Tuple[int, int], List[str]] = {}
    for index, face_id in enumerate(sorted(insert_face_ids, key=lambda item: (face_by_id[item]["features"]["centroid"][1], face_by_id[item]["features"]["centroid"][0], item))):
        face = face_by_id[face_id]
        polygon = _polygon_from_face(face)
        frame = _frame_from_polygon(polygon, eps=config.min_area_eps)
        node_id = f"insert_{index}"
        insert_id_by_face[face_id] = node_id
        support_face_id = insert_parent[face_id]
        group_key_to_children.setdefault((support_face_id, int(face["label"])), []).append(node_id)
        nodes.append(
            {
                "id": node_id,
                "role": "insert_object",
                "label": int(face["label"]),
                "support_id": support_id_by_face[support_face_id],
                "frame": frame,
                "geometry_model": "polygon_code",
                "geometry": _polygon_code_node_geometry(face, frame=frame),
                "evidence": {"face_ids": [face_id], "arc_ids": _face_arc_ids(face)},
            }
        )

    insert_group_nodes: List[Dict[str, object]] = []
    for group_index, ((support_face_id, label), children) in enumerate(
        sorted(group_key_to_children.items(), key=lambda item: (support_id_by_face[item[0][0]], item[0][1]))
    ):
        group_id = f"insert_group_{group_index}"
        insert_group_nodes.append(
            {
                "id": group_id,
                "role": "insert_object_group",
                "support_id": support_id_by_face[support_face_id],
                "label": int(label),
                "count": int(len(children)),
                "children": list(children),
                "evidence": {
                    "face_ids": [int(_face_id_from_insert_id(child, insert_id_by_face)) for child in children],
                },
            }
        )
        relations.append({"type": "contains", "parent": support_id_by_face[support_face_id], "child": group_id})
        for child in children:
            _set_insert_parent_group(nodes, child, group_id)
            relations.append({"type": "contains", "parent": group_id, "child": child})
            relations.append({"type": "inserted_in", "object": child, "support": support_id_by_face[support_face_id]})

    if insert_group_nodes:
        insert_start = next((idx for idx, node in enumerate(nodes) if node.get("role") == "insert_object"), len(nodes))
        nodes = nodes[:insert_start] + insert_group_nodes + nodes[insert_start:]

    residual_face_ids = {face_id for face_id, role in selected_roles.items() if role == "residual_region"}
    for index, face_id in enumerate(sorted(residual_face_ids, key=lambda item: (-face_by_id[item]["features"]["area"], item))):
        face = face_by_id[face_id]
        polygon = _polygon_from_face(face)
        frame = _frame_from_polygon(polygon, eps=config.min_area_eps)
        atoms = _residual_atoms(face, frame=frame)
        node_id = f"residual_{index}"
        node = {
            "id": node_id,
            "role": "residual_region",
            "label": int(face["label"]),
            "frame": frame,
            "geometry_model": "convex_atoms" if atoms else "polygon_code",
            "reason": "lowest_cost_or_unparented_baseline",
            "evidence": {"face_ids": [face_id], "arc_ids": _face_arc_ids(face)},
        }
        if atoms:
            node["atoms"] = atoms
        else:
            node["geometry"] = _polygon_code_node_geometry(face, frame=frame)
        nodes.append(node)
        residuals.append({"node_id": node_id, "face_ids": [face_id], "area": float(face["features"]["area"])})

    selected_explanations.extend(_selected_explanations_for_groups(group_key_to_children, support_id_by_face, insert_id_by_face, face_by_id))
    selected_explanations.extend(_selected_explanations_for_dividers(divider_id_by_face, support_id_by_face, adjacency_by_face))
    selected_explanations.extend(_selected_explanations_for_residuals(residuals))

    role_histogram: Dict[str, int] = {}
    for node in nodes:
        role = str(node.get("role"))
        role_histogram[role] = role_histogram.get(role, 0) + 1
    residual_area = float(sum(item["area"] for item in residuals))
    total_area = float(evidence_payload.get("statistics", {}).get("total_face_area", 0.0))
    validation = _validate_parse_graph(nodes, relations)
    code_length = _code_length(nodes, relations, residuals, config=config)
    generator_target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": evidence_payload.get("size", [0, 0]),
        "parse_graph": {
            "nodes": nodes,
            "relations": relations,
            "residuals": residuals,
        },
        "metadata": {
            "source_explanation": source_tag,
            "code_length": code_length,
            "render_iou": None,
            "valid": bool(validation["is_valid"]),
            "render_validation": {
                "status": "not_implemented",
                "reason": "parse_graph renderer is not implemented in the initial explainer.",
            },
        },
    }
    diagnostics = {
        "face_count": int(len(faces)),
        "candidate_count": int(sum(len(values) for values in role_candidates.values())),
        "selected_candidate_count": int(len(selected_explanations)),
        "residual_face_count": int(len(residuals)),
        "residual_area_ratio": float(residual_area / total_area) if total_area > config.min_area_eps else 0.0,
        "total_code_length": code_length,
        "selection_method": (
            "pairwise_relation_role_cost_with_label_consistency"
            if pairwise_relation_summary.get("enabled") and label_role_summary.get("enabled")
            else ("role_cost_with_label_consistency" if label_role_summary.get("enabled") else "role_cost_baseline")
        ),
        "global_optimal": False,
        "role_histogram": role_histogram,
        "label_role_summary": label_role_summary,
        "failure_reasons": [],
        "role_candidates": {
            str(face_id): values[: config.max_role_candidates_per_face]
            for face_id, values in role_candidates.items()
        },
    }
    return {
        "format": "maskgen_explanation_v1",
        "source_evidence": source_tag,
        "selected_explanations": selected_explanations,
        "generator_target": generator_target,
        "diagnostics": diagnostics,
        "validation": validation,
        "config": asdict(config),
    }


def _face_arc_ids(face: Dict[str, object]) -> List[int]:
    arc_ids = []
    for ref in face.get("outer_arc_refs", []):
        arc_ids.append(int(ref["arc_id"]))
    for refs in face.get("hole_arc_refs", []):
        for ref in refs:
            arc_ids.append(int(ref["arc_id"]))
    return sorted(set(arc_ids))


def _face_id_from_insert_id(insert_id: str, insert_id_by_face: Dict[int, str]) -> int:
    for face_id, node_id in insert_id_by_face.items():
        if node_id == insert_id:
            return int(face_id)
    return -1


def _set_insert_parent_group(nodes: List[Dict[str, object]], insert_id: str, group_id: str) -> None:
    for node in nodes:
        if node.get("id") == insert_id:
            node["parent_group"] = group_id
            return


def _selected_explanations_for_groups(
    group_key_to_children: Dict[Tuple[int, int], List[str]],
    support_id_by_face: Dict[int, str],
    insert_id_by_face: Dict[int, str],
    face_by_id: Dict[int, Dict[str, object]],
) -> List[Dict[str, object]]:
    output = []
    child_to_face = {node_id: face_id for face_id, node_id in insert_id_by_face.items()}
    for index, ((support_face_id, _label), children) in enumerate(sorted(group_key_to_children.items())):
        face_ids = [int(support_face_id), *[int(child_to_face[child]) for child in children]]
        output.append(
            {
                "patch_id": f"support_insert_patch_{index}",
                "evidence": {"face_ids": face_ids, "arc_ids": sorted(set(sum((_face_arc_ids(face_by_id[face_id]) for face_id in face_ids), [])))},
                "selected_candidate_id": "candidate_0",
                "score_gap": None,
                "selected_template": "support_with_inserts",
                "generated_node_ids": [support_id_by_face[support_face_id], *children],
                "generated_relation_ids": [],
                "cost": {"total": None, "template": 1.0, "topology": len(children) + 1, "geometry": None, "residual": 0.0, "invalid": 0.0},
            }
        )
    return output


def _selected_explanations_for_dividers(
    divider_id_by_face: Dict[int, str],
    support_id_by_face: Dict[int, str],
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    output = []
    for index, (divider_face_id, divider_node_id) in enumerate(sorted(divider_id_by_face.items())):
        support_nodes = []
        support_face_ids = []
        for item in adjacency_by_face.get(divider_face_id, []):
            left, right = [int(value) for value in item["faces"]]
            other = right if left == divider_face_id else left
            if other in support_id_by_face:
                support_face_ids.append(other)
                support_nodes.append(support_id_by_face[other])
        if not support_nodes:
            continue
        output.append(
            {
                "patch_id": f"divider_patch_{index}",
                "evidence": {"face_ids": [int(divider_face_id), *sorted(set(support_face_ids))], "arc_ids": []},
                "selected_candidate_id": "candidate_0",
                "score_gap": None,
                "selected_template": "split_by_divider",
                "generated_node_ids": [divider_node_id, *sorted(set(support_nodes))],
                "generated_relation_ids": [],
                "cost": {"total": None, "template": 1.0, "topology": len(set(support_nodes)) + 1, "geometry": None, "residual": 0.0, "invalid": 0.0},
            }
        )
    return output


def _selected_explanations_for_residuals(residuals: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for index, residual in enumerate(residuals):
        output.append(
            {
                "patch_id": f"residual_patch_{index}",
                "evidence": {"face_ids": list(residual.get("face_ids", [])), "arc_ids": []},
                "selected_candidate_id": "candidate_0",
                "score_gap": None,
                "selected_template": "independent_faces",
                "generated_node_ids": [residual["node_id"]],
                "generated_relation_ids": [],
                "cost": {"total": None, "template": 1.0, "topology": 1, "geometry": None, "residual": float(residual.get("area", 0.0)), "invalid": 0.0},
            }
        )
    return output


def _validate_parse_graph(nodes: Sequence[Dict[str, object]], relations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    node_ids = {str(node["id"]) for node in nodes}
    missing_refs: List[Dict[str, object]] = []
    for relation in relations:
        refs = []
        for key in ("parent", "child", "object", "support", "divider", "owner", "residual"):
            if key in relation:
                refs.append((key, str(relation[key])))
        for key, value in refs:
            if value not in node_ids:
                missing_refs.append({"relation": relation, "field": key, "value": value})
    return {
        "is_valid": bool(not missing_refs),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "missing_relation_refs": missing_refs,
        "render_validation": {
            "status": "not_implemented",
        },
    }
