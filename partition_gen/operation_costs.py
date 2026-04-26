from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Set

from partition_gen.operation_code_length import independent_code_length_for_faces, operation_code_length
from partition_gen.operation_false_cover import compute_false_cover_diagnostics, compute_false_cover_penalty
from partition_gen.operation_geometry import polygon_from_face, vertex_count_for_face, vertex_count_for_geometry
from partition_gen.operation_types import (
    DIVIDE_BY_REGION,
    OVERLAY_INSERT,
    PARALLEL_SUPPORTS,
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
)


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _adjacency_inside(evidence_payload: Dict[str, object], face_ids: Set[int]) -> int:
    count = 0
    for adjacency in evidence_payload.get("adjacency", []):
        faces = {int(value) for value in adjacency.get("faces", [])}
        if len(faces) == 2 and faces.issubset(face_ids):
            count += 1
    return int(count)


def _atom_stats(face: Dict[str, object]) -> tuple[int, int]:
    atoms = face.get("convex_partition", {}).get("atoms", [])
    atom_count = len(atoms)
    atom_vertex_count = sum(int(atom.get("vertex_count", len(atom.get("outer", [])))) for atom in atoms)
    return int(atom_count), int(atom_vertex_count)


def _face_independent_cost(face: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, float]:
    vertex_count = vertex_count_for_face(face)
    atom_count, atom_vertex_count = _atom_stats(face)
    polygon_cost = config.cost_vertex * vertex_count
    atom_cost = config.cost_atom * atom_count + config.cost_atom_vertex * atom_vertex_count
    if atom_count == 0:
        atom_cost += polygon_cost
    total = config.cost_object + polygon_cost + atom_cost
    return {
        "total": float(total),
        "object": float(config.cost_object),
        "polygon_vertices": float(polygon_cost),
        "atoms": float(atom_cost),
        "vertex_count": float(vertex_count),
        "atom_count": float(atom_count),
        "atom_vertex_count": float(atom_vertex_count),
    }


def heuristic_independent_cost_for_faces(
    face_ids: Sequence[int],
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    faces_by_id = _faces_by_id(evidence_payload)
    face_costs = {}
    total = 0.0
    for face_id in sorted(set(int(value) for value in face_ids)):
        face = faces_by_id.get(face_id)
        if face is None:
            continue
        cost = _face_independent_cost(face, config)
        face_costs[str(face_id)] = cost
        total += float(cost["total"])
    adjacency_count = _adjacency_inside(evidence_payload, set(int(value) for value in face_ids))
    relation_cost = config.cost_relation * adjacency_count
    total += relation_cost
    return {
        "total": float(total),
        "faces": face_costs,
        "adjacency_relation_count": int(adjacency_count),
        "adjacency_relation_cost": float(relation_cost),
    }


def _geometry_cost_for_node(node: Dict[str, object], config: OperationExplainerConfig) -> float:
    model = str(node.get("geometry_model", ""))
    if model == "none":
        return 0.0
    if model == "convex_atoms":
        atoms = node.get("atoms", [])
        atom_vertices = sum(int(atom.get("vertex_count", len(atom.get("outer_local", [])))) for atom in atoms)
        return float(config.cost_atom * len(atoms) + config.cost_atom_vertex * atom_vertices)
    geometry = node.get("geometry", {})
    if model == "polygon_code":
        polygons = geometry.get("polygons_local")
        if polygons:
            vertex_count = 0
            for polygon in polygons:
                vertex_count += len(polygon.get("outer_local", []))
                vertex_count += sum(len(ring) for ring in polygon.get("holes_local", []))
        else:
            vertex_count = len(geometry.get("outer_local", [])) + sum(len(ring) for ring in geometry.get("holes_local", []))
        return float(config.cost_vertex * vertex_count)
    return 0.0


def node_object_cost(node: Dict[str, object], config: OperationExplainerConfig) -> float:
    role = str(node.get("role", ""))
    if role == "insert_object_group":
        return float(config.cost_group_object)
    if role == "support_region":
        return float(config.cost_node_support if config.cost_node_support is not None else config.cost_object)
    if role == "divider_region":
        return float(config.cost_node_divider if config.cost_node_divider is not None else config.cost_object)
    if role == "insert_object":
        return float(config.cost_node_insert if config.cost_node_insert is not None else config.cost_object)
    if role == "residual_region":
        return float(config.cost_node_residual if config.cost_node_residual is not None else config.cost_object)
    return float(config.cost_node_default if config.cost_node_default is not None else config.cost_object)


def _template_cost(operation_type: str, config: OperationExplainerConfig) -> float:
    if operation_type == OVERLAY_INSERT:
        return float(config.cost_template_overlay_insert)
    if operation_type == DIVIDE_BY_REGION:
        return float(config.cost_template_divide_by_region)
    if operation_type == PARALLEL_SUPPORTS:
        return float(config.cost_template_parallel_supports)
    if operation_type == RESIDUAL:
        return float(config.cost_template_residual)
    return float(config.invalid_cost)


def heuristic_operation_cost(candidate: OperationCandidate, config: OperationExplainerConfig, false_cover: Dict[str, object]) -> Dict[str, object]:
    if candidate.operation_type == RESIDUAL:
        return {
            "total": 0.0,
            "template": 0.0,
            "node_object": 0.0,
            "node_count": 0,
            "group_node_count": 0,
            "geometry": 0.0,
            "relations": 0.0,
            "latent_geometry": 0.0,
            "false_cover": 0.0,
            "residual": 0.0,
            "invalid": 0.0,
        }
    template = _template_cost(candidate.operation_type, config)
    node_object = sum(node_object_cost(node, config) for node in candidate.nodes)
    group_node_count = sum(1 for node in candidate.nodes if node.get("role") == "insert_object_group")
    geometry = sum(_geometry_cost_for_node(node, config) for node in candidate.nodes)
    relations = config.cost_relation * len(candidate.relations)
    latent = candidate.metadata.get("latent_geometry", {}) if candidate.metadata else {}
    latent_extra = float(latent.get("extra_cost", 0.0)) if isinstance(latent, dict) else 0.0
    false_cover_cost = float(false_cover.get("cost", 0.0))
    metadata_penalty = 0.0
    if candidate.metadata:
        metadata_penalty += float(candidate.metadata.get("support_label_diversity_penalty", 0.0))
        metadata_penalty += float(candidate.metadata.get("label_pair_consistency_penalty", 0.0))
    residual = sum(float(item.get("area", 0.0)) * config.cost_residual_area for item in candidate.residuals)
    invalid = 0.0 if candidate.valid else config.invalid_cost
    total = template + node_object + geometry + relations + latent_extra + false_cover_cost + metadata_penalty + residual + invalid
    return {
        "total": float(total),
        "template": float(template),
        "node_object": float(node_object),
        "node_count": int(len(candidate.nodes)),
        "group_node_count": int(group_node_count),
        "geometry": float(geometry),
        "relations": float(relations),
        "latent_geometry": float(latent_extra),
        "false_cover": float(false_cover_cost),
        "metadata_penalty": float(metadata_penalty),
        "residual": float(residual),
        "invalid": float(invalid),
    }


def _geometry_validity(candidate: OperationCandidate, evidence_payload: Dict[str, object], config: OperationExplainerConfig) -> tuple[bool, str | None]:
    faces_by_id = _faces_by_id(evidence_payload)
    for face_id in candidate.covered_face_ids:
        if face_id not in faces_by_id:
            return False, f"missing_face:{face_id}"
    if candidate.operation_type == RESIDUAL:
        return True, None
    if (
        config.enable_label_pair_consistency
        and config.hard_enforce_label_pair_consistency
        and candidate.metadata
        and isinstance(candidate.metadata.get("label_pair_consistency"), dict)
        and bool(candidate.metadata["label_pair_consistency"].get("hard_inconsistent", False))
    ):
        return False, "label_pair_inconsistent"
    if not candidate.nodes:
        return False, "missing_nodes"
    if candidate.metadata:
        latent = candidate.metadata.get("latent_geometry")
        if isinstance(latent, dict) and not latent.get("valid", True):
            return False, str(latent.get("failure_reason") or "invalid_latent_geometry")
    for face_id in candidate.covered_face_ids:
        polygon = polygon_from_face(faces_by_id[face_id])
        if polygon.is_empty or polygon.area <= config.min_area_eps:
            return False, f"empty_face_geometry:{face_id}"
    return True, None


def score_operation_candidate_heuristic(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> OperationCandidate:
    independent = heuristic_independent_cost_for_faces(candidate.covered_face_ids, evidence_payload, config)
    valid, failure_reason = _geometry_validity(candidate, evidence_payload, config)
    valid = bool(candidate.valid and valid)
    failure_reason = candidate.failure_reason or failure_reason
    false_cover = compute_false_cover_penalty(candidate, evidence_payload, config)
    if bool(false_cover.get("hard_invalid", False)):
        valid = False
        failure_reason = failure_reason or "excessive_false_cover"
    operation = heuristic_operation_cost(replace(candidate, valid=valid), config, false_cover)
    if candidate.operation_type == RESIDUAL:
        geometry = sum(_geometry_cost_for_node(node, config) for node in candidate.nodes)
        node_object = sum(node_object_cost(node, config) for node in candidate.nodes)
        template = float(config.cost_template_residual)
        total = float(independent["total"])
        operation = {
            "total": total,
            "template": template,
            "node_object": float(node_object),
            "node_count": int(len(candidate.nodes)),
            "group_node_count": int(sum(1 for node in candidate.nodes if node.get("role") == "insert_object_group")),
            "geometry": float(geometry),
            "relations": 0.0,
            "latent_geometry": 0.0,
            "false_cover": 0.0,
            "residual": max(0.0, total - template - node_object - geometry),
            "invalid": 0.0,
        }
    compression_gain = float(independent["total"]) - float(operation["total"])
    breakdown = {
        "cost_profile": "heuristic_v1",
        "independent": independent,
        "operation": operation,
        "latent_geometry": _serializable_latent(candidate),
        "false_cover": false_cover,
        "invalid": float(0.0 if valid else config.invalid_cost),
        "compression_gain": float(compression_gain),
    }
    return replace(
        candidate,
        independent_cost=float(independent["total"]),
        operation_cost=float(operation["total"]),
        compression_gain=float(compression_gain),
        cost_breakdown=breakdown,
        valid=valid,
        failure_reason=failure_reason,
    )


def token_independent_cost_for_faces(
    face_ids: Sequence[int],
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    return independent_code_length_for_faces(face_ids, evidence_payload, config)


def token_operation_cost(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    return operation_code_length(candidate, evidence_payload, config)


def score_operation_candidate_token_length(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> OperationCandidate:
    independent = token_independent_cost_for_faces(candidate.covered_face_ids, evidence_payload, config)
    valid, failure_reason = _geometry_validity(candidate, evidence_payload, config)
    valid = bool(candidate.valid and valid)
    failure_reason = candidate.failure_reason or failure_reason
    false_cover = compute_false_cover_diagnostics(candidate, evidence_payload, config)
    if bool(false_cover.get("hard_invalid", False)):
        valid = False
        failure_reason = failure_reason or "false_cover"
    scored_candidate = replace(candidate, valid=valid)
    operation = token_operation_cost(scored_candidate, evidence_payload, config)
    compression_gain = int(independent["total"]) - int(operation["total"])
    breakdown = {
        "cost_profile": "token_length_v1",
        "independent": independent,
        "operation": operation,
        "latent_geometry": _serializable_latent(candidate),
        "false_cover": false_cover,
        "invalid": int(0 if valid else config.token_exception),
        "compression_gain": int(compression_gain),
    }
    return replace(
        candidate,
        independent_cost=float(independent["total"]),
        operation_cost=float(operation["total"]),
        compression_gain=float(compression_gain),
        cost_breakdown=breakdown,
        valid=valid,
        failure_reason=failure_reason,
    )


def score_operation_candidate(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> OperationCandidate:
    if config.cost_profile == "heuristic_v1":
        return score_operation_candidate_heuristic(candidate, evidence_payload, config)
    if config.cost_profile == "token_length_v1":
        return score_operation_candidate_token_length(candidate, evidence_payload, config)
    raise ValueError(f"Unsupported operation cost_profile: {config.cost_profile}")


independent_cost_for_faces = heuristic_independent_cost_for_faces


def _serializable_latent(candidate: OperationCandidate) -> Dict[str, object]:
    if not candidate.metadata:
        return {}
    latent = candidate.metadata.get("latent_geometry")
    if not isinstance(latent, dict):
        return {}
    geometry = latent.get("geometry")
    return {
        "policy": latent.get("policy"),
        "extra_cost": latent.get("extra_cost"),
        "valid": latent.get("valid"),
        "failure_reason": latent.get("failure_reason"),
        "area": float(geometry.area) if geometry is not None and not geometry.is_empty else 0.0,
        "vertex_count": vertex_count_for_geometry(geometry) if geometry is not None else 0,
    }


def score_operation_candidates(
    candidates: Sequence[OperationCandidate],
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> List[OperationCandidate]:
    return [score_operation_candidate(candidate, evidence_payload, config) for candidate in candidates]
