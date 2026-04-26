from __future__ import annotations

from typing import Dict, Sequence, Set

from partition_gen.operation_types import (
    DIVIDE_BY_REGION,
    OVERLAY_INSERT,
    PARALLEL_SUPPORTS,
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
)


def polygon_code_length_from_payload(geometry_payload: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    polygons = geometry_payload.get("polygons_local")
    if not polygons:
        polygons = [
            {
                "outer_local": geometry_payload.get("outer_local", []),
                "holes_local": geometry_payload.get("holes_local", []),
            }
        ]
    total = 0
    component_count = 0
    outer_vertex_count = 0
    hole_count = 0
    hole_vertex_count = 0
    for polygon in polygons:
        component_count += 1
        outer = polygon.get("outer_local", [])
        holes = polygon.get("holes_local", [])
        outer_count = len(outer)
        hole_vertices = sum(len(ring) for ring in holes)
        outer_vertex_count += outer_count
        hole_count += len(holes)
        hole_vertex_count += hole_vertices
        total += config.token_polygon_component
        total += config.token_polygon_start
        total += config.token_polygon_vertex * outer_count
        total += config.token_polygon_hole * len(holes)
        total += config.token_polygon_vertex * hole_vertices
        total += config.token_polygon_end
    return {
        "total": int(total),
        "component_count": int(component_count),
        "outer_vertex_count": int(outer_vertex_count),
        "hole_count": int(hole_count),
        "hole_vertex_count": int(hole_vertex_count),
        "breakdown": {
            "polygon_start": int(config.token_polygon_start * component_count),
            "polygon_end": int(config.token_polygon_end * component_count),
            "component": int(config.token_polygon_component * component_count),
            "vertices": int(config.token_polygon_vertex * (outer_vertex_count + hole_vertex_count)),
            "holes": int(config.token_polygon_hole * hole_count),
        },
    }


def convex_atoms_code_length(node: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    atoms = node.get("atoms", [])
    atom_count = len(atoms)
    atom_vertex_count = sum(int(atom.get("vertex_count", len(atom.get("outer_local", [])))) for atom in atoms)
    total = sum(config.token_atom_start + config.token_atom_vertex * int(atom.get("vertex_count", len(atom.get("outer_local", [])))) for atom in atoms)
    return {
        "total": int(total),
        "atom_count": int(atom_count),
        "atom_vertex_count": int(atom_vertex_count),
        "breakdown": {
            "atom_start": int(config.token_atom_start * atom_count),
            "atom_vertices": int(config.token_atom_vertex * atom_vertex_count),
        },
    }


def geometry_code_length_for_node(node: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    model = str(node.get("geometry_model", ""))
    if model == "polygon_code":
        output = polygon_code_length_from_payload(node.get("geometry", {}), config)
        output["geometry_model"] = model
        return output
    if model == "convex_atoms":
        output = convex_atoms_code_length(node, config)
        output["geometry_model"] = model
        return output
    if model == "none":
        return {"total": 0, "geometry_model": model, "breakdown": {}}
    return {
        "total": int(config.token_missing_geometry_fallback),
        "geometry_model": model,
        "unknown_geometry_model": True,
        "breakdown": {"missing_geometry_fallback": int(config.token_missing_geometry_fallback)},
    }


def node_code_length(node: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    role = str(node.get("role", ""))
    role_token = int(config.token_group_node if role == "insert_object_group" else config.token_node)
    label_token = int(config.token_label if "label" in node and node.get("label") is not None else 0)
    geometry_model_token = int(config.token_geometry_model if "geometry_model" in node and node.get("geometry_model") is not None else 0)
    total = role_token + label_token + geometry_model_token
    return {
        "total": int(total),
        "role": role,
        "label_encoded": bool(label_token),
        "geometry_model_encoded": bool(geometry_model_token),
        "breakdown": {
            "role": int(role_token),
            "label": int(label_token),
            "geometry_model": int(geometry_model_token),
        },
    }


def _reference_count(value: object) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if value is None:
        return 0
    return 1


def relation_reference_counts(relation: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    scalar_semantic_keys = ("parent", "child", "object", "support", "divider", "owner", "residual", "atom", "face")
    list_semantic_keys = ("faces",)
    scalar_evidence_keys = ("source_face_id", "source_atom_id")
    list_evidence_keys = ("face_ids", "source_face_ids", "arc_ids", "atom_ids", "source_arc_ids")

    semantic_endpoint_count = 0
    evidence_reference_count = 0
    semantic_keys = []
    evidence_keys = []

    for key in scalar_semantic_keys + list_semantic_keys:
        count = _reference_count(relation.get(key)) if key in relation else 0
        if count:
            semantic_endpoint_count += count
            semantic_keys.append(key)

    for key in scalar_evidence_keys + list_evidence_keys:
        count = _reference_count(relation.get(key)) if key in relation else 0
        if count:
            evidence_reference_count += count
            evidence_keys.append(key)

    evidence = relation.get("evidence")
    if isinstance(evidence, dict):
        for key in scalar_evidence_keys + list_evidence_keys:
            count = _reference_count(evidence.get(key)) if key in evidence else 0
            if count:
                evidence_reference_count += count
                evidence_keys.append(f"evidence.{key}")

    encoded_evidence_reference_count = evidence_reference_count if config.token_encode_evidence_refs else 0
    return {
        "semantic_endpoint_count": int(semantic_endpoint_count),
        "evidence_reference_count": int(evidence_reference_count),
        "encoded_evidence_reference_count": int(encoded_evidence_reference_count),
        "semantic_keys": semantic_keys,
        "evidence_keys": evidence_keys,
    }


def relation_code_length(relation: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    counts = relation_reference_counts(relation, config)
    relation_type = int(config.token_relation_type)
    semantic_endpoints = int(config.token_relation_endpoint * int(counts["semantic_endpoint_count"]))
    evidence_refs = int(config.token_evidence_reference * int(counts["encoded_evidence_reference_count"]))
    return {
        "total": int(relation_type + semantic_endpoints + evidence_refs),
        "type": relation.get("type"),
        "endpoint_count": int(counts["semantic_endpoint_count"]),
        "semantic_endpoint_count": int(counts["semantic_endpoint_count"]),
        "evidence_reference_count": int(counts["evidence_reference_count"]),
        "encoded_evidence_reference_count": int(counts["encoded_evidence_reference_count"]),
        "semantic_keys": list(counts["semantic_keys"]),
        "evidence_keys": list(counts["evidence_keys"]),
        "breakdown": {
            "relation_type": int(relation_type),
            "semantic_endpoints": int(semantic_endpoints),
            "evidence_refs": int(evidence_refs),
        },
    }


def latent_policy_code_length(candidate: OperationCandidate, config: OperationExplainerConfig) -> Dict[str, object]:
    latent = candidate.metadata.get("latent_geometry") if candidate.metadata else None
    policy = latent.get("policy") if isinstance(latent, dict) else "none"
    mapping = {
        "visible_union": config.token_latent_visible_union,
        "union_with_children": config.token_latent_union_with_children,
        "union_with_divider": config.token_latent_union_with_divider,
        "convex_hull_fill": config.token_latent_convex_hull_fill,
        "buffer_close_fill": config.token_latent_buffer_close_fill,
        "none": 0,
    }
    total = int(mapping.get(str(policy), config.token_exception))
    return {
        "total": total,
        "policy": policy,
        "breakdown": {str(policy): total},
    }


def _face_polygon_payload(face: Dict[str, object]) -> Dict[str, object]:
    geometry = face.get("geometry") or {}
    return {
        "outer_local": geometry.get("outer", []),
        "holes_local": geometry.get("holes", []),
    }


def _face_convex_atom_node(face: Dict[str, object]) -> Dict[str, object]:
    atoms = []
    for atom in face.get("convex_partition", {}).get("atoms", []):
        atoms.append(
            {
                "vertex_count": int(atom.get("vertex_count", len(atom.get("outer", [])))),
                "outer_local": atom.get("outer", []),
            }
        )
    return {"geometry_model": "convex_atoms", "atoms": atoms}


def face_independent_code_length(face: Dict[str, object], config: OperationExplainerConfig) -> Dict[str, object]:
    semantic_face = int(config.token_node)
    polygon = polygon_code_length_from_payload(_face_polygon_payload(face), config) if config.independent_include_face_polygon else {"total": 0}
    atom_node = _face_convex_atom_node(face)
    atoms_available = bool(atom_node["atoms"])
    convex_atoms = convex_atoms_code_length(atom_node, config) if config.independent_include_convex_atoms and atoms_available else {"total": 0, "atom_count": 0, "atom_vertex_count": 0}
    fallback = bool(config.independent_include_convex_atoms and not atoms_available)
    fallback_length = int(config.token_missing_geometry_fallback if fallback else 0)
    total = semantic_face + int(polygon["total"]) + int(convex_atoms["total"]) + fallback_length
    return {
        "total": int(total),
        "semantic_face": semantic_face,
        "polygon": polygon,
        "convex_atoms": convex_atoms,
        "fallback": fallback,
        "fallback_length": fallback_length,
        "atom_count": int(convex_atoms.get("atom_count", 0)),
        "atom_vertex_count": int(convex_atoms.get("atom_vertex_count", 0)),
    }


def _adjacency_inside(evidence_payload: Dict[str, object], face_ids: Set[int]) -> int:
    count = 0
    for adjacency in evidence_payload.get("adjacency", []):
        faces = {int(value) for value in adjacency.get("faces", [])}
        if len(faces) == 2 and faces.issubset(face_ids):
            count += 1
    return int(count)


def independent_code_length_for_faces(
    face_ids: Sequence[int],
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    faces_by_id = {int(face["id"]): face for face in evidence_payload.get("faces", [])}
    faces = {}
    total = 0
    for face_id in sorted(set(int(value) for value in face_ids)):
        face = faces_by_id.get(face_id)
        if face is None:
            continue
        length = face_independent_code_length(face, config)
        faces[str(face_id)] = length
        total += int(length["total"])
    adjacency_count = _adjacency_inside(evidence_payload, set(int(value) for value in face_ids))
    adjacency_unit_length = int(config.token_relation_type + 2 * config.token_relation_endpoint)
    adjacency_length = int(adjacency_unit_length * adjacency_count)
    total += adjacency_length
    return {
        "total": int(total),
        "faces": faces,
        "adjacency_relation_count": int(adjacency_count),
        "adjacency_relation_length": int(adjacency_length),
        "adjacency_relation_unit_length": int(adjacency_unit_length),
    }


def _template_code_length(operation_type: str, config: OperationExplainerConfig) -> int:
    if operation_type == OVERLAY_INSERT:
        return int(config.token_template_overlay_insert)
    if operation_type == DIVIDE_BY_REGION:
        return int(config.token_template_divide_by_region)
    if operation_type == PARALLEL_SUPPORTS:
        return int(config.token_template_parallel_supports)
    if operation_type == RESIDUAL:
        return int(config.token_template_residual)
    return int(config.token_exception)


def operation_code_length(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    if candidate.operation_type == RESIDUAL:
        independent = independent_code_length_for_faces(candidate.covered_face_ids, evidence_payload, config)
        return {
            "total": int(independent["total"]),
            "template": int(config.token_template_residual),
            "nodes": 0,
            "node_count": int(len(candidate.nodes)),
            "geometry": 0,
            "relations": 0,
            "relation_count": 0,
            "latent_policy": 0,
            "residual": 0,
            "exception": 0,
            "breakdown": {
                "residual_matches_independent": True,
                "total": int(independent["total"]),
                "independent": independent,
                "note": "Residual operation is costed as the independent baseline so it does not create artificial compression gain.",
            },
        }

    template = _template_code_length(candidate.operation_type, config)
    node_lengths = [node_code_length(node, config) for node in candidate.nodes]
    geometry_lengths = [geometry_code_length_for_node(node, config) for node in candidate.nodes]
    relation_lengths = [relation_code_length(relation, config) for relation in candidate.relations]
    latent = latent_policy_code_length(candidate, config)
    residual = 0
    exception = 0 if candidate.valid else int(config.token_exception)
    total = (
        template
        + sum(int(item["total"]) for item in node_lengths)
        + sum(int(item["total"]) for item in geometry_lengths)
        + sum(int(item["total"]) for item in relation_lengths)
        + int(latent["total"])
        + residual
        + exception
    )
    return {
        "total": int(total),
        "template": int(template),
        "nodes": int(sum(int(item["total"]) for item in node_lengths)),
        "node_count": int(len(candidate.nodes)),
        "geometry": int(sum(int(item["total"]) for item in geometry_lengths)),
        "relations": int(sum(int(item["total"]) for item in relation_lengths)),
        "relation_count": int(len(candidate.relations)),
        "latent_policy": int(latent["total"]),
        "residual": int(residual),
        "exception": int(exception),
        "breakdown": {
            "nodes": node_lengths,
            "geometry": geometry_lengths,
            "relations": relation_lengths,
            "latent_policy": latent,
        },
    }
