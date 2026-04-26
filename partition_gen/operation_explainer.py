from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

from partition_gen.explainer import ExplainerConfig, build_explanation_payload
from partition_gen.operation_candidates import propose_operation_candidates_with_diagnostics
from partition_gen.operation_costs import score_operation_candidates
from partition_gen.operation_patches import build_operation_patches
from partition_gen.operation_proxy_validation import validate_selected_operations_proxy
from partition_gen.operation_selector import select_operations_with_ortools
from partition_gen.operation_types import (
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
    OperationSelectionResult,
)
from partition_gen.pairwise_relation_explainer import PairwiseRelationConfig, build_pairwise_relation_payload
from partition_gen.weak_explainer import WeakExplainerConfig, build_weak_explanation_payload


def _face_ids(evidence_payload: Dict[str, object]) -> List[int]:
    return sorted(int(face["id"]) for face in evidence_payload.get("faces", []))


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _sanitize_metadata(metadata: Dict[str, object] | None) -> Dict[str, object]:
    if not metadata:
        return {}
    output: Dict[str, object] = {}
    for key, value in metadata.items():
        if key == "latent_geometry" and isinstance(value, dict):
            geometry = value.get("geometry")
            output[key] = {
                "policy": value.get("policy"),
                "extra_cost": value.get("extra_cost"),
                "valid": value.get("valid"),
                "failure_reason": value.get("failure_reason"),
                "area": float(geometry.area) if geometry is not None and not geometry.is_empty else 0.0,
            }
        else:
            output[key] = value
    return output


def _candidate_public(candidate: OperationCandidate) -> Dict[str, object]:
    return {
        "id": candidate.id,
        "operation_type": candidate.operation_type,
        "patch_id": candidate.patch_id,
        "covered_face_ids": list(candidate.covered_face_ids),
        "evidence_arc_ids": list(candidate.evidence_arc_ids),
        "independent_cost": float(candidate.independent_cost),
        "operation_cost": float(candidate.operation_cost),
        "compression_gain": float(candidate.compression_gain),
        "valid": bool(candidate.valid),
        "failure_reason": candidate.failure_reason,
        "metadata": _sanitize_metadata(candidate.metadata),
        "latent_policy": (
            candidate.metadata.get("latent_geometry", {}).get("policy")
            if candidate.metadata and isinstance(candidate.metadata.get("latent_geometry"), dict)
            else None
        ),
        "false_cover": candidate.cost_breakdown.get("false_cover", {}) if candidate.cost_breakdown else {},
        "cost_breakdown": candidate.cost_breakdown,
    }


def _relation_refs(relation: Dict[str, object]) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    for key in ("parent", "child", "object", "support", "divider", "owner", "residual"):
        if key in relation:
            refs.append((key, str(relation[key])))
    for key in ("faces",):
        if key in relation:
            for value in relation.get(key, []):
                refs.append((key, str(value)))
    return refs


def _rewrite_value(value, node_id_map: Dict[str, str]):
    if isinstance(value, str):
        return node_id_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_value(item, node_id_map) for item in value]
    return value


def _role_prefix(role: str) -> str:
    return {
        "support_region": "support",
        "divider_region": "divider",
        "insert_object_group": "insert_group",
        "insert_object": "insert",
        "residual_region": "residual",
    }.get(role, "node")


def _normalise_selected_parse_graph(
    selected_candidates: Sequence[OperationCandidate],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[str]], Dict[str, List[str]]]:
    nodes: List[Dict[str, object]] = []
    relations: List[Dict[str, object]] = []
    residuals: List[Dict[str, object]] = []
    generated_nodes_by_candidate: Dict[str, List[str]] = {}
    generated_relations_by_candidate: Dict[str, List[str]] = {}
    counters: Dict[str, int] = {}

    for candidate in selected_candidates:
        node_id_map: Dict[str, str] = {}
        generated_node_ids: List[str] = []
        for source_node in candidate.nodes:
            role = str(source_node.get("role", "node"))
            prefix = _role_prefix(role)
            index = counters.get(prefix, 0)
            counters[prefix] = index + 1
            new_id = f"{prefix}_{index}"
            node_id_map[str(source_node["id"])] = new_id
            node = dict(source_node)
            node["id"] = new_id
            nodes.append(node)
            generated_node_ids.append(new_id)

        # Rewrite references after all nodes in this candidate are mapped.
        for node in nodes[-len(candidate.nodes) :]:
            for key in ("children", "atom_ids"):
                if key in node:
                    node[key] = _rewrite_value(node[key], node_id_map)
            for key in ("support_id", "parent_group", "parent_face"):
                if key in node:
                    node[key] = _rewrite_value(node[key], node_id_map)

        generated_relation_ids: List[str] = []
        for source_relation in candidate.relations:
            relation = {key: _rewrite_value(value, node_id_map) for key, value in source_relation.items()}
            relation_id = f"relation_{len(relations)}"
            relation["id"] = relation_id
            relations.append(relation)
            generated_relation_ids.append(relation_id)

        for source_residual in candidate.residuals:
            residual = {key: _rewrite_value(value, node_id_map) for key, value in source_residual.items()}
            residuals.append(residual)

        generated_nodes_by_candidate[candidate.id] = generated_node_ids
        generated_relations_by_candidate[candidate.id] = generated_relation_ids

    return nodes, relations, residuals, generated_nodes_by_candidate, generated_relations_by_candidate


def _validate(
    evidence_payload: Dict[str, object],
    selected_candidates: Sequence[OperationCandidate],
    nodes: Sequence[Dict[str, object]],
    relations: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    all_face_ids = set(_face_ids(evidence_payload))
    coverage: Dict[int, int] = {face_id: 0 for face_id in all_face_ids}
    for candidate in selected_candidates:
        for face_id in candidate.covered_face_ids:
            coverage[int(face_id)] = int(coverage.get(int(face_id), 0)) + 1
    all_faces_covered_once = bool(all(coverage.get(face_id, 0) == 1 for face_id in all_face_ids))
    node_ids = {str(node["id"]) for node in nodes}
    missing_refs = []
    for relation in relations:
        for field, value in _relation_refs(relation):
            if value not in node_ids:
                missing_refs.append({"relation": relation.get("id"), "field": field, "value": value})
    evidence_validation = evidence_payload.get("evidence_validation", {})
    input_valid = bool(evidence_validation.get("usable_for_explainer", evidence_validation.get("is_valid", True)))
    return {
        "is_valid": bool(input_valid and all_faces_covered_once and not missing_refs),
        "input_evidence_valid": input_valid,
        "all_faces_covered_exactly_once": all_faces_covered_once,
        "coverage_by_face": {str(face_id): coverage.get(face_id, 0) for face_id in sorted(all_face_ids)},
        "node_reference_valid": bool(not missing_refs),
        "relation_reference_valid": bool(not missing_refs),
        "missing_relation_refs": missing_refs,
        "render_validation": {"status": "not_implemented"},
    }


def _selected_operations(
    selected_candidates: Sequence[OperationCandidate],
    generated_nodes_by_candidate: Dict[str, List[str]],
    generated_relations_by_candidate: Dict[str, List[str]],
) -> List[Dict[str, object]]:
    output = []
    for candidate in selected_candidates:
        output.append(
            {
                "id": candidate.id,
                "operation_type": candidate.operation_type,
                "patch_id": candidate.patch_id,
                "evidence": {"face_ids": list(candidate.covered_face_ids), "arc_ids": list(candidate.evidence_arc_ids)},
                "generated_node_ids": generated_nodes_by_candidate.get(candidate.id, []),
                "generated_relation_ids": generated_relations_by_candidate.get(candidate.id, []),
                "cost": {
                    "independent": float(candidate.independent_cost),
                    "operation": float(candidate.operation_cost),
                    "compression_gain": float(candidate.compression_gain),
                    "breakdown": candidate.cost_breakdown,
                },
                "metadata": _sanitize_metadata(candidate.metadata),
            }
        )
    return output


def build_operation_explanation_payload(
    evidence_payload: Dict[str, object],
    *,
    weak_payload: Dict[str, object] | None = None,
    role_prior_payload: Dict[str, object] | None = None,
    pairwise_prior_payload: Dict[str, object] | None = None,
    config: OperationExplainerConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or OperationExplainerConfig()
    source_tag = source_tag or str(evidence_payload.get("source_global_approx") or evidence_payload.get("source_partition_graph") or "")
    weak_payload = weak_payload or build_weak_explanation_payload(evidence_payload, config=WeakExplainerConfig(), source_tag=source_tag)
    role_prior_payload = role_prior_payload or build_explanation_payload(evidence_payload, config=ExplainerConfig(), source_tag=source_tag)
    pairwise_prior_payload = pairwise_prior_payload or build_pairwise_relation_payload(
        evidence_payload,
        config=PairwiseRelationConfig(convex_backend="fallback_cdt_greedy"),
        source_tag=source_tag,
    )

    patches = build_operation_patches(evidence_payload, role_prior_payload, pairwise_prior_payload, config)
    candidates, candidate_generation_diagnostics = propose_operation_candidates_with_diagnostics(
        evidence_payload,
        patches,
        role_prior_payload,
        pairwise_prior_payload,
        config,
    )
    candidates = score_operation_candidates(candidates, evidence_payload, config)
    selection = select_operations_with_ortools(candidates, _face_ids(evidence_payload), config)
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    selected_candidates = [candidate_by_id[candidate_id] for candidate_id in selection.selected_candidate_ids if candidate_id in candidate_by_id]

    nodes, relations, residuals, generated_nodes_by_candidate, generated_relations_by_candidate = _normalise_selected_parse_graph(selected_candidates)
    validation = _validate(evidence_payload, selected_candidates, nodes, relations)
    proxy_validation = validate_selected_operations_proxy(selected_candidates, evidence_payload, config)
    validation["proxy_validation"] = proxy_validation
    selected_operations = _selected_operations(selected_candidates, generated_nodes_by_candidate, generated_relations_by_candidate)

    operation_histogram: Dict[str, int] = {}
    role_histogram: Dict[str, int] = {}
    for candidate in selected_candidates:
        operation_histogram[candidate.operation_type] = int(operation_histogram.get(candidate.operation_type, 0)) + 1
    for node in nodes:
        role = str(node.get("role"))
        role_histogram[role] = int(role_histogram.get(role, 0)) + 1
    faces_by_id = _faces_by_id(evidence_payload)
    residual_area = sum(float(faces_by_id[face_id].get("features", {}).get("area", 0.0)) for face_id in selection.residual_face_ids if face_id in faces_by_id)
    total_area = float(evidence_payload.get("statistics", {}).get("total_face_area", sum(float(face.get("features", {}).get("area", 0.0)) for face in faces_by_id.values())))
    total_independent = sum(float(candidate.independent_cost) for candidate in selected_candidates)
    total_operation = sum(float(candidate.operation_cost) for candidate in selected_candidates)
    total_gain = sum(float(candidate.compression_gain) for candidate in selected_candidates)
    false_cover_area_total = sum(float(candidate.cost_breakdown.get("false_cover", {}).get("area", 0.0)) for candidate in selected_candidates)
    false_cover_ratio_max = max(
        [float(candidate.cost_breakdown.get("false_cover", {}).get("ratio", 0.0)) for candidate in selected_candidates] or [0.0]
    )

    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    hard_false_cover_candidate_count = sum(
        1
        for candidate in candidates
        if bool((candidate.cost_breakdown or {}).get("false_cover", {}).get("hard_invalid", False))
    )
    failure_reason_histogram: Dict[str, int] = {}
    for candidate in candidates:
        reason = candidate.failure_reason
        if not candidate.valid and not reason:
            reason = "unknown_invalid"
        if reason:
            failure_reason_histogram[str(reason)] = int(failure_reason_histogram.get(str(reason), 0)) + 1
    top_candidates = sorted(valid_candidates, key=lambda item: (-float(item.compression_gain), item.operation_type, item.id))[:50]
    candidate_summary = {
        "candidate_count": int(len(candidates)),
        "raw_candidate_count": int(candidate_generation_diagnostics["raw_candidate_count"]),
        "deduplicated_candidate_count": int(candidate_generation_diagnostics["deduplicated_candidate_count"]),
        "dropped_duplicate_count": int(candidate_generation_diagnostics["dropped_duplicate_count"]),
        "valid_candidate_count": int(len(valid_candidates)),
        "invalid_candidate_count": int(len(candidates) - len(valid_candidates)),
        "hard_false_cover_candidate_count": int(hard_false_cover_candidate_count),
        "failure_reason_histogram": dict(sorted(failure_reason_histogram.items())),
        "selected_candidate_ids": list(selection.selected_candidate_ids),
        "top_candidates": [_candidate_public(candidate) for candidate in top_candidates],
    }

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
            "target_profile": "operation_level_mdl_v1",
            "code_length": float(total_operation),
            "render_iou": None,
            "valid": bool(validation["is_valid"]),
            "render_validation": {
                "status": "not_implemented",
                "reason": "operation-level renderer is not implemented yet",
            },
        },
    }

    diagnostics = {
        "cost_profile": config.cost_profile,
        "face_count": int(len(faces_by_id)),
        "patch_count": int(len(patches)),
        "candidate_count": int(len(candidates)),
        "selected_operation_count": int(len(selected_candidates)),
        "raw_candidate_count": int(candidate_generation_diagnostics["raw_candidate_count"]),
        "deduplicated_candidate_count": int(candidate_generation_diagnostics["deduplicated_candidate_count"]),
        "dropped_duplicate_count": int(candidate_generation_diagnostics["dropped_duplicate_count"]),
        "residual_face_count": int(len(selection.residual_face_ids)),
        "selected_non_residual_count": int(sum(1 for candidate in selected_candidates if candidate.operation_type != RESIDUAL)),
        "selected_residual_count": int(sum(1 for candidate in selected_candidates if candidate.operation_type == RESIDUAL)),
        "residual_area_ratio": float(residual_area / total_area) if total_area > config.min_area_eps else 0.0,
        "total_independent_cost": float(total_independent),
        "total_operation_cost": float(total_operation),
        "total_compression_gain": float(total_gain),
        "false_cover_area_total": float(false_cover_area_total),
        "false_cover_ratio_max": float(false_cover_ratio_max),
        "hard_false_cover_candidate_count": int(hard_false_cover_candidate_count),
        "selection_method": selection.selection_method,
        "solver_status": selection.solver_status,
        "global_optimal": bool(selection.global_optimal),
        "operation_histogram": operation_histogram,
        "role_histogram": role_histogram,
        "failure_reasons": sorted(failure_reason_histogram.keys()),
        "failure_reason_histogram": dict(sorted(failure_reason_histogram.items())),
        "weak_profile": weak_payload.get("explainer_profile"),
        "role_prior_profile": role_prior_payload.get("explainer_profile", "initial_face_role_prior"),
        "pairwise_prior_format": pairwise_prior_payload.get("format"),
        "selection_diagnostics": selection.diagnostics,
    }

    return {
        "format": "maskgen_operation_explanation_v1",
        "source_evidence": source_tag,
        "explainer_profile": "operation_level_mdl_v1",
        "selected_operations": selected_operations,
        "candidate_summary": candidate_summary,
        "generator_target": generator_target,
        "diagnostics": diagnostics,
        "validation": validation,
        "config": asdict(config),
    }
