from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from shapely.geometry import Point

from partition_gen.latent_geometry import build_latent_geometry_candidates
from partition_gen.operation_geometry import (
    atom_to_local,
    frame_from_geometry,
    polygon_from_face,
    polygon_to_local_payload,
    union_face_polygons,
)
from partition_gen.operation_label_pair_prior import REL_DIVIDES, REL_INSERTED_IN, REL_PARALLEL, relation_for_labels
from partition_gen.operation_types import (
    DIVIDE_BY_REGION,
    OVERLAY_INSERT,
    PARALLEL_SUPPORTS,
    RESIDUAL,
    OperationCandidate,
    OperationExplainerConfig,
    OperationPatch,
)


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _adjacency_by_pair(evidence_payload: Dict[str, object]) -> Dict[Tuple[int, int], Dict[str, object]]:
    output: Dict[Tuple[int, int], Dict[str, object]] = {}
    for adjacency in evidence_payload.get("adjacency", []):
        faces = [int(value) for value in adjacency.get("faces", [])]
        if len(faces) != 2:
            continue
        output[tuple(sorted(faces))] = adjacency
    return output


def _adjacency_by_face(evidence_payload: Dict[str, object]) -> Dict[int, List[Dict[str, object]]]:
    output: Dict[int, List[Dict[str, object]]] = {}
    for adjacency in evidence_payload.get("adjacency", []):
        faces = [int(value) for value in adjacency.get("faces", [])]
        if len(faces) != 2:
            continue
        output.setdefault(faces[0], []).append(adjacency)
        output.setdefault(faces[1], []).append(adjacency)
    return output


def _are_adjacent(left_id: int, right_id: int, adjacency_by_pair: Dict[Tuple[int, int], Dict[str, object]]) -> bool:
    return tuple(sorted((int(left_id), int(right_id)))) in adjacency_by_pair


def _shared_length(left_id: int, right_id: int, adjacency_by_pair: Dict[Tuple[int, int], Dict[str, object]]) -> float:
    adjacency = adjacency_by_pair.get(tuple(sorted((int(left_id), int(right_id)))))
    if not adjacency:
        return 0.0
    return float(adjacency.get("shared_length", 0.0))


def _support_boundary_fraction(support_id: int, insert: Dict[str, object], adjacency_by_pair: Dict[Tuple[int, int], Dict[str, object]]) -> float:
    perimeter = float(insert.get("features", {}).get("perimeter", 0.0))
    if perimeter <= 1e-8:
        perimeter = polygon_from_face(insert).length
    if perimeter <= 1e-8:
        return 0.0
    return float(_shared_length(support_id, int(insert["id"]), adjacency_by_pair) / perimeter)


def _face_arc_ids(face: Dict[str, object]) -> List[int]:
    arc_ids = []
    for ref in face.get("outer_arc_refs", []):
        arc_ids.append(int(ref["arc_id"]))
    for refs in face.get("hole_arc_refs", []):
        for ref in refs:
            arc_ids.append(int(ref["arc_id"]))
    return sorted(set(arc_ids))


def _polygon_node(
    node_id: str,
    role: str,
    label: int,
    geometry,
    *,
    evidence_face_ids: Sequence[int],
    evidence_arc_ids: Sequence[int],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    payload = polygon_to_local_payload(geometry, eps=config.min_area_eps)
    return {
        "id": node_id,
        "role": role,
        "label": int(label),
        "frame": payload["frame"],
        "geometry_model": "polygon_code",
        "geometry": payload["geometry"],
        "evidence": {"face_ids": [int(value) for value in evidence_face_ids], "arc_ids": [int(value) for value in evidence_arc_ids]},
    }


def _face_polygon_node(
    node_id: str,
    role: str,
    face: Dict[str, object],
    *,
    evidence_arc_ids: Sequence[int],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    return _polygon_node(
        node_id,
        role,
        int(face.get("label", -1)),
        polygon_from_face(face),
        evidence_face_ids=[int(face["id"])],
        evidence_arc_ids=evidence_arc_ids,
        config=config,
    )


def _residual_node(
    node_id: str,
    face: Dict[str, object],
    *,
    evidence_arc_ids: Sequence[int],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    polygon = polygon_from_face(face)
    frame = frame_from_geometry(polygon, eps=config.min_area_eps)
    atoms = face.get("convex_partition", {}).get("atoms", [])
    if atoms:
        return {
            "id": node_id,
            "role": "residual_region",
            "label": int(face.get("label", -1)),
            "frame": frame,
            "geometry_model": "convex_atoms",
            "atoms": [atom_to_local(atom, frame) for atom in atoms],
            "evidence": {"face_ids": [int(face["id"])], "arc_ids": [int(value) for value in evidence_arc_ids]},
        }
    return _face_polygon_node(node_id, "residual_region", face, evidence_arc_ids=evidence_arc_ids, config=config)


def _empty_candidate(
    candidate_id: str,
    operation_type: str,
    patch_id: str,
    covered_face_ids: Sequence[int],
    arc_ids: Sequence[int],
    *,
    failure_reason: str,
) -> OperationCandidate:
    return OperationCandidate(
        id=candidate_id,
        operation_type=operation_type,
        patch_id=patch_id,
        covered_face_ids=tuple(sorted(set(int(value) for value in covered_face_ids))),
        evidence_arc_ids=tuple(sorted(set(int(value) for value in arc_ids))),
        nodes=[],
        relations=[],
        residuals=[],
        independent_cost=0.0,
        operation_cost=0.0,
        compression_gain=0.0,
        cost_breakdown={},
        valid=False,
        failure_reason=failure_reason,
        metadata={},
    )


def _candidate(
    candidate_id: str,
    operation_type: str,
    patch: OperationPatch,
    covered_face_ids: Sequence[int],
    nodes: List[Dict[str, object]],
    relations: List[Dict[str, object]],
    residuals: List[Dict[str, object]] | None = None,
    *,
    metadata: Dict[str, object] | None = None,
    valid: bool = True,
    failure_reason: str | None = None,
) -> OperationCandidate:
    return OperationCandidate(
        id=candidate_id,
        operation_type=operation_type,
        patch_id=patch.id,
        covered_face_ids=tuple(sorted(set(int(value) for value in covered_face_ids))),
        evidence_arc_ids=tuple(sorted(set(int(value) for value in patch.arc_ids))),
        nodes=nodes,
        relations=relations,
        residuals=residuals or [],
        independent_cost=0.0,
        operation_cost=0.0,
        compression_gain=0.0,
        cost_breakdown={},
        valid=valid,
        failure_reason=failure_reason,
        metadata=metadata or {},
    )


def _candidate_label(faces: Sequence[Dict[str, object]]) -> int:
    if not faces:
        return -1
    return int(max(faces, key=lambda face: float(face.get("features", {}).get("area", 0.0))).get("label", -1))


def _relation_matches(selected: Dict[str, object], expected_type: str, subject_label: int, object_label: int) -> bool:
    if not selected:
        return True
    if selected.get("relation_type") == REL_PARALLEL and expected_type == REL_PARALLEL:
        return True
    return bool(
        selected.get("relation_type") == expected_type
        and int(selected.get("subject_label", -9999)) == int(subject_label)
        and int(selected.get("object_label", -9999)) == int(object_label)
    )


def _label_pair_consistency_metadata(
    *,
    label_pair_prior_payload: Dict[str, object] | None,
    config: OperationExplainerConfig,
    expectations: Sequence[Tuple[int, int, str]],
) -> Dict[str, object]:
    if not config.enable_label_pair_consistency or not label_pair_prior_payload:
        return {}
    checks = []
    inconsistent = False
    hard_inconsistent = False
    for subject_label, object_label, expected_type in expectations:
        if int(subject_label) == int(object_label):
            continue
        selected = relation_for_labels(label_pair_prior_payload, int(subject_label), int(object_label))
        if not selected:
            continue
        consistent = _relation_matches(selected, expected_type, int(subject_label), int(object_label))
        inconsistent = inconsistent or not consistent
        confidence = float(selected.get("confidence", 0.0) or 0.0)
        hard_inconsistent = hard_inconsistent or (not consistent and confidence >= config.label_pair_hard_min_confidence)
        checks.append(
            {
                "subject_label": int(subject_label),
                "object_label": int(object_label),
                "expected_relation_type": expected_type,
                "selected_relation_type": selected.get("relation_type"),
                "selected_subject_label": selected.get("subject_label"),
                "selected_object_label": selected.get("object_label"),
                "selected_relation": selected.get("relation"),
                "selected_confidence": confidence,
                "selected_source": selected.get("source", "auto_label_pair_prior"),
                "selected_hard": bool(selected.get("hard", False)),
                "consistent": bool(consistent),
            }
        )
    if not checks:
        return {}
    return {
        "label_pair_consistency": {
            "checks": checks,
            "consistent": bool(not inconsistent),
            "hard_inconsistent": bool(hard_inconsistent),
            "exception_tokens": int(config.token_label_pair_consistency_exception if inconsistent else 0),
            "penalty": float(config.cost_label_pair_consistency_penalty if inconsistent else 0.0),
        },
        "label_pair_consistency_exception": int(config.token_label_pair_consistency_exception if inconsistent else 0),
        "label_pair_consistency_penalty": float(config.cost_label_pair_consistency_penalty if inconsistent else 0.0),
    }


def _overlay_candidates(
    patch: OperationPatch,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency_by_pair: Dict[Tuple[int, int], Dict[str, object]],
    config: OperationExplainerConfig,
    start_index: int,
    label_pair_prior_payload: Dict[str, object] | None,
) -> List[OperationCandidate]:
    if not config.enable_overlay_insert or len(patch.face_ids) < 2:
        return []
    faces = [faces_by_id[face_id] for face_id in patch.face_ids if face_id in faces_by_id]
    support = max(faces, key=lambda face: float(face.get("features", {}).get("area", 0.0)))
    support_area = float(support.get("features", {}).get("area", 0.0))
    inserts = []
    for face in faces:
        if int(face["id"]) == int(support["id"]):
            continue
        features = face.get("features", {})
        area = float(features.get("area", 0.0))
        boundary_fraction = _support_boundary_fraction(int(support["id"]), face, adjacency_by_pair)
        if (
            area <= max(support_area * config.small_area_ratio, config.min_area_eps)
            and _are_adjacent(int(support["id"]), int(face["id"]), adjacency_by_pair)
            and boundary_fraction >= config.min_insert_support_boundary_fraction
        ):
            inserts.append(face)
    if not inserts:
        return []
    insert_area = sum(float(face.get("features", {}).get("area", 0.0)) for face in inserts)
    if support_area <= config.min_area_eps or insert_area / support_area > config.max_insert_group_area_ratio:
        return []

    candidates: List[OperationCandidate] = []
    latent_candidates = build_latent_geometry_candidates([support], inserts, OVERLAY_INSERT, config)
    for local_index, latent in enumerate(latent_candidates[: config.max_candidates_per_patch]):
        if config.require_insert_touch_or_contained:
            latent_geometry = latent.get("geometry")
            invalid_insert_ids = []
            if latent_geometry is not None and not latent_geometry.is_empty:
                for insert in inserts:
                    centroid = polygon_from_face(insert).centroid
                    touches = _are_adjacent(int(support["id"]), int(insert["id"]), adjacency_by_pair)
                    contained = bool(latent_geometry.covers(Point(centroid.x, centroid.y)))
                    if not (touches or contained):
                        invalid_insert_ids.append(int(insert["id"]))
            if invalid_insert_ids:
                continue
        candidate_id = f"cand_{start_index + len(candidates)}"
        support_node_id = f"{candidate_id}_support"
        group_node_id = f"{candidate_id}_insert_group"
        arc_ids = patch.arc_ids
        nodes = [
            _polygon_node(
                support_node_id,
                "support_region",
                int(support.get("label", -1)),
                latent["geometry"],
                evidence_face_ids=[int(support["id"]), *[int(face["id"]) for face in inserts]],
                evidence_arc_ids=arc_ids,
                config=config,
            ),
            {
                "id": group_node_id,
                "role": "insert_object_group",
                "label": int(inserts[0].get("label", -1)) if inserts else -1,
                "geometry_model": "none",
                "children": [],
                "evidence": {"face_ids": [int(face["id"]) for face in inserts], "arc_ids": list(arc_ids)},
            },
        ]
        relations = [
            {
                "type": "inserted_in",
                "object": group_node_id,
                "container": support_node_id,
                "support": support_node_id,
            }
        ]
        for insert_index, insert in enumerate(inserts):
            insert_node_id = f"{candidate_id}_insert_{insert_index}"
            nodes.append(_face_polygon_node(insert_node_id, "insert_object", insert, evidence_arc_ids=arc_ids, config=config))
            nodes[1]["children"].append(insert_node_id)
            relations.append({"type": "contains", "parent": group_node_id, "child": insert_node_id})
        covered = [int(support["id"]), *[int(face["id"]) for face in inserts]]
        insert_labels = sorted({int(face.get("label", -1)) for face in inserts})
        consistency = _label_pair_consistency_metadata(
            label_pair_prior_payload=label_pair_prior_payload,
            config=config,
            expectations=[(insert_label, int(support.get("label", -1)), REL_INSERTED_IN) for insert_label in insert_labels],
        )
        metadata = {
            "latent_geometry": latent,
            "support_face_ids": [int(support["id"])],
            "insert_face_ids": [int(face["id"]) for face in inserts],
        }
        metadata.update(consistency)
        candidates.append(
            _candidate(
                candidate_id,
                OVERLAY_INSERT,
                patch,
                covered,
                nodes,
                relations,
                metadata=metadata,
                valid=bool(latent["valid"]),
                failure_reason=latent["failure_reason"],
            )
        )
    return candidates


def _divider_candidates(
    patch: OperationPatch,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
    config: OperationExplainerConfig,
    start_index: int,
    label_pair_prior_payload: Dict[str, object] | None,
) -> List[OperationCandidate]:
    if not config.enable_divide_by_region or len(patch.face_ids) < 3:
        return []
    if patch.patch_type == "support_centered":
        return []
    if patch.patch_type == "pair_contact" and patch.metadata.get("pairwise_template") not in {"split_by_divider"}:
        return []
    faces = [faces_by_id[face_id] for face_id in patch.face_ids if face_id in faces_by_id]
    if not faces:
        return []
    divider_label = patch.metadata.get("divider_label")
    support_label = patch.metadata.get("support_label")
    if divider_label is not None and support_label is not None:
        divider_faces = [face for face in faces if int(face.get("label", -1)) == int(divider_label)]
        support_pool = [face for face in faces if int(face.get("label", -1)) == int(support_label)]
        if patch.seed_face_id in faces_by_id and int(faces_by_id[int(patch.seed_face_id)].get("label", -1)) == int(divider_label):
            seed = faces_by_id[int(patch.seed_face_id)]
            divider_faces = [seed, *[face for face in divider_faces if int(face["id"]) != int(seed["id"])]]
        if not divider_faces:
            return []
        divider_ids = {int(face["id"]) for face in divider_faces}
        adjacent_support_faces = []
        for face in support_pool:
            for divider_id in divider_ids:
                for adjacency in adjacency_by_face.get(divider_id, []):
                    if int(face["id"]) in [int(value) for value in adjacency.get("faces", [])]:
                        adjacent_support_faces.append(face)
                        break
                if adjacent_support_faces and int(adjacent_support_faces[-1]["id"]) == int(face["id"]):
                    break
        support_faces = list({int(face["id"]): face for face in adjacent_support_faces}.values())
    else:
        divider = faces_by_id.get(patch.seed_face_id) if patch.seed_face_id in faces_by_id else None
        if divider is None:
            divider = max(
                faces,
                key=lambda face: (
                    float(face.get("features", {}).get("oriented_aspect_ratio", 0.0)),
                    int(face.get("features", {}).get("degree", 0)),
                ),
            )
        divider_features = divider.get("features", {})
        divider_degree = int(divider_features.get("degree", 0))
        divider_aspect = float(divider_features.get("oriented_aspect_ratio", 0.0))
        if not (divider_degree >= config.min_divider_neighbor_count or divider_aspect >= config.thin_aspect_ratio * 1.5):
            return []
        divider_faces = [divider]
        support_faces = [face for face in faces if int(face["id"]) != int(divider["id"])]
        if len(support_faces) < 2:
            return []
        adjacent_support_faces = []
        for face in support_faces:
            for adjacency in adjacency_by_face.get(int(divider["id"]), []):
                if int(face["id"]) in [int(value) for value in adjacency.get("faces", [])]:
                    adjacent_support_faces.append(face)
                    break
        support_faces = adjacent_support_faces
    if len(support_faces) < config.min_divider_neighbor_count:
        return []
    label_counts: Dict[int, int] = {}
    for face in support_faces:
        label = int(face.get("label", -1))
        label_counts[label] = int(label_counts.get(label, 0)) + 1
    if max(label_counts.values() or [0]) < config.min_divider_same_label_neighbor_count:
        return []
    areas = [float(face.get("features", {}).get("area", 0.0)) for face in support_faces]
    if areas and max(areas) <= config.min_area_eps:
        return []
    if areas and sum(1 for area in areas if area >= max(areas) * 0.1) < config.min_divider_neighbor_count:
        return []
    divider_area = float(sum(float(face.get("features", {}).get("area", 0.0)) for face in divider_faces))
    support_area_sum = float(sum(areas))
    if support_area_sum <= config.min_area_eps or divider_area / support_area_sum > config.max_divider_to_support_area_ratio:
        return []
    label_diversity = len({int(face.get("label", -1)) for face in support_faces})
    label_diversity_penalty = (
        float(max(0, label_diversity - config.max_support_label_diversity_without_penalty) * config.support_label_diversity_penalty)
    )

    candidates: List[OperationCandidate] = []
    latent_candidates = build_latent_geometry_candidates(support_faces, divider_faces, DIVIDE_BY_REGION, config)
    for latent in latent_candidates[: config.max_candidates_per_patch]:
        candidate_id = f"cand_{start_index + len(candidates)}"
        support_node_id = f"{candidate_id}_support"
        divider_node_id = f"{candidate_id}_divider"
        arc_ids = patch.arc_ids
        divider_face_ids = [int(face["id"]) for face in divider_faces]
        support_face_ids = [int(face["id"]) for face in support_faces]
        divider_label_value = _candidate_label(divider_faces)
        divider_geometry = union_face_polygons(divider_faces)
        divider_node = _polygon_node(
            divider_node_id,
            "divider_region",
            divider_label_value,
            divider_geometry,
            evidence_face_ids=divider_face_ids,
            evidence_arc_ids=arc_ids,
            config=config,
        )
        nodes = [
            _polygon_node(
                support_node_id,
                "support_region",
                _candidate_label(support_faces),
                latent["geometry"],
                evidence_face_ids=support_face_ids,
                evidence_arc_ids=arc_ids,
                config=config,
            ),
            divider_node,
        ]
        relations = [
            {
                "type": "divides",
                "divider": divider_node_id,
                "target": support_node_id,
                "support": support_node_id,
            }
        ]
        covered = [*divider_face_ids, *support_face_ids]
        consistency = _label_pair_consistency_metadata(
            label_pair_prior_payload=label_pair_prior_payload,
            config=config,
            expectations=[(divider_label_value, _candidate_label(support_faces), REL_DIVIDES)],
        )
        metadata = {
            "latent_geometry": latent,
            "divider_face_ids": divider_face_ids,
            "support_face_ids": support_face_ids,
            "support_label_diversity": int(label_diversity),
            "support_label_diversity_penalty": float(label_diversity_penalty),
        }
        metadata.update(consistency)
        candidates.append(
            _candidate(
                candidate_id,
                DIVIDE_BY_REGION,
                patch,
                covered,
                nodes,
                relations,
                metadata=metadata,
                valid=bool(latent["valid"]),
                failure_reason=latent["failure_reason"],
            )
        )
    return candidates


def _parallel_candidates(
    patch: OperationPatch,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency_by_pair: Dict[Tuple[int, int], Dict[str, object]],
    config: OperationExplainerConfig,
    start_index: int,
    label_pair_prior_payload: Dict[str, object] | None,
) -> List[OperationCandidate]:
    if not config.enable_parallel_supports or len(patch.face_ids) < 2:
        return []
    candidates: List[OperationCandidate] = []
    local_pairs = []
    for left_index, left_id in enumerate(patch.face_ids):
        for right_id in patch.face_ids[left_index + 1 :]:
            key = tuple(sorted((int(left_id), int(right_id))))
            if key in adjacency_by_pair:
                local_pairs.append((key, adjacency_by_pair[key]))
    local_pairs.sort(key=lambda item: -float(item[1].get("shared_length", 0.0)))
    for key, adjacency in local_pairs[: max(1, config.max_candidates_per_patch // 4)]:
        left = faces_by_id[key[0]]
        right = faces_by_id[key[1]]
        left_area = float(left.get("features", {}).get("area", 0.0))
        right_area = float(right.get("features", {}).get("area", 0.0))
        if min(left_area, right_area) < max(left_area, right_area) * config.small_area_ratio:
            continue
        candidate_id = f"cand_{start_index + len(candidates)}"
        left_node_id = f"{candidate_id}_support_a"
        right_node_id = f"{candidate_id}_support_b"
        arc_ids = tuple(int(value) for value in adjacency.get("arc_ids", [])) or patch.arc_ids
        nodes = [
            _face_polygon_node(left_node_id, "support_region", left, evidence_arc_ids=arc_ids, config=config),
            _face_polygon_node(right_node_id, "support_region", right, evidence_arc_ids=arc_ids, config=config),
        ]
        relations = [{"type": "adjacent_to", "faces": [left_node_id, right_node_id], "arc_ids": list(arc_ids)}]
        consistency = _label_pair_consistency_metadata(
            label_pair_prior_payload=label_pair_prior_payload,
            config=config,
            expectations=[(int(left.get("label", -1)), int(right.get("label", -1)), REL_PARALLEL)],
        )
        metadata = {"support_face_ids": [int(left["id"]), int(right["id"])]}
        metadata.update(consistency)
        candidates.append(
            _candidate(
                candidate_id,
                PARALLEL_SUPPORTS,
                patch,
                [int(left["id"]), int(right["id"])],
                nodes,
                relations,
                metadata=metadata,
            )
        )
    return candidates


def residual_candidate_for_face(face: Dict[str, object], config: OperationExplainerConfig, *, index: int) -> OperationCandidate:
    patch = OperationPatch(
        id=f"residual_patch_{int(face['id'])}",
        patch_type="residual",
        seed_face_id=int(face["id"]),
        face_ids=(int(face["id"]),),
        arc_ids=tuple(_face_arc_ids(face)),
        metadata={"source": "residual_fallback"},
    )
    candidate_id = f"residual_{index}"
    node_id = f"{candidate_id}_node"
    node = _residual_node(node_id, face, evidence_arc_ids=patch.arc_ids, config=config)
    residual = {
        "node_id": node_id,
        "face_ids": [int(face["id"])],
        "area": float(face.get("features", {}).get("area", 0.0)),
        "reason": "operation_residual_fallback",
    }
    return _candidate(candidate_id, RESIDUAL, patch, [int(face["id"])], [node], [], [residual], metadata={"face_id": int(face["id"])})


def _dedupe_key(candidate: OperationCandidate):
    metadata = candidate.metadata or {}
    latent = metadata.get("latent_geometry")
    latent_policy = latent.get("policy") if isinstance(latent, dict) else ""
    if candidate.operation_type == OVERLAY_INSERT:
        return (
            candidate.operation_type,
            tuple(metadata.get("support_face_ids", candidate.covered_face_ids)),
            tuple(metadata.get("insert_face_ids", ())),
            latent_policy,
        )
    if candidate.operation_type == DIVIDE_BY_REGION:
        return (
            candidate.operation_type,
            tuple(metadata.get("divider_face_ids", ())),
            tuple(metadata.get("support_face_ids", candidate.covered_face_ids)),
            latent_policy,
        )
    if candidate.operation_type == PARALLEL_SUPPORTS:
        return (candidate.operation_type, tuple(metadata.get("support_face_ids", candidate.covered_face_ids)))
    return (candidate.operation_type, candidate.id)


def propose_operation_candidates_with_diagnostics(
    evidence_payload: Dict[str, object],
    patches: Sequence[OperationPatch],
    role_prior_payload: Dict[str, object] | None,
    pairwise_prior_payload: Dict[str, object] | None,
    config: OperationExplainerConfig,
    label_pair_prior_payload: Dict[str, object] | None = None,
) -> List[OperationCandidate]:
    del role_prior_payload, pairwise_prior_payload
    faces_by_id = _faces_by_id(evidence_payload)
    adjacency_by_pair = _adjacency_by_pair(evidence_payload)
    adjacency_by_face = _adjacency_by_face(evidence_payload)
    candidates: List[OperationCandidate] = []
    for patch in patches:
        start = len(candidates)
        patch_candidates: List[OperationCandidate] = []
        patch_candidates.extend(
            _overlay_candidates(patch, faces_by_id, adjacency_by_pair, config, start + len(patch_candidates), label_pair_prior_payload)
        )
        patch_candidates.extend(
            _divider_candidates(patch, faces_by_id, adjacency_by_face, config, start + len(patch_candidates), label_pair_prior_payload)
        )
        patch_candidates.extend(
            _parallel_candidates(patch, faces_by_id, adjacency_by_pair, config, start + len(patch_candidates), label_pair_prior_payload)
        )
        candidates.extend(patch_candidates[: config.max_candidates_per_patch])

    if config.enable_residual:
        for index, face in enumerate(sorted(faces_by_id.values(), key=lambda item: int(item["id"]))):
            candidates.append(residual_candidate_for_face(face, config, index=index))

    unique: List[OperationCandidate] = []
    seen = set()
    for candidate in candidates:
        key = _dedupe_key(candidate)
        if key in seen and candidate.operation_type != RESIDUAL:
            continue
        seen.add(key)
        unique.append(candidate)
    diagnostics = {
        "raw_candidate_count": int(len(candidates)),
        "deduplicated_candidate_count": int(len(unique)),
        "dropped_duplicate_count": int(len(candidates) - len(unique)),
    }
    return unique, diagnostics


def propose_operation_candidates(
    evidence_payload: Dict[str, object],
    patches: Sequence[OperationPatch],
    role_prior_payload: Dict[str, object] | None,
    pairwise_prior_payload: Dict[str, object] | None,
    config: OperationExplainerConfig,
    label_pair_prior_payload: Dict[str, object] | None = None,
) -> List[OperationCandidate]:
    candidates, _ = propose_operation_candidates_with_diagnostics(
        evidence_payload,
        patches,
        role_prior_payload,
        pairwise_prior_payload,
        config,
        label_pair_prior_payload,
    )
    return candidates
