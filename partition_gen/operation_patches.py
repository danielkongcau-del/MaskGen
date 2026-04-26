from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from partition_gen.operation_label_pair_prior import REL_DIVIDES, REL_INSERTED_IN
from partition_gen.operation_types import OperationExplainerConfig, OperationPatch


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _adjacency_by_face(evidence_payload: Dict[str, object]) -> Dict[int, List[Dict[str, object]]]:
    output: Dict[int, List[Dict[str, object]]] = {}
    for adjacency in evidence_payload.get("adjacency", []):
        faces = [int(value) for value in adjacency.get("faces", [])]
        if len(faces) != 2:
            continue
        output.setdefault(faces[0], []).append(adjacency)
        output.setdefault(faces[1], []).append(adjacency)
    return output


def _neighbors(face_id: int, adjacency_by_face: Dict[int, List[Dict[str, object]]]) -> Set[int]:
    output: Set[int] = set()
    for adjacency in adjacency_by_face.get(face_id, []):
        left, right = [int(value) for value in adjacency.get("faces", [])]
        output.add(right if left == face_id else left)
    return output


def _arc_ids_for_faces(face_ids: Sequence[int], adjacency_by_face: Dict[int, List[Dict[str, object]]]) -> Tuple[int, ...]:
    face_set = set(int(face_id) for face_id in face_ids)
    arc_ids = set()
    for face_id in face_set:
        for adjacency in adjacency_by_face.get(face_id, []):
            faces = {int(value) for value in adjacency.get("faces", [])}
            if faces.issubset(face_set):
                arc_ids.update(int(value) for value in adjacency.get("arc_ids", []))
    return tuple(sorted(arc_ids))


def _role_prior_cost(role_prior_payload: Dict[str, object] | None, face_id: int, role: str) -> float | None:
    if not role_prior_payload:
        return None
    candidates = role_prior_payload.get("diagnostics", {}).get("role_candidates", {}).get(str(face_id), [])
    for candidate in candidates:
        if candidate.get("role") == role:
            return float(candidate.get("cost", 0.0))
    return None


def _role_prior_best(role_prior_payload: Dict[str, object] | None, face_id: int) -> str | None:
    if not role_prior_payload:
        return None
    candidates = role_prior_payload.get("diagnostics", {}).get("role_candidates", {}).get(str(face_id), [])
    if not candidates:
        return None
    best = min(candidates, key=lambda item: (float(item.get("cost", 0.0)), str(item.get("role"))))
    return str(best.get("role"))


def _shared_length_with_seed(seed_face_id: int, face_id: int, adjacency_by_face: Dict[int, List[Dict[str, object]]]) -> float:
    if seed_face_id == face_id:
        return float("inf")
    best = 0.0
    for adjacency in adjacency_by_face.get(seed_face_id, []):
        faces = {int(value) for value in adjacency.get("faces", [])}
        if face_id in faces and seed_face_id in faces:
            best = max(best, float(adjacency.get("shared_length", 0.0)))
    return float(best)


def _role_relevance(patch_type: str, face_id: int, role_prior_payload: Dict[str, object] | None) -> float:
    if patch_type == "divider_centered":
        cost = _role_prior_cost(role_prior_payload, face_id, "divider_region")
    elif patch_type == "support_centered":
        cost = _role_prior_cost(role_prior_payload, face_id, "insert_object")
    else:
        cost = None
    if cost is None:
        return 0.0
    return float(1.0 / (1.0 + max(0.0, cost)))


def trim_patch_faces_preserve_seed(
    face_ids: Sequence[int],
    *,
    seed_face_id: int | None,
    patch_type: str,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
    role_prior_payload: Dict[str, object] | None,
    max_patch_size: int,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    original = list(dict.fromkeys(int(face_id) for face_id in face_ids if int(face_id) in faces_by_id))
    if seed_face_id is not None and seed_face_id in faces_by_id and seed_face_id not in original:
        original.insert(0, int(seed_face_id))
    if max_patch_size <= 0:
        return tuple(), {
            "trimmed": bool(original),
            "original_face_count": int(len(original)),
            "final_face_count": 0,
            "dropped_face_ids": original,
        }
    if seed_face_id is None:
        ordered = sorted(
            set(original),
            key=lambda face_id: (
                -float(faces_by_id[face_id].get("features", {}).get("area", 0.0)),
                face_id,
            ),
        )
    else:
        seed_area = float(faces_by_id[int(seed_face_id)].get("features", {}).get("area", 0.0)) if int(seed_face_id) in faces_by_id else 0.0
        ordered = sorted(
            set(original),
            key=lambda face_id: (
                0 if face_id == int(seed_face_id) else 1,
                0 if _shared_length_with_seed(int(seed_face_id), face_id, adjacency_by_face) > 0 else 1,
                -_shared_length_with_seed(int(seed_face_id), face_id, adjacency_by_face),
                -_role_relevance(patch_type, face_id, role_prior_payload),
                abs(float(faces_by_id[face_id].get("features", {}).get("area", 0.0)) - seed_area),
                face_id,
            ),
        )
    limited = tuple(ordered[:max_patch_size])
    dropped = [face_id for face_id in ordered if face_id not in set(limited)]
    return limited, {
        "trimmed": bool(len(limited) < len(set(original))),
        "original_face_count": int(len(set(original))),
        "final_face_count": int(len(limited)),
        "dropped_face_ids": dropped,
    }


def _is_divider_like(face: Dict[str, object], role_prior_payload: Dict[str, object] | None, config: OperationExplainerConfig) -> bool:
    face_id = int(face["id"])
    if _role_prior_best(role_prior_payload, face_id) == "divider_region":
        return True
    features = face.get("features", {})
    aspect = float(features.get("oriented_aspect_ratio", 0.0))
    degree = int(features.get("degree", 0))
    return bool(degree >= config.min_divider_neighbor_count or aspect >= config.thin_aspect_ratio * 1.5)


def _is_support_like(face: Dict[str, object], median_area: float, role_prior_payload: Dict[str, object] | None) -> bool:
    face_id = int(face["id"])
    if _role_prior_best(role_prior_payload, face_id) == "support_region":
        return True
    features = face.get("features", {})
    return bool(float(features.get("area", 0.0)) >= median_area and not features.get("is_thin", False))


def _make_patch(
    patch_id: str,
    patch_type: str,
    seed_face_id: int | None,
    face_ids: Sequence[int],
    adjacency_by_face: Dict[int, List[Dict[str, object]]],
    metadata: Dict[str, object] | None = None,
) -> OperationPatch:
    unique_face_ids = tuple(sorted(set(int(face_id) for face_id in face_ids)))
    return OperationPatch(
        id=patch_id,
        patch_type=patch_type,
        seed_face_id=seed_face_id,
        face_ids=unique_face_ids,
        arc_ids=_arc_ids_for_faces(unique_face_ids, adjacency_by_face),
        metadata=metadata or {},
    )


def _pairwise_label_pairs(pairwise_prior_payload: Dict[str, object] | None) -> Dict[Tuple[int, int], Dict[str, object]]:
    output: Dict[Tuple[int, int], Dict[str, object]] = {}
    if not pairwise_prior_payload:
        return output
    for pair in pairwise_prior_payload.get("pairs", []):
        labels = pair.get("labels", [])
        if len(labels) != 2:
            continue
        key = tuple(sorted(int(value) for value in labels))
        output[key] = pair
    return output


def build_operation_patches(
    evidence_payload: Dict[str, object],
    role_prior_payload: Dict[str, object] | None,
    pairwise_prior_payload: Dict[str, object] | None,
    config: OperationExplainerConfig,
    label_pair_prior_payload: Dict[str, object] | None = None,
) -> List[OperationPatch]:
    faces_by_id = _faces_by_id(evidence_payload)
    adjacency = _adjacency_by_face(evidence_payload)
    areas = sorted(float(face.get("features", {}).get("area", 0.0)) for face in faces_by_id.values())
    median_area = areas[len(areas) // 2] if areas else 0.0

    patches: List[OperationPatch] = []
    seen: Set[Tuple[str, Tuple[int, ...]]] = set()

    def add_patch(patch_type: str, seed_face_id: int | None, face_ids: Sequence[int], metadata: Dict[str, object] | None = None) -> None:
        limited_face_ids, trim_metadata = trim_patch_faces_preserve_seed(
            face_ids,
            seed_face_id=seed_face_id,
            patch_type=patch_type,
            faces_by_id=faces_by_id,
            adjacency_by_face=adjacency,
            role_prior_payload=role_prior_payload,
            max_patch_size=config.max_patch_size,
        )
        if not limited_face_ids:
            return
        key = (patch_type, limited_face_ids)
        if key in seen:
            return
        seen.add(key)
        patch_id = f"{patch_type}_{len(patches)}"
        patch_metadata = dict(metadata or {})
        patch_metadata.update(trim_metadata)
        patches.append(_make_patch(patch_id, patch_type, seed_face_id, limited_face_ids, adjacency, patch_metadata))

    for face in faces_by_id.values():
        face_id = int(face["id"])
        neighbor_ids = sorted(_neighbors(face_id, adjacency))
        if _is_divider_like(face, role_prior_payload, config):
            same_label_thin = [
                neighbor_id
                for neighbor_id in neighbor_ids
                if int(faces_by_id[neighbor_id].get("label", -1)) == int(face.get("label", -2))
                and _is_divider_like(faces_by_id[neighbor_id], role_prior_payload, config)
            ]
            add_patch("divider_centered", face_id, [face_id, *same_label_thin, *neighbor_ids], {"source": "role_or_shape_prior"})

        if _is_support_like(face, median_area, role_prior_payload):
            support_area = float(face.get("features", {}).get("area", 0.0))
            compact_neighbors = []
            for neighbor_id in neighbor_ids:
                neighbor = faces_by_id[neighbor_id]
                features = neighbor.get("features", {})
                neighbor_area = float(features.get("area", 0.0))
                if neighbor_area <= max(support_area * config.small_area_ratio, median_area):
                    compact_neighbors.append(neighbor_id)
            if compact_neighbors:
                add_patch("support_centered", face_id, [face_id, *compact_neighbors], {"source": "area_compact_neighbor_prior"})

    pairwise_pairs = _pairwise_label_pairs(pairwise_prior_payload)
    for adjacency_item in evidence_payload.get("adjacency", []):
        face_ids = [int(value) for value in adjacency_item.get("faces", [])]
        labels = [int(value) for value in adjacency_item.get("labels", [])]
        if len(face_ids) != 2 or len(labels) != 2:
            continue
        pair_key = tuple(sorted(labels))
        pair = pairwise_pairs.get(pair_key)
        if not pair:
            continue
        selected_template = str(pair.get("selected", {}).get("template", "unknown"))
        if selected_template not in {"support_with_inserts", "split_by_divider", "adjacent_supports", "independent_faces"}:
            continue
        local = set(face_ids)
        for face_id in face_ids:
            local.update(_neighbors(face_id, adjacency))
        add_patch(
            "pair_contact",
            None,
            sorted(local),
            {
                "source": "pairwise_prior",
                "labels": labels,
                "pairwise_template": selected_template,
                "pairwise_cost": pair.get("selected", {}).get("cost"),
            },
        )

    if config.enable_label_pair_consistency and label_pair_prior_payload:
        for pair in label_pair_prior_payload.get("pairs", []):
            selected = pair.get("selected", {})
            relation_type = selected.get("relation_type")
            subject_label = selected.get("subject_label")
            object_label = selected.get("object_label")
            if subject_label is None or object_label is None:
                continue
            subject_label = int(subject_label)
            object_label = int(object_label)
            if relation_type == REL_DIVIDES:
                for face in faces_by_id.values():
                    face_id = int(face["id"])
                    if int(face.get("label", -1)) != subject_label:
                        continue
                    neighbor_ids = sorted(_neighbors(face_id, adjacency))
                    object_neighbors = [neighbor_id for neighbor_id in neighbor_ids if int(faces_by_id[neighbor_id].get("label", -1)) == object_label]
                    if not object_neighbors:
                        continue
                    same_label_neighbors = [
                        neighbor_id
                        for neighbor_id in neighbor_ids
                        if int(faces_by_id[neighbor_id].get("label", -1)) == subject_label
                    ]
                    expanded_object_neighbors = set(object_neighbors)
                    for neighbor_id in same_label_neighbors:
                        for second_hop in _neighbors(neighbor_id, adjacency):
                            if int(faces_by_id[second_hop].get("label", -1)) == object_label:
                                expanded_object_neighbors.add(second_hop)
                    add_patch(
                        "label_pair_divider",
                        face_id,
                        [face_id, *same_label_neighbors, *sorted(expanded_object_neighbors)],
                        {
                            "source": "label_pair_prior",
                            "label_pair_relation": selected.get("relation"),
                            "relation_type": relation_type,
                            "divider_label": subject_label,
                            "support_label": object_label,
                            "relation_confidence": selected.get("confidence"),
                        },
                    )
            elif relation_type == REL_INSERTED_IN:
                for face in faces_by_id.values():
                    face_id = int(face["id"])
                    if int(face.get("label", -1)) != object_label:
                        continue
                    neighbor_ids = sorted(_neighbors(face_id, adjacency))
                    insert_neighbors = [neighbor_id for neighbor_id in neighbor_ids if int(faces_by_id[neighbor_id].get("label", -1)) == subject_label]
                    if not insert_neighbors:
                        continue
                    add_patch(
                        "label_pair_insert",
                        face_id,
                        [face_id, *insert_neighbors],
                        {
                            "source": "label_pair_prior",
                            "label_pair_relation": selected.get("relation"),
                            "relation_type": relation_type,
                            "support_label": object_label,
                            "insert_label": subject_label,
                            "relation_confidence": selected.get("confidence"),
                        },
                    )

    if not patches and faces_by_id:
        add_patch("fallback_global", None, sorted(faces_by_id), {"source": "fallback"})

    return patches
