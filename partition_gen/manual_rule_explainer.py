from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

from partition_gen.operation_geometry import (
    atom_to_local,
    frame_from_geometry,
    polygon_from_face,
    polygon_to_local_payload,
    union_face_polygons,
)
from partition_gen.operation_label_pair_prior import REL_DIVIDES, REL_INSERTED_IN, REL_PARALLEL
from partition_gen.operation_role_spec import validate_role_spec
from partition_gen.parse_graph_relations import relation_refs


@dataclass(frozen=True)
class ManualRuleExplainerConfig:
    include_all_faces_of_support_labels: bool = False
    include_all_faces_of_divider_labels: bool = False
    include_soft_rules: bool = False
    split_support_by_connected_components: bool = True
    split_divider_by_connected_components: bool = True
    min_shared_length: float = 0.0
    min_area_eps: float = 1e-8


def _active_role_rules(role_spec_payload: Dict[str, object], config: ManualRuleExplainerConfig) -> List[Dict[str, object]]:
    if config.include_soft_rules:
        return list(role_spec_payload.get("relations", []))
    return [rule for rule in role_spec_payload.get("relations", []) if bool(rule.get("hard", True))]


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _faces_by_label(evidence_payload: Dict[str, object]) -> Dict[int, List[Dict[str, object]]]:
    output: Dict[int, List[Dict[str, object]]] = {}
    for face in evidence_payload.get("faces", []):
        output.setdefault(int(face.get("label", -1)), []).append(face)
    return output


def _face_area(face: Dict[str, object]) -> float:
    return float(face.get("features", {}).get("area", 0.0))


def _face_centroid(face: Dict[str, object]) -> Tuple[float, float]:
    centroid = face.get("features", {}).get("centroid")
    if isinstance(centroid, list) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    point = polygon_from_face(face).centroid
    return float(point.x), float(point.y)


def _face_arc_ids(face: Dict[str, object]) -> List[int]:
    arc_ids = []
    for ref in face.get("outer_arc_refs", []):
        arc_ids.append(int(ref["arc_id"]))
    for refs in face.get("hole_arc_refs", []):
        for ref in refs:
            arc_ids.append(int(ref["arc_id"]))
    return sorted(set(arc_ids))


def _adjacency_records(evidence_payload: Dict[str, object]) -> List[Dict[str, object]]:
    return [item for item in evidence_payload.get("adjacency", []) if len(item.get("faces", [])) == 2]


def _adjacent_face_ids_by_label(
    face_id: int,
    label: int,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency: Sequence[Dict[str, object]],
    *,
    min_shared_length: float,
) -> List[int]:
    output = []
    for item in adjacency:
        face_ids = [int(value) for value in item.get("faces", [])]
        if int(face_id) not in face_ids:
            continue
        other_id = face_ids[0] if face_ids[1] == int(face_id) else face_ids[1]
        other = faces_by_id.get(other_id)
        if not other or int(other.get("label", -1)) != int(label):
            continue
        if float(item.get("shared_length", 0.0)) < min_shared_length:
            continue
        output.append(other_id)
    return sorted(set(output))


def _shared_length_between_labels(
    face_id: int,
    label: int,
    faces_by_id: Dict[int, Dict[str, object]],
    adjacency: Sequence[Dict[str, object]],
) -> float:
    total = 0.0
    for item in adjacency:
        face_ids = [int(value) for value in item.get("faces", [])]
        if int(face_id) not in face_ids:
            continue
        other_id = face_ids[0] if face_ids[1] == int(face_id) else face_ids[1]
        other = faces_by_id.get(other_id)
        if other and int(other.get("label", -1)) == int(label):
            total += float(item.get("shared_length", 0.0))
    return float(total)


def _shared_arc_ids_between_face_sets(
    left_face_ids: Iterable[int],
    right_face_ids: Iterable[int],
    adjacency: Sequence[Dict[str, object]],
) -> List[int]:
    left = {int(value) for value in left_face_ids}
    right = {int(value) for value in right_face_ids}
    arc_ids = []
    for item in adjacency:
        face_ids = [int(value) for value in item.get("faces", [])]
        if len(face_ids) != 2:
            continue
        a, b = face_ids
        if (a in left and b in right) or (a in right and b in left):
            arc_ids.extend(int(value) for value in item.get("arc_ids", []))
    return sorted(set(arc_ids))


def _shared_length_between_face_sets(
    left_face_ids: Iterable[int],
    right_face_ids: Iterable[int],
    adjacency: Sequence[Dict[str, object]],
) -> float:
    left = {int(value) for value in left_face_ids}
    right = {int(value) for value in right_face_ids}
    total = 0.0
    for item in adjacency:
        face_ids = [int(value) for value in item.get("faces", [])]
        if len(face_ids) != 2:
            continue
        a, b = face_ids
        if (a in left and b in right) or (a in right and b in left):
            total += float(item.get("shared_length", 0.0))
    return float(total)


def _component_face_groups(
    face_ids: Iterable[int],
    adjacency: Sequence[Dict[str, object]],
    *,
    split: bool,
    min_shared_length: float,
) -> List[List[int]]:
    selected = sorted(set(int(value) for value in face_ids))
    if not selected:
        return []
    if not split:
        return [selected]

    selected_set = set(selected)
    neighbors: Dict[int, set[int]] = {face_id: set() for face_id in selected}
    for item in adjacency:
        if float(item.get("shared_length", 0.0)) < min_shared_length:
            continue
        pair = [int(value) for value in item.get("faces", [])]
        if len(pair) != 2:
            continue
        a, b = pair
        if a in selected_set and b in selected_set:
            neighbors[a].add(b)
            neighbors[b].add(a)

    groups: List[List[int]] = []
    seen: set[int] = set()
    for seed in selected:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for other in sorted(neighbors[current], reverse=True):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: (group[0], len(group)))


def _node_info_face_ids(info: Dict[str, object]) -> set[int]:
    return {int(value) for value in info.get("face_ids", [])}


def _node_infos_intersecting(
    infos: Sequence[Dict[str, object]],
    face_ids: Iterable[int],
) -> List[Dict[str, object]]:
    targets = {int(value) for value in face_ids}
    if not targets:
        return list(infos)
    output = [info for info in infos if _node_info_face_ids(info) & targets]
    return output


def _node_infos_touching(
    left_infos: Sequence[Dict[str, object]],
    right_infos: Sequence[Dict[str, object]],
    adjacency: Sequence[Dict[str, object]],
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    output = []
    for left in left_infos:
        left_face_ids = _node_info_face_ids(left)
        for right in right_infos:
            right_face_ids = _node_info_face_ids(right)
            if _shared_length_between_face_sets(left_face_ids, right_face_ids, adjacency) > 0.0:
                output.append((left, right))
    return output


def _union_geometry(face_ids: Iterable[int], faces_by_id: Dict[int, Dict[str, object]]):
    faces = [faces_by_id[face_id] for face_id in sorted(set(int(value) for value in face_ids)) if face_id in faces_by_id]
    if not faces:
        return None
    geometry = union_face_polygons(faces)
    if geometry is None or geometry.is_empty:
        return None
    return geometry


def _polygon_node(
    node_id: str,
    role: str,
    label: int,
    face_ids: Sequence[int],
    faces_by_id: Dict[int, Dict[str, object]],
    *,
    arc_ids: Sequence[int],
    config: ManualRuleExplainerConfig,
) -> Dict[str, object] | None:
    geometry = _union_geometry(face_ids, faces_by_id)
    if geometry is None or geometry.is_empty or float(geometry.area) <= config.min_area_eps:
        return None
    payload = polygon_to_local_payload(geometry, eps=config.min_area_eps)
    return {
        "id": node_id,
        "role": role,
        "label": int(label),
        "frame": payload["frame"],
        "geometry_model": "polygon_code",
        "geometry": payload["geometry"],
        "evidence": {
            "face_ids": [int(value) for value in sorted(set(face_ids))],
            "owned_face_ids": [int(value) for value in sorted(set(face_ids))],
            "referenced_face_ids": [],
            "arc_ids": [int(value) for value in sorted(set(arc_ids))],
        },
    }


def _insert_node(
    node_id: str,
    face: Dict[str, object],
    *,
    arc_ids: Sequence[int],
    config: ManualRuleExplainerConfig,
) -> Dict[str, object] | None:
    return _polygon_node(
        node_id,
        "insert_object",
        int(face.get("label", -1)),
        [int(face["id"])],
        {int(face["id"]): face},
        arc_ids=arc_ids,
        config=config,
    )


def _residual_node(
    node_id: str,
    face: Dict[str, object],
    *,
    config: ManualRuleExplainerConfig,
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
            "evidence": {
                "face_ids": [int(face["id"])],
                "owned_face_ids": [int(face["id"])],
                "referenced_face_ids": [],
                "arc_ids": _face_arc_ids(face),
            },
        }
    payload = polygon_to_local_payload(polygon, eps=config.min_area_eps)
    return {
        "id": node_id,
        "role": "residual_region",
        "label": int(face.get("label", -1)),
        "frame": payload["frame"],
        "geometry_model": "polygon_code",
        "geometry": payload["geometry"],
        "evidence": {
            "face_ids": [int(face["id"])],
            "owned_face_ids": [int(face["id"])],
            "referenced_face_ids": [],
            "arc_ids": _face_arc_ids(face),
        },
    }


def _insert_assignments(
    role_spec_payload: Dict[str, object],
    evidence_payload: Dict[str, object],
    config: ManualRuleExplainerConfig,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    faces_by_id = _faces_by_id(evidence_payload)
    faces_by_label = _faces_by_label(evidence_payload)
    adjacency = _adjacency_records(evidence_payload)
    insert_rules = [
        (int(rule["subject_label"]), int(rule["object_label"]), int(index), rule)
        for index, rule in enumerate(_active_role_rules(role_spec_payload, config))
        if str(rule.get("relation")) == REL_INSERTED_IN
    ]
    assignments: Dict[Tuple[int, int], Dict[str, object]] = {}
    for insert_label, support_label, _, _ in insert_rules:
        assignments.setdefault((support_label, insert_label), {"insert_face_ids": set(), "support_face_ids": set(), "rule_ids": []})

    for insert_label in sorted({rule[0] for rule in insert_rules}):
        for face in faces_by_label.get(insert_label, []):
            face_id = int(face["id"])
            candidates = []
            for candidate_insert_label, support_label, rule_index, rule in insert_rules:
                if candidate_insert_label != insert_label:
                    continue
                support_face_ids = _adjacent_face_ids_by_label(
                    face_id,
                    support_label,
                    faces_by_id,
                    adjacency,
                    min_shared_length=config.min_shared_length,
                )
                if not support_face_ids:
                    continue
                shared_length = _shared_length_between_labels(face_id, support_label, faces_by_id, adjacency)
                candidates.append((shared_length, -rule_index, support_label, support_face_ids, rule))
            if not candidates:
                continue
            shared_length, _, support_label, support_face_ids, rule = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
            key = (int(support_label), int(insert_label))
            bucket = assignments.setdefault(key, {"insert_face_ids": set(), "support_face_ids": set(), "rule_ids": []})
            bucket["insert_face_ids"].add(face_id)
            bucket["support_face_ids"].update(support_face_ids)
            if rule.get("id") not in bucket["rule_ids"]:
                bucket["rule_ids"].append(rule.get("id"))
    return assignments


def _collect_rule_faces(
    role_spec_payload: Dict[str, object],
    evidence_payload: Dict[str, object],
    config: ManualRuleExplainerConfig,
) -> Tuple[Dict[int, set[int]], Dict[int, set[int]], Dict[Tuple[int, int], set[int]], List[Dict[str, object]]]:
    faces_by_id = _faces_by_id(evidence_payload)
    faces_by_label = _faces_by_label(evidence_payload)
    adjacency = _adjacency_records(evidence_payload)
    support_face_ids_by_label: Dict[int, set[int]] = {}
    divider_face_ids_by_label: Dict[int, set[int]] = {}
    parallel_face_ids_by_pair: Dict[Tuple[int, int], set[int]] = {}
    relation_records: List[Dict[str, object]] = []

    insert_assignments = _insert_assignments(role_spec_payload, evidence_payload, config)
    for (support_label, insert_label), data in insert_assignments.items():
        if not data["insert_face_ids"] or not data["support_face_ids"]:
            continue
        support_face_ids_by_label.setdefault(int(support_label), set()).update(data["support_face_ids"])
        relation_records.append(
            {
                "relation_type": REL_INSERTED_IN,
                "subject_label": int(insert_label),
                "object_label": int(support_label),
                "insert_face_ids": sorted(data["insert_face_ids"]),
                "support_face_ids": sorted(data["support_face_ids"]),
                "rule_ids": list(data["rule_ids"]),
            }
        )

    for rule in _active_role_rules(role_spec_payload, config):
        relation = str(rule.get("relation"))
        subject_label = int(rule["subject_label"])
        object_label = int(rule["object_label"])
        if relation == REL_DIVIDES:
            divider_ids = set()
            support_ids = set()
            for face in faces_by_label.get(subject_label, []):
                adjacent_support_ids = _adjacent_face_ids_by_label(
                    int(face["id"]),
                    object_label,
                    faces_by_id,
                    adjacency,
                    min_shared_length=config.min_shared_length,
                )
                if adjacent_support_ids:
                    divider_ids.add(int(face["id"]))
                    support_ids.update(adjacent_support_ids)
            if config.include_all_faces_of_divider_labels:
                divider_ids.update(int(face["id"]) for face in faces_by_label.get(subject_label, []))
            if config.include_all_faces_of_support_labels and support_ids:
                support_ids.update(int(face["id"]) for face in faces_by_label.get(object_label, []))
            if divider_ids and support_ids:
                divider_face_ids_by_label.setdefault(subject_label, set()).update(divider_ids)
                support_face_ids_by_label.setdefault(object_label, set()).update(support_ids)
                relation_records.append(
                    {
                        "relation_type": REL_DIVIDES,
                        "subject_label": subject_label,
                        "object_label": object_label,
                        "divider_face_ids": sorted(divider_ids),
                        "support_face_ids": sorted(support_ids),
                        "rule_ids": [rule.get("id")],
                    }
                )
        elif relation == REL_PARALLEL:
            left_ids = set()
            right_ids = set()
            for face in faces_by_label.get(subject_label, []):
                adjacent_ids = _adjacent_face_ids_by_label(
                    int(face["id"]),
                    object_label,
                    faces_by_id,
                    adjacency,
                    min_shared_length=config.min_shared_length,
                )
                if adjacent_ids:
                    left_ids.add(int(face["id"]))
                    right_ids.update(adjacent_ids)
            if left_ids and right_ids:
                support_face_ids_by_label.setdefault(subject_label, set()).update(left_ids)
                support_face_ids_by_label.setdefault(object_label, set()).update(right_ids)
                key = tuple(sorted((subject_label, object_label)))
                parallel_face_ids_by_pair.setdefault(key, set()).update(left_ids)
                parallel_face_ids_by_pair.setdefault(key, set()).update(right_ids)
                relation_records.append(
                    {
                        "relation_type": REL_PARALLEL,
                        "subject_label": subject_label,
                        "object_label": object_label,
                        "left_face_ids": sorted(left_ids),
                        "right_face_ids": sorted(right_ids),
                        "rule_ids": [rule.get("id")],
                    }
                )
    return support_face_ids_by_label, divider_face_ids_by_label, parallel_face_ids_by_pair, relation_records


def _make_reference_support_node(
    node_id: str,
    label: int,
    face_ids: Sequence[int],
    faces_by_id: Dict[int, Dict[str, object]],
    *,
    config: ManualRuleExplainerConfig,
) -> Dict[str, object] | None:
    arc_ids = []
    for face_id in sorted(set(int(value) for value in face_ids)):
        if face_id in faces_by_id:
            arc_ids.extend(_face_arc_ids(faces_by_id[face_id]))
    node = _polygon_node(
        node_id,
        "support_region",
        int(label),
        face_ids,
        faces_by_id,
        arc_ids=arc_ids,
        config=config,
    )
    if not node:
        return None
    referenced_face_ids = sorted(set(int(value) for value in face_ids if int(value) in faces_by_id))
    node["is_reference_only"] = True
    node["evidence"]["face_ids"] = []
    node["evidence"]["owned_face_ids"] = []
    node["evidence"]["referenced_face_ids"] = referenced_face_ids
    return node


def _relation_sort_key(relation: Dict[str, object]) -> Tuple[int, str]:
    order = {"divides": 0, "inserted_in": 1, "contains": 2, "adjacent_to": 3, "has_residual": 4}
    return int(order.get(str(relation.get("type")), 99)), str(relation.get("id", ""))


def _validate_parse_graph(nodes: Sequence[Dict[str, object]], relations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    node_ids = {str(node.get("id")) for node in nodes}
    missing_refs = []
    for relation in relations:
        for key, value in relation_refs(relation):
            if str(value) not in node_ids:
                missing_refs.append({"relation": relation.get("id"), "field": key, "value": value})
    return {
        "node_reference_valid": bool(not missing_refs),
        "relation_reference_valid": bool(not missing_refs),
        "missing_relation_refs": missing_refs,
    }


def _owned_face_histogram(nodes: Sequence[Dict[str, object]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for node in nodes:
        evidence = node.get("evidence", {})
        owned_face_ids = evidence.get("owned_face_ids")
        if owned_face_ids is None:
            owned_face_ids = evidence.get("face_ids", [])
        for face_id in owned_face_ids:
            counts[int(face_id)] = int(counts.get(int(face_id), 0)) + 1
    return counts


def _referenced_face_ids(nodes: Sequence[Dict[str, object]]) -> List[int]:
    face_ids = []
    for node in nodes:
        evidence = node.get("evidence", {})
        face_ids.extend(int(value) for value in evidence.get("referenced_face_ids", []))
    return sorted(set(face_ids))


def build_manual_rule_explanation_payload(
    evidence_payload: Dict[str, object],
    role_spec_payload: Dict[str, object],
    *,
    config: ManualRuleExplainerConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or ManualRuleExplainerConfig()
    validate_role_spec(role_spec_payload)
    source_tag = source_tag or str(evidence_payload.get("source_global_approx") or evidence_payload.get("source_partition_graph") or "")
    faces_by_id = _faces_by_id(evidence_payload)
    adjacency = _adjacency_records(evidence_payload)
    support_face_ids_by_label, divider_face_ids_by_label, _, relation_records = _collect_rule_faces(
        role_spec_payload,
        evidence_payload,
        config,
    )
    labels_in_image = {int(face.get("label", -1)) for face in faces_by_id.values()}
    if len(labels_in_image) == 1 and faces_by_id:
        label = next(iter(labels_in_image))
        support_face_ids_by_label.setdefault(label, set()).update(faces_by_id)
    raw_support_face_ids_by_label = {int(label): set(face_ids) for label, face_ids in support_face_ids_by_label.items()}
    insert_owned_face_ids = {
        int(face_id)
        for record in relation_records
        if record["relation_type"] == REL_INSERTED_IN
        for face_id in record.get("insert_face_ids", [])
    }
    divider_owned_face_ids = {
        int(face_id)
        for record in relation_records
        if record["relation_type"] == REL_DIVIDES
        for face_id in record.get("divider_face_ids", [])
    }
    non_support_owned_face_ids = insert_owned_face_ids | divider_owned_face_ids
    for label in list(support_face_ids_by_label):
        support_face_ids_by_label[label].difference_update(non_support_owned_face_ids)
        if not support_face_ids_by_label[label]:
            del support_face_ids_by_label[label]

    nodes: List[Dict[str, object]] = []
    relations: List[Dict[str, object]] = []
    residuals: List[Dict[str, object]] = []
    support_node_infos_by_label: Dict[int, List[Dict[str, object]]] = {}
    divider_node_infos_by_label: Dict[int, List[Dict[str, object]]] = {}
    insert_group_infos_by_label: Dict[int, List[Dict[str, object]]] = {}
    reference_support_node_infos_by_label: Dict[int, List[Dict[str, object]]] = {}
    owned_face_ids: set[int] = set()
    selected_explanations: List[Dict[str, object]] = []

    support_node_counter = 0
    divider_node_counter = 0
    support_component_count = 0
    divider_component_count = 0

    def add_node_info(
        mapping: Dict[int, List[Dict[str, object]]],
        label: int,
        node_id: str,
        face_ids: Sequence[int],
        *,
        role: str,
        reference_only: bool = False,
    ) -> None:
        mapping.setdefault(int(label), []).append(
            {
                "id": str(node_id),
                "label": int(label),
                "role": str(role),
                "face_ids": set(int(value) for value in face_ids),
                "reference_only": bool(reference_only),
            }
        )

    for label in sorted(support_face_ids_by_label):
        groups = _component_face_groups(
            support_face_ids_by_label[label],
            adjacency,
            split=config.split_support_by_connected_components,
            min_shared_length=config.min_shared_length,
        )
        support_component_count += len(groups)
        for face_ids in groups:
            arc_ids = []
            for face_id in face_ids:
                if face_id in faces_by_id:
                    arc_ids.extend(_face_arc_ids(faces_by_id[face_id]))
            node_id = f"support_{support_node_counter}"
            support_node_counter += 1
            node = _polygon_node(node_id, "support_region", label, face_ids, faces_by_id, arc_ids=arc_ids, config=config)
            if node:
                nodes.append(node)
                add_node_info(support_node_infos_by_label, label, node_id, face_ids, role="support_region")
                owned_face_ids.update(face_ids)

    for label in sorted(divider_face_ids_by_label):
        groups = _component_face_groups(
            divider_face_ids_by_label[label],
            adjacency,
            split=config.split_divider_by_connected_components,
            min_shared_length=config.min_shared_length,
        )
        divider_component_count += len(groups)
        for face_ids in groups:
            arc_ids = []
            for face_id in face_ids:
                if face_id in faces_by_id:
                    arc_ids.extend(_face_arc_ids(faces_by_id[face_id]))
            node_id = f"divider_{divider_node_counter}"
            divider_node_counter += 1
            node = _polygon_node(node_id, "divider_region", label, face_ids, faces_by_id, arc_ids=arc_ids, config=config)
            if node:
                nodes.append(node)
                add_node_info(divider_node_infos_by_label, label, node_id, face_ids, role="divider_region")
                owned_face_ids.update(face_ids)

    def ensure_support_node_infos_for_label(label: int, fallback_face_ids: Sequence[int]) -> List[Dict[str, object]]:
        label = int(label)
        existing = _node_infos_intersecting(support_node_infos_by_label.get(label, []), fallback_face_ids)
        if existing:
            return existing
        existing_refs = _node_infos_intersecting(reference_support_node_infos_by_label.get(label, []), fallback_face_ids)
        if existing_refs:
            return existing_refs
        face_ids = sorted(set(raw_support_face_ids_by_label.get(label, set())) | set(int(value) for value in fallback_face_ids))
        if not face_ids:
            return []
        groups = _component_face_groups(
            face_ids,
            adjacency,
            split=config.split_support_by_connected_components,
            min_shared_length=config.min_shared_length,
        )
        fallback_set = {int(value) for value in fallback_face_ids}
        created: List[Dict[str, object]] = []
        for group in groups:
            if fallback_set and not (set(group) & fallback_set):
                continue
            node_id = f"support_ref_{sum(len(items) for items in reference_support_node_infos_by_label.values())}"
            node = _make_reference_support_node(node_id, label, group, faces_by_id, config=config)
            if not node:
                continue
            nodes.append(node)
            add_node_info(
                reference_support_node_infos_by_label,
                label,
                node_id,
                group,
                role="support_region",
                reference_only=True,
            )
            created.append(reference_support_node_infos_by_label[label][-1])
        return created

    def relation_endpoint_infos_for_label(label: int, preferred: str, fallback_face_ids: Sequence[int]) -> List[Dict[str, object]]:
        label = int(label)
        if preferred == "divider":
            infos = _node_infos_intersecting(divider_node_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            infos = _node_infos_intersecting(insert_group_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            infos = _node_infos_intersecting(support_node_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            return ensure_support_node_infos_for_label(label, fallback_face_ids)
        if preferred == "insert_group":
            infos = _node_infos_intersecting(insert_group_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            infos = _node_infos_intersecting(support_node_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            infos = _node_infos_intersecting(divider_node_infos_by_label.get(label, []), fallback_face_ids)
            if infos:
                return infos
            return ensure_support_node_infos_for_label(label, fallback_face_ids)

        infos = _node_infos_intersecting(support_node_infos_by_label.get(label, []), fallback_face_ids)
        if infos:
            return infos
        infos = _node_infos_intersecting(insert_group_infos_by_label.get(label, []), fallback_face_ids)
        if infos:
            return infos
        infos = _node_infos_intersecting(divider_node_infos_by_label.get(label, []), fallback_face_ids)
        if infos:
            return infos
        return ensure_support_node_infos_for_label(label, fallback_face_ids)

    def best_support_info_for_face(face_id: int, support_infos: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
        if not support_infos:
            return None
        scored = []
        for info in support_infos:
            shared_length = _shared_length_between_face_sets([face_id], _node_info_face_ids(info), adjacency)
            scored.append((shared_length, -min(_node_info_face_ids(info) or {0}), info))
        return max(scored, key=lambda item: item[:2])[2]

    insert_records = [record for record in relation_records if record["relation_type"] == REL_INSERTED_IN]
    insert_node_counter = 0
    insert_group_counter = 0
    for record in sorted(insert_records, key=lambda item: (item["object_label"], item["subject_label"])):
        support_label = int(record["object_label"])
        insert_label = int(record["subject_label"])
        support_infos = ensure_support_node_infos_for_label(support_label, record.get("support_face_ids", []))
        if not support_infos:
            continue
        insert_face_ids = [face_id for face_id in record["insert_face_ids"] if face_id in faces_by_id]
        if not insert_face_ids:
            continue
        assigned_face_ids_by_support_id: Dict[str, List[int]] = {}
        support_info_by_id = {str(info["id"]): info for info in support_infos}
        for face_id in insert_face_ids:
            support_info = best_support_info_for_face(face_id, support_infos)
            if not support_info:
                continue
            assigned_face_ids_by_support_id.setdefault(str(support_info["id"]), []).append(face_id)

        for support_node_id in sorted(assigned_face_ids_by_support_id):
            group_insert_face_ids = sorted(
                assigned_face_ids_by_support_id[support_node_id],
                key=lambda value: (_face_centroid(faces_by_id[value])[1], _face_centroid(faces_by_id[value])[0], value),
            )
            if not group_insert_face_ids:
                continue
            group_id = f"insert_group_{insert_group_counter}"
            insert_group_counter += 1
            group_node = {
                "id": group_id,
                "role": "insert_object_group",
                "label": insert_label,
                "geometry_model": "none",
                "support_id": support_node_id,
                "children": [],
                "count": int(len(group_insert_face_ids)),
                "evidence": {
                    "face_ids": [],
                    "owned_face_ids": [],
                    "referenced_face_ids": sorted(group_insert_face_ids),
                    "arc_ids": [],
                },
            }
            add_node_info(
                insert_group_infos_by_label,
                insert_label,
                group_id,
                group_insert_face_ids,
                role="insert_object_group",
                reference_only=True,
            )
            nodes.append(group_node)
            relations.append(
                {
                    "id": f"relation_{len(relations)}",
                    "type": "inserted_in",
                    "object": group_id,
                    "container": support_node_id,
                    "support": support_node_id,
                    "face_ids": sorted(group_insert_face_ids),
                    "rule_ids": list(record.get("rule_ids", [])),
                }
            )
            for face_id in group_insert_face_ids:
                face = faces_by_id[face_id]
                node_id = f"insert_{insert_node_counter}"
                insert_node_counter += 1
                node = _insert_node(node_id, face, arc_ids=_face_arc_ids(face), config=config)
                if not node:
                    continue
                node["parent_group"] = group_id
                node["support_id"] = support_node_id
                nodes.append(node)
                group_node["children"].append(node_id)
                owned_face_ids.add(face_id)
                relations.append(
                    {
                        "id": f"relation_{len(relations)}",
                        "type": "contains",
                        "parent": group_id,
                        "child": node_id,
                        "face_ids": [face_id],
                    }
                )

    for record in sorted([item for item in relation_records if item["relation_type"] == REL_DIVIDES], key=lambda item: (item["subject_label"], item["object_label"])):
        divider_infos = relation_endpoint_infos_for_label(
            int(record["subject_label"]),
            "divider",
            record.get("divider_face_ids", []),
        )
        support_infos = relation_endpoint_infos_for_label(
            int(record["object_label"]),
            "support",
            record.get("support_face_ids", []),
        )
        if not divider_infos or not support_infos:
            continue
        for divider_info, support_info in _node_infos_touching(divider_infos, support_infos, adjacency):
            divider_face_ids = sorted(_node_info_face_ids(divider_info))
            support_face_ids = sorted(_node_info_face_ids(support_info))
            arc_ids = _shared_arc_ids_between_face_sets(divider_face_ids, support_face_ids, adjacency)
            relations.append(
                {
                    "id": f"relation_{len(relations)}",
                    "type": "divides",
                    "divider": divider_info["id"],
                    "target": support_info["id"],
                    "support": support_info["id"],
                    "induced_face_ids": support_face_ids,
                    "divider_face_ids": divider_face_ids,
                    "arc_ids": arc_ids,
                    "rule_ids": list(record.get("rule_ids", [])),
                }
            )

    for record in sorted([item for item in relation_records if item["relation_type"] == REL_PARALLEL], key=lambda item: (item["subject_label"], item["object_label"])):
        left_infos = relation_endpoint_infos_for_label(int(record["subject_label"]), "support", record.get("left_face_ids", []))
        right_infos = relation_endpoint_infos_for_label(int(record["object_label"]), "support", record.get("right_face_ids", []))
        if not left_infos or not right_infos:
            continue
        for left_info, right_info in _node_infos_touching(left_infos, right_infos, adjacency):
            left_face_ids = sorted(_node_info_face_ids(left_info))
            right_face_ids = sorted(_node_info_face_ids(right_info))
            arc_ids = _shared_arc_ids_between_face_sets(left_face_ids, right_face_ids, adjacency)
            relations.append(
                {
                    "id": f"relation_{len(relations)}",
                    "type": "adjacent_to",
                    "faces": [left_info["id"], right_info["id"]],
                    "arc_ids": arc_ids,
                    "rule_ids": list(record.get("rule_ids", [])),
                }
            )

    all_face_ids = set(faces_by_id)
    residual_face_ids = sorted(all_face_ids - owned_face_ids, key=lambda value: (-_face_area(faces_by_id[value]), value))
    for index, face_id in enumerate(residual_face_ids):
        node_id = f"residual_{index}"
        node = _residual_node(node_id, faces_by_id[face_id], config=config)
        nodes.append(node)
        residuals.append(
            {
                "node_id": node_id,
                "face_ids": [face_id],
                "area": _face_area(faces_by_id[face_id]),
                "reason": "no_explicit_manual_rule_match",
            }
        )

    relations.sort(key=_relation_sort_key)
    for index, relation in enumerate(relations):
        relation["id"] = f"relation_{index}"

    owned_face_counts = _owned_face_histogram(nodes)
    duplicate_owned_face_ids = sorted(face_id for face_id, count in owned_face_counts.items() if count > 1)
    referenced_face_ids_only = _referenced_face_ids(nodes)
    unowned_face_ids = sorted(face_id for face_id in all_face_ids if owned_face_counts.get(face_id, 0) == 0)
    validation_refs = _validate_parse_graph(nodes, relations)
    evidence_validation = evidence_payload.get("evidence_validation", {})
    input_valid = bool(evidence_validation.get("usable_for_explainer", evidence_validation.get("is_valid", True)))
    all_faces_owned_once = bool(
        all(owned_face_counts.get(face_id, 0) == 1 for face_id in all_face_ids)
        and not duplicate_owned_face_ids
        and not unowned_face_ids
    )
    validation = {
        "is_valid": bool(input_valid and validation_refs["relation_reference_valid"] and all_faces_owned_once),
        "input_evidence_valid": input_valid,
        "all_faces_owned_exactly_once": bool(all_faces_owned_once),
        "all_faces_referenced_or_residual": bool(all_faces_owned_once),
        "ownership_by_face": {str(face_id): int(owned_face_counts.get(face_id, 0)) for face_id in sorted(all_face_ids)},
        "duplicate_owned_face_ids": duplicate_owned_face_ids,
        "unowned_face_ids": unowned_face_ids,
        "referenced_face_ids": referenced_face_ids_only,
        "node_reference_valid": validation_refs["node_reference_valid"],
        "relation_reference_valid": validation_refs["relation_reference_valid"],
        "missing_relation_refs": validation_refs["missing_relation_refs"],
        "render_validation": {"status": "not_implemented"},
    }

    operation_histogram = {
        "manual_inserted_in": sum(1 for item in relation_records if item["relation_type"] == REL_INSERTED_IN),
        "manual_divides": sum(1 for item in relation_records if item["relation_type"] == REL_DIVIDES),
        "manual_parallel": sum(1 for item in relation_records if item["relation_type"] == REL_PARALLEL),
        "residual": len(residual_face_ids),
    }
    role_histogram: Dict[str, int] = {}
    for node in nodes:
        role = str(node.get("role"))
        role_histogram[role] = int(role_histogram.get(role, 0)) + 1
    total_area = float(evidence_payload.get("statistics", {}).get("total_face_area", sum(_face_area(face) for face in faces_by_id.values())))
    residual_area = sum(_face_area(faces_by_id[face_id]) for face_id in residual_face_ids)

    for record in relation_records:
        selected_explanations.append({"type": f"manual_{str(record['relation_type']).lower()}", **record})

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
            "target_profile": "manual_role_spec_parse_graph_v1",
            "code_length": None,
            "render_iou": None,
            "valid": bool(validation["is_valid"]),
            "render_validation": {
                "status": "not_implemented",
                "reason": "manual rule parse graph renderer is not implemented yet",
            },
        },
    }

    diagnostics = {
        "profile": "manual_role_spec_parse_graph_v1",
        "face_count": int(len(faces_by_id)),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "residual_face_count": int(len(residual_face_ids)),
        "residual_area_ratio": float(residual_area / total_area) if total_area > config.min_area_eps else 0.0,
        "operation_histogram": operation_histogram,
        "role_histogram": role_histogram,
        "role_spec_name": role_spec_payload.get("name"),
        "role_spec_relation_count": int(len(role_spec_payload.get("relations", []))),
        "role_spec_semantics": "direct_parse_graph_rules",
        "active_role_spec_relation_count": int(len(_active_role_rules(role_spec_payload, config))),
        "soft_role_spec_relation_count": int(
            sum(1 for rule in role_spec_payload.get("relations", []) if not bool(rule.get("hard", True)))
        ),
        "include_soft_rules": bool(config.include_soft_rules),
        "split_support_by_connected_components": bool(config.split_support_by_connected_components),
        "split_divider_by_connected_components": bool(config.split_divider_by_connected_components),
        "support_component_count": int(support_component_count),
        "divider_component_count": int(divider_component_count),
        "support_node_count": int(sum(len(items) for items in support_node_infos_by_label.values())),
        "divider_node_count": int(sum(len(items) for items in divider_node_infos_by_label.values())),
        "insert_group_count": int(sum(len(items) for items in insert_group_infos_by_label.values())),
        "reference_support_node_count": int(sum(len(items) for items in reference_support_node_infos_by_label.values())),
        "owned_face_count": int(len(owned_face_counts)),
        "referenced_face_count": int(len(referenced_face_ids_only)),
        "duplicate_owned_face_ids": duplicate_owned_face_ids,
        "duplicate_owned_face_count": int(len(duplicate_owned_face_ids)),
        "unowned_face_ids": unowned_face_ids,
        "unowned_face_count": int(len(unowned_face_ids)),
        "duplicate_evidence_face_ids": duplicate_owned_face_ids,
        "duplicate_evidence_face_count": int(len(duplicate_owned_face_ids)),
        "selection_method": "manual_role_spec_direct",
        "uses_ortools": False,
        "uses_candidate_search": False,
    }

    return {
        "format": "maskgen_manual_rule_explanation_v1",
        "source_evidence": source_tag,
        "explainer_profile": "manual_role_spec_parse_graph_v1",
        "selected_explanations": selected_explanations,
        "generator_target": generator_target,
        "diagnostics": diagnostics,
        "validation": validation,
        "role_spec": {
            "format": role_spec_payload.get("format"),
            "name": role_spec_payload.get("name"),
            "semantics": "direct_parse_graph_rules",
            "relation_count": int(len(role_spec_payload.get("relations", []))),
            "active_relation_count": int(len(_active_role_rules(role_spec_payload, config))),
            "soft_relation_count": int(sum(1 for rule in role_spec_payload.get("relations", []) if not bool(rule.get("hard", True)))),
            "include_soft_rules": bool(config.include_soft_rules),
            "defaults": role_spec_payload.get("defaults", {}),
        },
        "config": asdict(config),
    }
