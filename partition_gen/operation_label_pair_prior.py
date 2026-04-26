from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from partition_gen.operation_types import OperationExplainerConfig


REL_DIVIDES = "DIVIDES"
REL_INSERTED_IN = "INSERTED_IN"
REL_PARALLEL = "PARALLEL"
REL_UNDECIDED = "UNDECIDED"


def _faces_by_id(evidence_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(face["id"]): face for face in evidence_payload.get("faces", [])}


def _face_label(face: Dict[str, object]) -> int:
    return int(face.get("label", -1))


def _face_area(face: Dict[str, object]) -> float:
    return float(face.get("features", {}).get("area", 0.0))


def _face_perimeter(face: Dict[str, object]) -> float:
    return float(face.get("features", {}).get("perimeter", 0.0))


def _pair_key(left_label: int, right_label: int) -> str:
    a, b = sorted((int(left_label), int(right_label)))
    return f"{a}:{b}"


def _iter_cross_label_adjacency(evidence_payload: Dict[str, object]) -> Iterable[Tuple[int, int, float]]:
    for adjacency in evidence_payload.get("adjacency", []):
        face_ids = [int(value) for value in adjacency.get("faces", [])]
        labels = [int(value) for value in adjacency.get("labels", [])]
        if len(face_ids) != 2 or len(labels) != 2 or labels[0] == labels[1]:
            continue
        yield face_ids[0], face_ids[1], float(adjacency.get("shared_length", 0.0))


def _direction_stats(
    *,
    subject_label: int,
    object_label: int,
    pair_adjacencies: List[Tuple[int, int, float]],
    faces_by_id: Dict[int, Dict[str, object]],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    subject_ids = set()
    object_ids = set()
    shared_by_subject: Dict[int, float] = defaultdict(float)
    object_ids_by_subject: Dict[int, set[int]] = defaultdict(set)
    total_shared = 0.0

    for left_id, right_id, shared_length in pair_adjacencies:
        left = faces_by_id.get(left_id)
        right = faces_by_id.get(right_id)
        if left is None or right is None:
            continue
        left_label = _face_label(left)
        right_label = _face_label(right)
        if left_label == subject_label and right_label == object_label:
            subject_id, object_id = left_id, right_id
        elif left_label == object_label and right_label == subject_label:
            subject_id, object_id = right_id, left_id
        else:
            continue
        subject_ids.add(subject_id)
        object_ids.add(object_id)
        object_ids_by_subject[subject_id].add(object_id)
        shared_by_subject[subject_id] += shared_length
        total_shared += shared_length

    subject_area = sum(_face_area(faces_by_id[face_id]) for face_id in subject_ids if face_id in faces_by_id)
    object_area = sum(_face_area(faces_by_id[face_id]) for face_id in object_ids if face_id in faces_by_id)
    area_ratio = subject_area / max(object_area, config.min_area_eps)
    wrapped_count = 0
    wrapped_fraction_sum = 0.0
    for subject_id in subject_ids:
        perimeter = _face_perimeter(faces_by_id[subject_id])
        fraction = shared_by_subject[subject_id] / max(perimeter, config.min_area_eps)
        wrapped_fraction_sum += fraction
        if fraction >= config.label_pair_insert_boundary_fraction:
            wrapped_count += 1
    subject_count = len(subject_ids)
    object_fragment_count = len(object_ids)
    max_object_neighbors_per_subject = max((len(values) for values in object_ids_by_subject.values()), default=0)
    mean_object_neighbors_per_subject = (
        sum(len(values) for values in object_ids_by_subject.values()) / max(subject_count, 1)
    )
    wrapped_fraction_mean = wrapped_fraction_sum / max(subject_count, 1)

    insert_score = 0.0
    if subject_count and object_fragment_count and area_ratio <= config.label_pair_insert_max_area_ratio:
        insert_score = (wrapped_count / max(subject_count, 1)) + min(wrapped_fraction_mean, 1.0)

    divide_score = 0.0
    if (
        object_fragment_count >= config.label_pair_min_divided_fragments
        and max_object_neighbors_per_subject >= config.label_pair_min_divided_fragments
        and area_ratio <= config.label_pair_divider_max_area_ratio
    ):
        divide_score = (
            float(max_object_neighbors_per_subject - 1)
            + max(0.0, float(mean_object_neighbors_per_subject - 1.0))
            + min(total_shared / max(subject_area, config.min_area_eps), 1.0)
        )

    return {
        "subject_label": int(subject_label),
        "object_label": int(object_label),
        "subject_touching_count": int(subject_count),
        "object_touching_count": int(object_fragment_count),
        "max_object_neighbors_per_subject": int(max_object_neighbors_per_subject),
        "mean_object_neighbors_per_subject": float(mean_object_neighbors_per_subject),
        "subject_area": float(subject_area),
        "object_area": float(object_area),
        "subject_to_object_area_ratio": float(area_ratio),
        "total_shared_length": float(total_shared),
        "wrapped_subject_count": int(wrapped_count),
        "wrapped_fraction_mean": float(wrapped_fraction_mean),
        "insert_score": float(insert_score),
        "divide_score": float(divide_score),
    }


def _select_relation(a_stats: Dict[str, object], b_stats: Dict[str, object]) -> Dict[str, object]:
    candidates = [
        (float(a_stats["divide_score"]), REL_DIVIDES, int(a_stats["subject_label"]), int(a_stats["object_label"])),
        (float(b_stats["divide_score"]), REL_DIVIDES, int(b_stats["subject_label"]), int(b_stats["object_label"])),
        (float(a_stats["insert_score"]), REL_INSERTED_IN, int(a_stats["subject_label"]), int(a_stats["object_label"])),
        (float(b_stats["insert_score"]), REL_INSERTED_IN, int(b_stats["subject_label"]), int(b_stats["object_label"])),
    ]
    best_score, relation_type, subject_label, object_label = max(candidates, key=lambda item: (item[0], item[1]))
    parallel_score = min(float(a_stats["object_touching_count"]), float(b_stats["object_touching_count"]))
    if best_score <= 0.0 and parallel_score > 0:
        return {
            "relation_type": REL_PARALLEL,
            "relation": "PARALLEL",
            "subject_label": int(min(int(a_stats["subject_label"]), int(b_stats["subject_label"]))),
            "object_label": int(max(int(a_stats["subject_label"]), int(b_stats["subject_label"]))),
            "score": float(parallel_score),
            "confidence": 0.25,
        }
    if best_score <= 0.0:
        return {
            "relation_type": REL_UNDECIDED,
            "relation": "UNDECIDED",
            "subject_label": int(min(int(a_stats["subject_label"]), int(b_stats["subject_label"]))),
            "object_label": int(max(int(a_stats["subject_label"]), int(b_stats["subject_label"]))),
            "score": 0.0,
            "confidence": 0.0,
        }
    relation = f"{subject_label}_{relation_type}_{object_label}"
    runner_up = sorted((item[0] for item in candidates), reverse=True)[1] if len(candidates) > 1 else 0.0
    confidence = (best_score - runner_up) / max(best_score, 1e-8)
    return {
        "relation_type": relation_type,
        "relation": relation,
        "subject_label": int(subject_label),
        "object_label": int(object_label),
        "score": float(best_score),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def build_label_pair_relation_priors(
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    faces_by_id = _faces_by_id(evidence_payload)
    pair_adjacencies: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    labels_by_key: Dict[str, Tuple[int, int]] = {}
    for left_id, right_id, shared_length in _iter_cross_label_adjacency(evidence_payload):
        left = faces_by_id.get(left_id)
        right = faces_by_id.get(right_id)
        if left is None or right is None:
            continue
        left_label = _face_label(left)
        right_label = _face_label(right)
        key = _pair_key(left_label, right_label)
        labels_by_key[key] = tuple(sorted((left_label, right_label)))
        pair_adjacencies[key].append((left_id, right_id, shared_length))

    pairs = []
    for key in sorted(pair_adjacencies):
        a_label, b_label = labels_by_key[key]
        a_stats = _direction_stats(
            subject_label=a_label,
            object_label=b_label,
            pair_adjacencies=pair_adjacencies[key],
            faces_by_id=faces_by_id,
            config=config,
        )
        b_stats = _direction_stats(
            subject_label=b_label,
            object_label=a_label,
            pair_adjacencies=pair_adjacencies[key],
            faces_by_id=faces_by_id,
            config=config,
        )
        selected = _select_relation(a_stats, b_stats)
        pairs.append(
            {
                "key": key,
                "labels": [int(a_label), int(b_label)],
                "selected": selected,
                "scores": {
                    f"{a_label}_divides_{b_label}": float(a_stats["divide_score"]),
                    f"{b_label}_divides_{a_label}": float(b_stats["divide_score"]),
                    f"{a_label}_inserted_in_{b_label}": float(a_stats["insert_score"]),
                    f"{b_label}_inserted_in_{a_label}": float(b_stats["insert_score"]),
                },
                "direction_stats": {
                    f"{a_label}->{b_label}": a_stats,
                    f"{b_label}->{a_label}": b_stats,
                },
            }
        )

    histogram: Dict[str, int] = {}
    for pair in pairs:
        relation_type = str(pair["selected"]["relation_type"])
        histogram[relation_type] = int(histogram.get(relation_type, 0)) + 1
    return {
        "format": "maskgen_label_pair_relation_priors_v1",
        "pairs": pairs,
        "relation_histogram": dict(sorted(histogram.items())),
    }


def label_pair_lookup(label_pair_prior_payload: Dict[str, object] | None) -> Dict[str, Dict[str, object]]:
    if not label_pair_prior_payload:
        return {}
    output = {}
    for pair in label_pair_prior_payload.get("pairs", []):
        labels = pair.get("labels", [])
        if len(labels) != 2:
            continue
        output[_pair_key(int(labels[0]), int(labels[1]))] = pair
    return output


def relation_for_labels(
    label_pair_prior_payload: Dict[str, object] | None,
    subject_label: int,
    object_label: int,
) -> Dict[str, object] | None:
    pair = label_pair_lookup(label_pair_prior_payload).get(_pair_key(subject_label, object_label))
    if not pair:
        return None
    selected = pair.get("selected", {})
    if not selected or selected.get("relation_type") in {REL_UNDECIDED, None}:
        return None
    return selected
