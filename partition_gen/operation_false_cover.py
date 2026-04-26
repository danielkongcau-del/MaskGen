from __future__ import annotations

from typing import Dict, List

from partition_gen.operation_geometry import polygon_from_face
from partition_gen.operation_types import RESIDUAL, OperationCandidate, OperationExplainerConfig


def _empty_false_cover() -> Dict[str, object]:
    return {
        "enabled": False,
        "area": 0.0,
        "ratio": 0.0,
        "cost": 0.0,
        "overlapped_face_ids": [],
        "overlapped_labels": [],
        "overlaps": [],
        "by_face": [],
        "hard_invalid": False,
    }


def compute_false_cover_diagnostics(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    if candidate.operation_type == RESIDUAL or not candidate.metadata:
        return _empty_false_cover()
    latent = candidate.metadata.get("latent_geometry")
    if not isinstance(latent, dict):
        return _empty_false_cover()
    geometry = latent.get("geometry")
    if geometry is None or geometry.is_empty or float(geometry.area) <= config.min_area_eps:
        return _empty_false_cover()

    allowed = set(int(face_id) for face_id in candidate.covered_face_ids)
    false_area = 0.0
    overlaps: List[Dict[str, object]] = []
    labels = set()
    for face in evidence_payload.get("faces", []):
        face_id = int(face["id"])
        if face_id in allowed:
            continue
        polygon = polygon_from_face(face)
        if polygon.is_empty:
            continue
        intersection = geometry.intersection(polygon)
        area = float(intersection.area) if not intersection.is_empty else 0.0
        if area <= config.min_area_eps:
            continue
        false_area += area
        label = int(face.get("label", -1))
        labels.add(label)
        overlaps.append(
            {
                "face_id": face_id,
                "label": label,
                "area": area,
            }
        )

    ratio = float(false_area / max(float(geometry.area), config.min_area_eps))
    hard_invalid = bool(ratio > config.false_cover_ratio_invalid)
    return {
        "enabled": True,
        "area": float(false_area),
        "ratio": ratio,
        "cost": 0.0,
        "overlapped_face_ids": [item["face_id"] for item in overlaps],
        "overlapped_labels": sorted(labels),
        "overlaps": overlaps,
        "by_face": overlaps,
        "hard_invalid": hard_invalid,
    }


def compute_false_cover_penalty(
    candidate: OperationCandidate,
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    diagnostics = compute_false_cover_diagnostics(candidate, evidence_payload, config)
    ratio = float(diagnostics.get("ratio", 0.0))
    area = float(diagnostics.get("area", 0.0))
    cost = float(area * config.false_cover_area_weight + ratio * config.false_cover_ratio_weight)
    if ratio > config.max_false_cover_ratio:
        cost += float((ratio - config.max_false_cover_ratio) * config.false_cover_ratio_weight)
    diagnostics["cost"] = cost
    diagnostics["hard_invalid"] = bool(ratio > config.hard_invalid_false_cover_ratio)
    return diagnostics
