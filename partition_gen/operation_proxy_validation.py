from __future__ import annotations

from typing import Dict, Sequence

from partition_gen.operation_types import RESIDUAL, OperationCandidate, OperationExplainerConfig


def _false_cover(candidate: OperationCandidate) -> Dict[str, object]:
    return candidate.cost_breakdown.get("false_cover", {}) if candidate.cost_breakdown else {}


def validate_selected_operations_proxy(
    selected_candidates: Sequence[OperationCandidate],
    evidence_payload: Dict[str, object],
    config: OperationExplainerConfig,
) -> Dict[str, object]:
    del evidence_payload, config
    false_area_total = 0.0
    false_ratio_max = 0.0
    latent_count = 0
    invalid_count = 0
    residual_count = 0
    for candidate in selected_candidates:
        if candidate.operation_type == RESIDUAL:
            residual_count += 1
        if not candidate.valid:
            invalid_count += 1
        latent = candidate.metadata.get("latent_geometry") if candidate.metadata else None
        if isinstance(latent, dict):
            latent_count += 1
        false_cover = _false_cover(candidate)
        false_area_total += float(false_cover.get("area", 0.0))
        false_ratio_max = max(false_ratio_max, float(false_cover.get("ratio", 0.0)))
    return {
        "status": "proxy_only",
        "selected_operation_count": int(len(selected_candidates)),
        "selected_non_residual_count": int(len(selected_candidates) - residual_count),
        "selected_residual_count": int(residual_count),
        "latent_geometry_count": int(latent_count),
        "latent_false_cover_area_total": float(false_area_total),
        "latent_false_cover_ratio_max": float(false_ratio_max),
        "selected_invalid_count": int(invalid_count),
        "notes": [
            "This is not a renderer and does not compute render_iou.",
        ],
    }
