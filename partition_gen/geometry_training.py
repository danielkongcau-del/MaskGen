from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask], weight=weight)


def compute_geometry_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    support_class_weights: torch.Tensor | None = None,
    vertex_count_class_weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    face_mask = batch["face_mask"]
    geometry_support = batch["geometry_support"]
    vertex_counts = batch["vertex_counts"]
    vertices = batch["vertices"]
    vertex_mask = batch["vertex_mask"]
    hole_counts = batch["hole_counts"]
    hole_vertex_counts = batch["hole_vertex_counts"]
    hole_vertices = batch["hole_vertices"]
    hole_mask = batch["hole_mask"]
    hole_vertex_mask = batch["hole_vertex_mask"]

    losses: Dict[str, torch.Tensor] = {}
    losses["support"] = _masked_ce_loss(
        outputs["support_logits"],
        geometry_support,
        face_mask,
        support_class_weights,
    )

    supported_mask = face_mask & geometry_support.bool()
    if supported_mask.sum() > 0:
        losses["vertex_count"] = F.cross_entropy(
            outputs["vertex_count_logits"][supported_mask],
            vertex_counts[supported_mask],
            weight=vertex_count_class_weights,
        )
    else:
        losses["vertex_count"] = outputs["vertex_count_logits"].sum() * 0.0

    coord_mask = vertex_mask & geometry_support.bool().unsqueeze(-1)
    if coord_mask.sum() > 0:
        losses["coords"] = F.smooth_l1_loss(
            outputs["vertex_coords"][coord_mask],
            vertices[coord_mask],
        )
    else:
        losses["coords"] = outputs["vertex_coords"].sum() * 0.0

    supported_mask = face_mask & geometry_support.bool()
    if outputs["hole_count_logits"].shape[-1] > 0 and supported_mask.sum() > 0:
        losses["hole_count"] = F.cross_entropy(
            outputs["hole_count_logits"][supported_mask],
            hole_counts[supported_mask],
        )
    else:
        losses["hole_count"] = outputs["vertex_coords"].sum() * 0.0

    hole_slot_mask = supported_mask.unsqueeze(-1).expand_as(hole_vertex_counts)
    if outputs["hole_vertex_count_logits"].numel() > 0 and hole_slot_mask.sum() > 0:
        losses["hole_vertex_count"] = F.cross_entropy(
            outputs["hole_vertex_count_logits"][hole_slot_mask],
            hole_vertex_counts[hole_slot_mask],
        )
    else:
        losses["hole_vertex_count"] = outputs["vertex_coords"].sum() * 0.0

    hole_coord_mask = hole_vertex_mask & hole_mask.unsqueeze(-1) & geometry_support.bool().unsqueeze(-1).unsqueeze(-1)
    if outputs["hole_vertex_coords"].numel() > 0 and hole_coord_mask.sum() > 0:
        losses["hole_coords"] = F.smooth_l1_loss(
            outputs["hole_vertex_coords"][hole_coord_mask],
            hole_vertices[hole_coord_mask],
        )
    else:
        losses["hole_coords"] = outputs["vertex_coords"].sum() * 0.0

    losses["total"] = (
        losses["support"]
        + losses["vertex_count"]
        + 2.0 * losses["coords"]
        + 0.5 * losses["hole_count"]
        + 0.5 * losses["hole_vertex_count"]
        + 1.0 * losses["hole_coords"]
    )
    return losses
