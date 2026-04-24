from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def _masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    logits = logits[mask]
    targets = targets[mask]
    return F.cross_entropy(logits, targets)


def compute_topology_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    face_mask = batch["face_mask"]
    node_features = batch["node_features"]
    prev_neighbor_mask = batch["prev_neighbor_mask"]
    prev_neighbor_indices = batch["prev_neighbor_indices"]
    prev_neighbor_tokens = batch["prev_neighbor_tokens"]
    prev_counts = prev_neighbor_mask.sum(dim=-1)
    active_slots = prev_neighbor_mask.shape[-1]

    losses: Dict[str, torch.Tensor] = {}

    losses["face_exists"] = F.cross_entropy(
        outputs["face_exists_logits"].reshape(-1, outputs["face_exists_logits"].shape[-1]),
        face_mask.reshape(-1).long(),
    )

    feature_losses: List[torch.Tensor] = []
    for feature_index, logits in enumerate(outputs["node_feature_logits"]):
        feature_losses.append(
            _masked_ce_loss(logits, node_features[:, :, feature_index], face_mask)
        )
    losses["node_features"] = torch.stack(feature_losses).mean() if feature_losses else node_features.sum() * 0.0

    losses["prev_count"] = _masked_ce_loss(
        outputs["prev_count_logits"],
        prev_counts,
        face_mask,
    )

    if prev_neighbor_mask.sum() > 0:
        pointer_logits = outputs["prev_neighbor_logits"][:, :, :active_slots, :][prev_neighbor_mask]
        pointer_targets = prev_neighbor_indices[prev_neighbor_mask]
        losses["prev_neighbor_index"] = F.cross_entropy(pointer_logits, pointer_targets)

        edge_logits = outputs["edge_token_logits"][:, :, :active_slots, :][prev_neighbor_mask]
        edge_targets = prev_neighbor_tokens[prev_neighbor_mask]
        losses["prev_neighbor_token"] = F.cross_entropy(edge_logits, edge_targets)
    else:
        zero = outputs["prev_count_logits"].sum() * 0.0
        losses["prev_neighbor_index"] = zero
        losses["prev_neighbor_token"] = zero

    losses["total"] = (
        losses["face_exists"]
        + losses["node_features"]
        + 0.5 * losses["prev_count"]
        + losses["prev_neighbor_index"]
        + 0.5 * losses["prev_neighbor_token"]
    )
    return losses
