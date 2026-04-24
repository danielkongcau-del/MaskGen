from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def compute_boundary_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    pos_weight: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    logits = outputs["boundary_logits"]
    targets = batch["boundary_mask"]

    losses: Dict[str, torch.Tensor] = {}
    losses["bce"] = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    losses["dice"] = dice_loss_from_logits(logits, targets)
    losses["total"] = losses["bce"] + losses["dice"]
    return losses
