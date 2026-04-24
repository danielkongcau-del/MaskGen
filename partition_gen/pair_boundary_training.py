from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _dice_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _pair_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid_mask], targets[valid_mask], pos_weight=pos_weight)


def _pair_dice(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits[valid_mask])
    return _dice_from_probs(probs, targets[valid_mask])


def _union_probs(pair_logits: torch.Tensor, pair_valid: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(pair_logits).clamp(min=1e-6, max=1.0 - 1e-6)
    valid = pair_valid[:, :, None, None, None].to(probs.dtype)
    log_background = torch.log1p(-probs) * valid
    return 1.0 - torch.exp(log_background.sum(dim=1))


def _pair_count_maps(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    pair_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(pair_logits)
    valid = pair_valid[:, :, None, None, None].to(probs.dtype)
    pred_counts = (probs * valid).sum(dim=1)
    target_counts = (pair_targets * valid).sum(dim=1)
    return pred_counts, target_counts


def compute_pair_boundary_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    pos_weight: torch.Tensor | None = None,
    count_loss_weight: float = 0.25,
    overlap_loss_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    pair_logits = outputs["pair_logits"]
    pair_targets = batch["pair_masks"]
    pair_valid = batch["pair_valid"]

    valid_mask = pair_valid[:, :, None, None, None].expand_as(pair_logits)
    losses: Dict[str, torch.Tensor] = {}
    losses["pair_bce"] = _pair_bce(pair_logits, pair_targets, valid_mask, pos_weight=pos_weight)
    losses["pair_dice"] = _pair_dice(pair_logits, pair_targets, valid_mask)

    union_probs = _union_probs(pair_logits, pair_valid)
    union_targets = batch["union_mask"]
    with torch.autocast(device_type=union_probs.device.type, enabled=False):
        losses["union_bce"] = F.binary_cross_entropy(
            union_probs.float().clamp(min=1e-6, max=1.0 - 1e-6),
            union_targets.float(),
        )
    losses["union_dice"] = _dice_from_probs(union_probs, union_targets)

    pred_counts, target_counts = _pair_count_maps(pair_logits, pair_targets, pair_valid)
    boundary_weight = 1.0 + union_targets
    losses["count_l1"] = ((pred_counts - target_counts).abs() * boundary_weight).mean()
    overlap = torch.relu(pred_counts - target_counts)
    losses["overlap_l1"] = (overlap * boundary_weight).mean()

    losses["total"] = (
        losses["pair_bce"]
        + losses["pair_dice"]
        + 0.5 * (losses["union_bce"] + losses["union_dice"])
        + float(count_loss_weight) * losses["count_l1"]
        + float(overlap_loss_weight) * losses["overlap_l1"]
    )
    return losses
