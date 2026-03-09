from typing import Dict, Optional

import torch
import torch.nn as nn


def masked_cross_entropy(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits is None or labels is None or logits.numel() == 0:
        device = logits.device if logits is not None else (
            labels.device if labels is not None else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)
    if labels.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    valid_mask = labels != ignore_index
    if int(valid_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=logits.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fn(logits, labels)


def masked_soft_cross_entropy(
    logits: Optional[torch.Tensor],
    soft_targets: Optional[torch.Tensor],
) -> torch.Tensor:
    if logits is None or soft_targets is None or logits.numel() == 0:
        device = logits.device if logits is not None else (
            soft_targets.device if soft_targets is not None else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)
    if soft_targets.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    # Rows with negative values are unsupervised (masked out).
    valid_mask = (soft_targets >= 0.0).all(dim=-1)
    if int(valid_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=logits.device)

    valid_logits = logits[valid_mask]
    valid_targets = soft_targets[valid_mask]
    row_sums = valid_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    valid_targets = valid_targets / row_sums

    log_probs = torch.log_softmax(valid_logits, dim=-1)
    loss = -(valid_targets * log_probs).sum(dim=-1).mean()
    return loss


def combine_multitask_losses(
    proposal_loss: torch.Tensor,
    type_loss: torch.Tensor,
    sensitivity_loss: torch.Tensor,
    weights: Dict[str, float],
) -> torch.Tensor:
    return (
        weights.get("proposal", 1.0) * proposal_loss
        + weights.get("type", 1.0) * type_loss
        + weights.get("sensitivity", 1.0) * sensitivity_loss
    )
