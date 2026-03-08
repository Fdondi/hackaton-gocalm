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
