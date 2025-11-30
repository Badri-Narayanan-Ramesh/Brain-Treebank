# src/training/losses.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for two sets of embeddings u, v.

    u, v: (B, d)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if u.shape != v.shape:
            raise ValueError(f"InfoNCE inputs have different shapes: {u.shape} vs {v.shape}")

        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)

        logits = (u @ v.t()) / self.temperature  # (B, B)
        labels = torch.arange(u.size(0), device=u.device, dtype=torch.long)

        loss_u_to_v = F.cross_entropy(logits, labels)
        loss_v_to_u = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_u_to_v + loss_v_to_u)


def temporal_smoothing_loss(
    z_brain: torch.Tensor,
    meta: Optional[Dict] = None,
    lambda_smooth: float = 0.0,
) -> torch.Tensor:
    """
    Simple temporal smoothing loss on brain embeddings.

    z_brain: (B, d_model)
    meta:    optional dict containing e.g. t_center, subject, trial.
             For now we ignore meta and just smooth over batch order.
    """
    if lambda_smooth <= 0.0:
        return torch.tensor(0.0, device=z_brain.device)

    if z_brain.size(0) < 2:
        return torch.tensor(0.0, device=z_brain.device)

    diffs = z_brain[1:] - z_brain[:-1]  # (B-1, d)
    loss = (diffs ** 2).sum(dim=-1).mean()
    return lambda_smooth * loss
