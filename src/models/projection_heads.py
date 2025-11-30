# src/models/projection_heads.py

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Small MLP projection head used before contrastive loss.

    Maps input_dim → d_projection with a hidden layer and LayerNorm.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
