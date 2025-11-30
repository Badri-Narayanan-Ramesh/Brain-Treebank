# src/models/brain_transformer.py

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PopulationTransformer(nn.Module):
    """
    PopT-style transformer over electrodes.

    Expected input:
        brain:  (B, N, input_dim)      # N electrodes, input_dim = 768 from BrainBERT
        coords: (B, N, coord_dim) or None

    Output:
        cls_token: (B, d_model)
        seq_embs:  (B, N+1, d_model)   # including CLS at position 0
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        coord_dim: int = 0,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.coord_dim = int(coord_dim)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)
        self.use_cls_token = use_cls_token

        feat_dim = self.input_dim + self.coord_dim

        # Project [features (+ coords)] → d_model
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Optional CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # PopT uses GELU as layer activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # match PopT
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        brain: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            brain:  (B, N, input_dim)
            coords: (B, N, coord_dim) or None

        Returns:
            cls:    (B, d_model)
            out:    (B, N+1, d_model)  # CLS + electrodes
        """
        # brain shape checks
        if brain.dim() != 3:
            raise ValueError(f"Expected brain to be 3D (B, N, F), got {brain.shape}")

        B, N, F = brain.shape
        if F != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {F} (brain.shape={brain.shape})"
            )

        if self.coord_dim > 0:
            if coords is None:
                raise ValueError(
                    f"coord_dim={self.coord_dim} but coords=None was provided."
                )
            if coords.shape[:2] != (B, N):
                raise ValueError(
                    f"coords shape {coords.shape} incompatible with brain {brain.shape}"
                )
            x = torch.cat([brain, coords], dim=-1)  # (B, N, F+coord_dim)
        else:
            x = brain  # (B, N, F)

        # Project to d_model
        x = self.input_proj(x)  # (B, N, d_model)

        # Add CLS token
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(B, 1, self.d_model)  # (B, 1, d_model)
            x = torch.cat([cls_tok, x], dim=1)  # (B, N+1, d_model)

        # No explicit mask for now (assumes fixed N per subject)
        out = self.transformer(x)  # (B, N+1, d_model)
        out = self.norm(out)

        if self.use_cls_token:
            cls = out[:, 0, :]    # (B, d_model)
        else:
            # fall back to mean pool if no CLS
            cls = out.mean(dim=1)

        return cls, out
