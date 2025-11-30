# src/models/audio_encoder.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvAudioEncoder(nn.Module):
    """
    Simple 1D convolutional encoder over proxy audio features.

    Expected input:
        audio: (B, T, F_in)
    Output:
        z:     (B, d_model)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        d_model: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.conv_stack = None  # built lazily if input_dim is None

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(self.hidden_dim, self.d_model)
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(self.dropout)

    def _build_stack(self, in_dim: int) -> None:
        layers = []
        ch_in = in_dim
        ch_out = self.hidden_dim

        for i in range(self.num_layers):
            conv = nn.Conv1d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )
            bn = nn.BatchNorm1d(ch_out)
            layers.extend([conv, bn, nn.GELU(), nn.Dropout(self.dropout)])
            ch_in = ch_out

        self.conv_stack = nn.Sequential(*layers)
        # Kaiming init for convs
        for m in self.conv_stack.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, T, F_in)

        Returns:
            z: (B, d_model)
        """
        if audio.dim() != 3:
            raise ValueError(f"Expected audio to be 3D (B, T, F), got {audio.shape}")

        B, T, F = audio.shape

        if self.conv_stack is None:
            in_dim = self.input_dim if self.input_dim is not None else F
            self._build_stack(in_dim)

        # (B, T, F) → (B, F, T)
        x = audio.transpose(1, 2)

        x = self.conv_stack(x)  # (B, hidden_dim, T)

        # Global average pooling over time
        x = self.global_pool(x)  # (B, hidden_dim, 1)
        x = x.squeeze(-1)        # (B, hidden_dim)

        x = self.proj(x)         # (B, d_model)
        x = self.act(x)
        x = self.dropout_layer(x)

        return x
