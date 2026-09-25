"""Sequence backbones for the learned barge-in policies.

Both backbones map (B, T, D) step embeddings to (B, T, D) hidden states and
are strictly causal: output at step t depends only on steps <= t. The
transformer uses window-relative sinusoidal positions (training crops random
40-step windows, runtime slides a 40-step window, so positions must not be
absolute).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LSTMBackbone(nn.Module):
    """Arm B: 2-layer LSTM over step embeddings (inherently causal)."""

    def __init__(self, d_model: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class CausalTransformerBackbone(nn.Module):
    """Arm C: causal transformer encoder (~6 layers x 256 hidden)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_period_positions: int = 512,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.d_model = d_model
        self.register_buffer("pe", self._sinusoidal(max_period_positions, d_model), persistent=False)

    @staticmethod
    def _sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.size(1)
        if steps > self.pe.size(0):
            raise ValueError(
                f"sequence length {steps} exceeds positional table {self.pe.size(0)}"
            )
        x = x + self.pe[:steps].to(dtype=x.dtype)
        mask = torch.triu(
            torch.full((steps, steps), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        return self.encoder(x, mask=mask, is_causal=True)
