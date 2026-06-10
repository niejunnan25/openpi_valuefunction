"""Action-conditioned distributional gain critic."""

from __future__ import annotations

import torch
from torch import nn


class ActionGainCritic(nn.Module):
    """Predict p(DeltaV^K) from PI0 action suffix hidden states."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_gain_bins: int = 101,
        horizon: int = 5,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_gain_bins <= 1:
            raise ValueError(f"num_gain_bins must be > 1, got {num_gain_bins}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gain_bins = num_gain_bins
        self.horizon = horizon
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gain_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_gain_bins),
        )

        nn.init.trunc_normal_(self.gain_query, std=0.02)

    def forward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        """Return gain logits with shape [B, num_gain_bins]."""
        if action_hidden.ndim != 3:
            raise ValueError(f"action_hidden must have shape [B, H, D], got {tuple(action_hidden.shape)}")
        if action_hidden.shape[-1] != self.input_dim:
            raise ValueError(f"Expected action_hidden dim {self.input_dim}, got {action_hidden.shape[-1]}")
        if action_hidden.shape[1] < self.horizon:
            raise ValueError(f"Need at least {self.horizon} action tokens, got {action_hidden.shape[1]}")

        x = action_hidden[:, : self.horizon]
        x = self.input_proj(self.input_norm(x))

        query = self.gain_query.expand(x.shape[0], -1, -1)
        x = torch.cat([query, x], dim=1)
        x = self.encoder(x)
        return self.output_head(x[:, 0])
