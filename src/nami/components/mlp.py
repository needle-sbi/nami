"""Generic MLP backbone operating on the last tensor dimension."""

from __future__ import annotations

import torch
from torch import nn

from nami.components.activation import get_activation


def _validate_mlp_config(hidden: int, layers: int, dropout: float) -> None:
    """Validate a simple MLP backbone configuration."""
    if hidden <= 0:
        msg = f"hidden must be positive, got {hidden}"
        raise ValueError(msg)
    if layers < 0:
        msg = f"layers must be non-negative, got {layers}"
        raise ValueError(msg)
    if not 0.0 <= dropout < 1.0:
        msg = f"dropout must be in [0, 1), got {dropout}"
        raise ValueError(msg)


def _build_mlp(
    in_dim: int,
    out_dim: int,
    *,
    hidden: int,
    layers: int,
    activation: str,
    dropout: float,
    layer_norm: bool,
) -> nn.Sequential:
    """Build an MLP that acts on the final tensor dimension."""
    _validate_mlp_config(hidden, layers, dropout)

    modules: list[nn.Module] = []
    prev = in_dim
    for _ in range(layers):
        modules.append(nn.Linear(prev, hidden))
        if layer_norm:
            modules.append(nn.LayerNorm(hidden))
        modules.append(get_activation(activation))
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = hidden
    modules.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*modules)


class MLPBackbone(nn.Module):
    """Generic MLP backbone operating on the last tensor dimension."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: int = 256,
        layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = _build_mlp(
            self.in_dim,
            self.out_dim,
            hidden=hidden,
            layers=layers,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP along the last dimension of ``x``."""
        return self.net(x)
