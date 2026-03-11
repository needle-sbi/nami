from __future__ import annotations

import torch
from torch import nn

from ..components import SinusoidalTimeEmbedding, TransformerBackbone
from ..core.specs import event_numel, flatten_event, unflatten_event, validate_shapes
from ._common import normalise_event_shape, validate_context
from .base import VectorField


class TransformerVelocityField(VectorField):
    """Transformer velocity field over flattened event tokens.

    Each scalar feature in the flattened event is treated as one token. Time is
    embedded once per sample, then broadcast across all tokens. Optional context
    is projected to a single cross-attention token, so attention cost scales
    quadratically with the flattened event size.
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        model_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        time_dim: int = 32,
        condition_dim: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if model_dim <= 0:
            msg = f"model_dim must be positive, got {model_dim}"
            raise ValueError(msg)
        if time_dim <= 0:
            msg = f"time_dim must be positive, got {time_dim}"
            raise ValueError(msg)
        if condition_dim < 0:
            msg = f"condition_dim must be non-negative, got {condition_dim}"
            raise ValueError(msg)

        self.event_shape = normalise_event_shape(dim)
        self.condition_dim = int(condition_dim)
        self.flat_dim = event_numel(self.event_shape)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_proj = nn.Linear(1 + time_dim, model_dim)
        self.backbone = TransformerBackbone(
            dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            cross_attention=self.condition_dim > 0,
        )
        self.output_proj = nn.Linear(model_dim, 1)
        self.context_proj = (
            nn.Linear(self.condition_dim, model_dim) if self.condition_dim > 0 else None
        )

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    def _context_tokens(
        self,
        c: torch.Tensor | None,
        lead_shape: tuple[int, ...],
    ) -> torch.Tensor | None:
        validate_context(c, self.condition_dim, lead_shape)
        if c is None or self.context_proj is None:
            return None
        return self.context_proj(c).unsqueeze(-2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_shapes(x, self.event_ndim, expected_event_shape=self.event_shape)
        x_flat = flatten_event(x, self.event_ndim)
        lead_shape = tuple(x_flat.shape[:-1])

        time = self.time_embedding(
            t,
            leading_shape=lead_shape,
            device=x.device,
            dtype=x.dtype,
        )
        time_tokens = time.unsqueeze(-2).expand(
            *lead_shape, self.flat_dim, time.shape[-1]
        )
        tokens = self.input_proj(torch.cat([x_flat.unsqueeze(-1), time_tokens], dim=-1))

        out = self.backbone(tokens, context=self._context_tokens(c, lead_shape))
        out = self.output_proj(out).squeeze(-1)
        return unflatten_event(out, self.event_shape)
