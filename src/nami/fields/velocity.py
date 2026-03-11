from __future__ import annotations

import torch

from ..components import MLPBackbone, ScalarTimeEmbedding
from ..core.specs import (
    event_numel,
    flatten_event,
    unflatten_event,
    validate_shapes,
)
from ._common import normalise_event_shape, validate_context
from .base import VectorField


class VelocityField(VectorField):
    """MLP velocity field for flow matching.

    Supports unconditional and conditional workflows. When ``condition_dim``
    is non-zero the field expects a context vector ``c`` concatenated to the
    input; otherwise ``c`` should be ``None``.  Conditioning is handled by
    the process layer via lazy binding. This field simply receives whatever
    context the process passes through.

    Args:
        dim: Data dimensionality or event shape.
        condition_dim: Conditioning vector dimensionality (0 for unconditional).
        hidden: Hidden layer width.
        layers: Number of hidden layers.
        activation: Activation function ('silu', 'relu', 'gelu', 'tanh').
        dropout: Dropout probability (0 disables).
        layer_norm: Whether to apply layer normalisation in hidden layers.
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        condition_dim: int = 0,
        hidden: int = 256,
        layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if condition_dim < 0:
            msg = f"condition_dim must be non-negative, got {condition_dim}"
            raise ValueError(msg)

        self.event_shape = normalise_event_shape(dim)
        self.condition_dim = int(condition_dim)
        self.flat_dim = event_numel(self.event_shape)
        self.time_embedding = ScalarTimeEmbedding()
        self.backbone = MLPBackbone(
            self.flat_dim + 1 + self.condition_dim,
            self.flat_dim,
            hidden=hidden,
            layers=layers,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_shapes(x, self.event_ndim, expected_event_shape=self.event_shape)
        x_flat = flatten_event(x, self.event_ndim)
        lead_shape = tuple(x_flat.shape[:-1])
        t_features = self.time_embedding(
            t,
            leading_shape=lead_shape,
            device=x.device,
            dtype=x.dtype,
        )
        validate_context(c, self.condition_dim, lead_shape)
        inputs = torch.cat([x_flat, t_features], dim=-1)
        if c is not None:
            inputs = torch.cat([inputs, c], dim=-1)
        return unflatten_event(self.backbone(inputs), self.event_shape)
