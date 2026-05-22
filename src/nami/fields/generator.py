"""MLP field that emits packed generator-operator parameters.

Backbone is shape-aware: it maps the flattened event tensor plus a
scalar-time feature (and optional context) to a tensor in the
operator's ``parameter_shape``. Consumed by :class:`GeneratorMatching`
via :class:`GeneratorParams` targets.

References
----------
- Holderrieth et al., *Generator Matching*, 2024.
"""

from __future__ import annotations

import torch

from nami.components import MLPBackbone, ScalarTimeEmbedding
from nami.core.specs import event_numel, flatten_event, validate_shapes
from nami.fields._common import normalise_event_shape, validate_context
from nami.fields.base import VectorField


class GeneratorField(VectorField):
    """MLP field that predicts generator parameters."""

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        operator,
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
        if tuple(operator.event_shape) != self.event_shape:
            msg = (
                f"operator.event_shape must match dim: expected {self.event_shape}, "
                f"got {tuple(operator.event_shape)}"
            )
            raise ValueError(msg)

        self.operator = operator
        self.condition_dim = int(condition_dim)
        self.flat_dim = event_numel(self.event_shape)
        self.parameter_shape = tuple(operator.parameter_shape)
        self.parameter_dim = event_numel(self.parameter_shape)
        self.time_embedding = ScalarTimeEmbedding()
        self.backbone = MLPBackbone(
            self.flat_dim + 1 + self.condition_dim,
            self.parameter_dim,
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
        outputs = self.backbone(inputs)
        return outputs.reshape(*lead_shape, *self.parameter_shape)
