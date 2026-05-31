r"""Denoiser field for the masking-CTMC generator.

Takes an integer token state (data tokens plus the absorbing ``MASK``) and
token) and emits per-coordinate logits over the ``num_states`` data tokens.
This is the categorical denoiser posterior ``p_\theta(z \mid x_t)`` that the masking
CTMC unmasks with. Token coordinates are one-hot encoded over the full
vocabulary (including ``MASK``) so the network can tell revealed positions from
masked ones.
"""

from __future__ import annotations

import torch

from nami.components import MLPBackbone, ScalarTimeEmbedding
from nami.fields._common import validate_context
from nami.fields.base import VectorField
from nami.generators.ctmc import CTMCGeneratorOperator


class CTMCField(VectorField):
    """MLP denoiser emitting per-coordinate categorical logits.

    Args:
        operator (CTMCGeneratorOperator): Operator defining the vocabulary and
            the ``(d, num_states)`` parameter layout.
        condition_dim (int): Context feature dimension (0 for unconditional).
        hidden (int): Backbone width.
        layers (int): Backbone depth.
        activation (str): Backbone activation name.
        dropout (float): Backbone dropout probability.
        layer_norm (bool): Whether the backbone applies layer norm.
    """

    def __init__(
        self,
        operator: CTMCGeneratorOperator,
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
        if len(operator.event_shape) != 1:
            msg = (
                "CTMCField supports a single token axis; "
                f"got event_shape={operator.event_shape}"
            )
            raise ValueError(msg)

        self.operator = operator
        self.num_tokens = operator.event_shape[0]
        self.num_states = operator.num_states
        self.vocab_size = operator.vocab_size
        self.condition_dim = int(condition_dim)
        self.parameter_shape = tuple(operator.parameter_shape)

        self.time_embedding = ScalarTimeEmbedding()
        self.backbone = MLPBackbone(
            self.num_tokens * self.vocab_size + 1 + self.condition_dim,
            self.num_tokens * self.num_states,
            hidden=hidden,
            layers=layers,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.num_tokens:
            msg = (
                f"expected {self.num_tokens} token coordinates, got {x.shape[-1]}"
            )
            raise ValueError(msg)
        lead_shape = tuple(x.shape[:-1])
        one_hot = torch.nn.functional.one_hot(
            x.long(), num_classes=self.vocab_size
        ).to(self.backbone_dtype)
        x_flat = one_hot.reshape(*lead_shape, self.num_tokens * self.vocab_size)
        t_features = self.time_embedding(
            t,
            leading_shape=lead_shape,
            device=x.device,
            dtype=x_flat.dtype,
        )
        validate_context(c, self.condition_dim, lead_shape)
        inputs = torch.cat([x_flat, t_features], dim=-1)
        if c is not None:
            inputs = torch.cat([inputs, c], dim=-1)
        outputs = self.backbone(inputs)
        return outputs.reshape(*lead_shape, *self.parameter_shape)

    @property
    def backbone_dtype(self) -> torch.dtype:
        for p in self.backbone.parameters():
            return p.dtype
        return torch.get_default_dtype()
