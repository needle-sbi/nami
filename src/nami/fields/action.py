from __future__ import annotations

import torch
from torch import nn

from nami.components import MLPBackbone, ScalarTimeEmbedding
from nami.core.specs import event_numel, flatten_event, validate_shapes
from nami.fields._common import normalise_event_shape, validate_context


class ActionHead(nn.Module):
    """Scalar field predicting the action potential :math:`s(x, t)`.

    Follows the same ``(x, t, c)`` calling convention as
    :class:`~nami.VelocityField` but outputs a single scalar per sample.
    The conditional velocity is recovered by autograd:
    :math:`u_t(x) = \\nabla_x s(x, t)`.  Intended for use with
    :func:`~nami.action_matching_loss` and
    :class:`~nami.ActionMatching` — both the training loss and the
    runtime integrator differentiate ``s`` with respect to ``x``.

    Structurally identical to
    :class:`~nami.fields.consistency.LogDensityHead` (also scalar-out),
    but kept as a distinct class so the calling convention names the
    quantity it predicts.  Sharing the class would muddle two
    semantically different roles: ``LogDensityHead`` predicts
    :math:`\\log p_t(x)`; ``ActionHead`` predicts a scalar whose
    *gradient* is the velocity.

    Parameters
    ----------
    dim : int or tuple[int, ...]
        Data dimensionality (event shape).
    condition_dim : int
        Conditioning vector dimensionality (0 for unconditional).
    hidden : int
        Hidden layer width.
    layers : int
        Number of hidden layers.
    activation : str
        Activation function.
    dropout : float
        Dropout probability.
    layer_norm : bool
        Whether to apply layer normalisation.
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        condition_dim: int = 0,
        hidden: int = 128,
        layers: int = 2,
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
            1,  # scalar output
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
        """Predict the scalar action potential ``s(x, t)``.

        Returns
        -------
        Tensor, shape ``(*lead,)``
            Scalar potential per sample.  Its gradient w.r.t. ``x`` is
            the velocity used by :class:`~nami.ActionMatching`.
        """
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
        return self.backbone(inputs).squeeze(-1)
