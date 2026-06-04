from __future__ import annotations

import torch
from torch import nn

from nami.components import MLPBackbone, ScalarTimeEmbedding
from nami.core.specs import TensorSpec, flatten_event, validate_shapes
from nami.fields._common import normalise_event_shape, validate_context


class LogDensityHead(nn.Module):
    """Scalar head that predicts :math:`\\log p_t(x_t)`.

    Follows the same ``(x, t, c)`` calling convention as
    :class:`~nami.VelocityField` but outputs a single scalar per sample
    rather than a vector.  Intended for use with
    :func:`~nami.log_density_consistency_loss`.

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

        self.spec = TensorSpec(normalise_event_shape(dim))
        self.condition_dim = int(condition_dim)
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
    def event_shape(self) -> tuple[int, ...]:
        return self.spec.event_shape

    @property
    def event_ndim(self) -> int:
        return self.spec.event_ndim

    @property
    def flat_dim(self) -> int:
        return self.spec.numel

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log p_t(x).

        Returns
        -------
        Tensor, shape ``(*lead,)``
            Scalar log-density prediction per sample.
        """
        validate_shapes(x, self.spec)
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
