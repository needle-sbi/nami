"""MLP velocity field with adaptive-LayerNorm (adaLN-zero) conditioning.

A drop-in alternative to the input-concat MLP :class:`VelocityField` that
re-injects time and context at every residual block via adaptive LayerNorm
modulation. This is the conventional remedy to the failure mode where
shallow concatenation of ``t`` and ``c`` at the input layer is washed out
by network depth, leaving the field either time-dominant or context-flat.

References
----------
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT),
  2023 (arXiv:2212.09748) — adaLN-zero modulation and zero-initialisation.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*,
  2017 (arXiv:1709.07871) — the closely related FiLM modulation.
"""

from __future__ import annotations

import torch
from torch import nn

from nami.components import SinusoidalTimeEmbedding
from nami.core.specs import (
    event_numel,
    flatten_event,
    unflatten_event,
    validate_shapes,
)
from nami.fields._common import normalise_event_shape, validate_context
from nami.fields.base import VectorField

_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class _AdaLNBlock(nn.Module):
    """Residual MLP block whose (shift, scale, gate) modulation is supplied externally.

    The modulation triplet is produced by a single shared conditioning MLP for
    the whole stack rather than per-block, so this module is purely a
    residual computation and owns no conditioning parameters.
    """

    def __init__(self, hidden: int, mlp_ratio: float, activation: str, dropout: float):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        inner = int(hidden * mlp_ratio)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, inner),
            act_cls(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner, hidden),
        )

    def forward(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm(x) * (1.0 + scale) + shift
        return x + gate * self.mlp(h)


class AdaLNVelocityField(VectorField):
    """MLP velocity field with adaLN-zero conditioning on ``t`` and optional ``c``.

    Time and context are embedded once and concatenated into a single
    conditioning vector that drives an adaLN modulation at every residual
    block, so both signals are re-injected at every depth rather than only
    at the input. The shared modulation MLP is zero-initialised (adaLN-zero),
    so at initialisation every (shift, scale, gate) is zero and the network
    behaves as a residual identity in the data stream; it learns to bend the
    flow during training.

    The output projection is also zero-initialised, so the field outputs
    zero at init — a quiet starting point for the velocity head in flow
    matching and diffusion training.

    Args:
        dim: Data dimensionality or event shape.
        condition_dim: Conditioning vector dimensionality (0 for unconditional).
        hidden: Residual stream width.
        layers: Number of residual blocks.
        time_dim: Sinusoidal time embedding dimensionality.
        cond_hidden: Hidden width of the conditioning MLP that produces
            per-block (shift, scale, gate) triplets.
        mlp_ratio: Inner-to-hidden width ratio inside each block's MLP.
        activation: Activation inside each block ('silu', 'relu', 'gelu', 'tanh').
        dropout: Dropout probability inside each block (0 disables).
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        condition_dim: int = 0,
        hidden: int = 256,
        layers: int = 4,
        time_dim: int = 128,
        cond_hidden: int = 256,
        mlp_ratio: float = 4.0,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if condition_dim < 0:
            msg = f"condition_dim must be non-negative, got {condition_dim}"
            raise ValueError(msg)
        if hidden <= 0:
            msg = f"hidden must be positive, got {hidden}"
            raise ValueError(msg)
        if layers <= 0:
            msg = f"layers must be positive, got {layers}"
            raise ValueError(msg)
        if time_dim <= 0:
            msg = f"time_dim must be positive, got {time_dim}"
            raise ValueError(msg)
        if cond_hidden <= 0:
            msg = f"cond_hidden must be positive, got {cond_hidden}"
            raise ValueError(msg)
        if mlp_ratio <= 0:
            msg = f"mlp_ratio must be positive, got {mlp_ratio}"
            raise ValueError(msg)
        if activation not in _ACTIVATIONS:
            msg = (
                f"unknown activation: {activation}; "
                f"expected one of {sorted(_ACTIVATIONS)}"
            )
            raise ValueError(msg)

        self.event_shape = normalise_event_shape(dim)
        self.condition_dim = int(condition_dim)
        self.flat_dim = event_numel(self.event_shape)
        self.layers = int(layers)
        self.hidden = int(hidden)

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        time_features = self.time_embedding.out_dim

        self.input_proj = nn.Linear(self.flat_dim, hidden)
        self.output_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.output_proj = nn.Linear(hidden, self.flat_dim)

        cond_input_dim = time_features + self.condition_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, cond_hidden),
            _ACTIVATIONS[activation](),
            nn.Linear(cond_hidden, 3 * hidden * self.layers),
        )
        cond_out = self.cond_mlp[-1]
        if not isinstance(cond_out, nn.Linear):
            msg = "final conditioning layer must be nn.Linear"
            raise TypeError(msg)
        nn.init.zeros_(cond_out.weight)
        if cond_out.bias is not None:
            nn.init.zeros_(cond_out.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.blocks = nn.ModuleList(
            [
                _AdaLNBlock(hidden, mlp_ratio, activation, dropout)
                for _ in range(self.layers)
            ]
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
        validate_context(c, self.condition_dim, lead_shape)

        t_emb = self.time_embedding(
            t,
            leading_shape=lead_shape,
            device=x.device,
            dtype=x.dtype,
        )

        cond = torch.cat([t_emb, c], dim=-1) if c is not None else t_emb

        modulation = self.cond_mlp(cond)
        modulation = modulation.view(*lead_shape, self.layers, 3, self.hidden)

        h = self.input_proj(x_flat)
        for i, block in enumerate(self.blocks):
            shift = modulation[..., i, 0, :]
            scale = modulation[..., i, 1, :]
            gate = modulation[..., i, 2, :]
            h = block(h, shift, scale, gate)

        out = self.output_proj(self.output_norm(h))
        return unflatten_event(out, self.event_shape)
