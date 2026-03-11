from __future__ import annotations

import math

import torch
from torch import nn


def _broadcast_time(
    t: torch.Tensor,
    leading_shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast time values to a target leading shape."""
    t = torch.as_tensor(t, device=device, dtype=dtype)
    return torch.broadcast_to(t, leading_shape)


class ScalarTimeEmbedding(nn.Module):
    """Return scalar time as a one-dimensional feature."""

    out_dim = 1

    def forward(
        self,
        t: torch.Tensor,
        *,
        leading_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _broadcast_time(
            t,
            leading_shape,
            device=device,
            dtype=dtype,
        ).unsqueeze(-1)


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar time to sinusoidal features.

    When ``dim=1`` no sinusoids are produced; the output is simply the
    raw scalar time (equivalent to :class:`ScalarTimeEmbedding`).
    """

    def __init__(self, dim: int, *, max_period: float = 10000.0):
        super().__init__()
        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)
        if max_period <= 0:
            msg = f"max_period must be positive, got {max_period}"
            raise ValueError(msg)
        self.dim = int(dim)
        self.max_period = float(max_period)

    @property
    def out_dim(self) -> int:
        return self.dim

    def forward(
        self,
        t: torch.Tensor,
        *,
        leading_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = _broadcast_time(t, leading_shape, device=device, dtype=dtype)
        half = self.dim // 2
        if half == 0:
            return t.unsqueeze(-1)

        scale = math.log(self.max_period) / max(half, 1)
        freqs = torch.exp(-scale * torch.arange(half, device=device, dtype=dtype))
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, t.unsqueeze(-1)], dim=-1)
        return emb
