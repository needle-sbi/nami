from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import ProbabilityPath


@dataclass(frozen=True)
class BrownianBridgePath(ProbabilityPath):
    """Brownian bridge probability path for Schrodinger bridge matching.

    Interpolates between ``x_target`` (at ``t=0``) and ``x_source`` (at ``t=1``)
    along a Brownian bridge with diffusion coefficient ``sigma``.

    When reconstructing probability-flow or markovization drift from flow+score
    models, use a gamma schedule consistent with this path variance and epsilon.
    ``gamma_schedule()`` provides this mapping.
    """

    sigma: float = 1.0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        if self.eps <= 0:
            msg = "eps must be positive"
            raise ValueError(msg)
        if self.eps >= 0.5:
            msg = "eps must be < 0.5"
            raise ValueError(msg)

    def gamma_schedule(self):
        """Return a gamma schedule consistent with this path's ``sigma`` and ``eps``."""
        from ..interpolants.gamma import ScaledBrownianGamma

        return ScaledBrownianGamma.from_sigma(self.sigma, eps=self.eps)

    def sample_xt(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t) * x_target + t * x_source
        if z is None:
            z = torch.randn_like(mu)
        std = self.sigma * torch.sqrt(t * (1.0 - t))
        return mu + std * z

    def target_ut(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        xt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xt is None:
            return x_source - x_target
        t = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t) * x_target + t * x_source
        denom = 2.0 * torch.clamp(t * (1.0 - t), min=self.eps)
        coeff = (1.0 - 2.0 * t) / denom
        return (x_source - x_target) + coeff * (xt - mu)

    def score_target(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        t = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t) * x_target + t * x_source
        var = self.sigma**2 * torch.clamp(t * (1.0 - t), min=self.eps)
        return (mu - xt) / var
