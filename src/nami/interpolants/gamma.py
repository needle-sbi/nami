from __future__ import annotations

from dataclasses import dataclass

import torch

# Based on https://github.com/malbergo/stochastic-interpolants/tree/main [https://arxiv.org/abs/2303.08797 Albergo et al.]


class GammaSchedule:
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return self.gamma(t) * self.gamma_dot(t)


class ZeroGamma(GammaSchedule):
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


@dataclass(frozen=True)
class BrownianGamma(GammaSchedule):
    eps: float = 1e-12

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(t * (1 - t), min=self.eps))

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        numerator = 1 - 2 * t
        denominator = 2 * torch.clamp(self.gamma(t), min=self.eps)
        return numerator / denominator

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - 2 * t)


@dataclass(frozen=True)
class ScaledBrownianGamma(GammaSchedule):
    """Brownian gamma schedule with variance scale.

    The schedule is
    ``gamma(t)^2 = scale * t * (1 - t)``.
    For bridge paths parameterised by ``sigma`` where
    ``gamma(t) = sigma * sqrt(t * (1 - t))``,
    use ``scale = sigma**2`` (or ``from_sigma``).
    """

    scale: float = 1.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.scale <= 0:
            msg = "scale must be positive"
            raise ValueError(msg)

    @classmethod
    def from_sigma(cls, sigma: float, eps: float = 1e-12) -> ScaledBrownianGamma:
        """Construct a schedule with ``gamma(t) = sigma * sqrt(t * (1 - t))``."""
        if sigma <= 0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        return cls(scale=float(sigma) ** 2, eps=eps)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.scale * t * (1 - t), min=self.eps))

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        numerator = self.scale * (1 - 2 * t)
        denominator = 2 * torch.clamp(self.gamma(t), min=self.eps)
        return numerator / denominator

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.scale * (1 - 2 * t)
