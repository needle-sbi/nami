"""Gamma schedules for stochastic-interpolant noise scaling.

Encodes the ``\\gamma(t)`` profile that multiplies the Gaussian latent
in a stochastic interpolant ``x_t = I(t, x_0, x_1) + \\gamma(t) z``.
Implementations expose ``gamma``, ``gamma_dot``, and the common
product ``gamma * gamma_dot`` used in score / drift identities.

References
----------
- Albergo, Boffi, Vanden-Eijnden, *Stochastic Interpolants: A Unifying
  Framework*, 2023 (arXiv:2303.08797).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class GammaSchedule:
    """Abstract noise-scaling schedule ``\\gamma(t)`` for stochastic interpolants."""

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise scale ``\\gamma(t)``."""
        raise NotImplementedError  # pragma: no cover - abstract stub

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative ``\\dot{\\gamma}(t)``."""
        raise NotImplementedError  # pragma: no cover - abstract stub

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Convenience product ``\\gamma(t) \\dot{\\gamma}(t)``.

        Subclasses with closed-form simplifications (e.g. Brownian
        ``\\gamma^2 = t(1-t)`` gives ``\\gamma \\dot{\\gamma} = (1-2t)/2``) override this
        to avoid the singularity at endpoints.
        """
        return self.gamma(t) * self.gamma_dot(t)


class ZeroGamma(GammaSchedule):
    """Degenerate schedule ``\\gamma(t) \\equiv 0`` recovers deterministic interpolation."""

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def gamma_gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


@dataclass(frozen=True)
class BrownianGamma(GammaSchedule):
    """Brownian-bridge schedule ``\\gamma(t) = \\sqrt{t(1-t)}``.

    Default choice from Albergo et al. the noise scale vanishes at
    both endpoints so the interpolant matches the source / target
    distributions exactly at ``t=0`` and ``t=1``.
    """

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
