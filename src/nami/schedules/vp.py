"""Variance-Preserving (VP) noise schedule.

Linear ``\\beta(t) = \\beta_{\\min} + t (\\beta_{\\max} - \\beta_{\\min})`` from
the score-SDE / DDPM family: ``\\alpha(t)^2 + \\sigma(t)^2 = 1``.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456).
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020
  (arXiv:2006.11239) — original DDPM noise convention.
"""

from __future__ import annotations



import torch

from nami.schedules.base import NoiseSchedule


class VPSchedule(NoiseSchedule):
    """Linear-beta variance-preserving schedule (DDPM / VP-SDE).

    Parameters
    ----------
    beta_min, beta_max : float
        Endpoints of the linear ``\\beta(t)`` ramp.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if beta_min <= 0 or beta_max <= 0:
            msg = "beta_min and beta_max must be positive"
            raise ValueError(msg)
        if beta_max <= beta_min:
            msg = "beta_max must be > beta_min"
            raise ValueError(msg)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _beta_int(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t**2)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self._beta_int(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(-torch.expm1(-self._beta_int(t)))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))
