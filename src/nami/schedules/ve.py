"""Variance-Exploding (VE) noise schedule.

``\\alpha(t) = 1``, ``\\sigma(t) = \\sigma_{\\min} (\\sigma_{\\max}/\\sigma_{\\min})^t``
— the geometric VE SDE from Song et al.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456).
"""

from __future__ import annotations



import math

import torch

from nami.schedules.base import NoiseSchedule


class VESchedule(NoiseSchedule):
    """Geometric variance-exploding schedule of Song et al. (2020).

    Parameters
    ----------
    sigma_min, sigma_max : float
        Endpoint noise levels (defaults match score-SDE conventions).
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if self.sigma_min <= 0 or self.sigma_max <= 0:
            msg = "sigma_min and sigma_max must be positive"
            raise ValueError(msg)
        if self.sigma_max <= self.sigma_min:
            msg = "sigma_max must be > sigma_min"
            raise ValueError(msg)
        self._log_r = math.log(self.sigma_max / self.sigma_min)
        self._diffusion_const = math.sqrt(2.0 * self._log_r)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) * self._diffusion_const
