"""EDM noise schedule (Karras et al., 2022).

Variance-exploding schedule with rho-warped sigma grid. ``\\alpha(t) = 1``,
``\\sigma(t) = (\\sigma_{\\min}^{1/\\rho} + t (\\sigma_{\\max}^{1/\\rho} - \\sigma_{\\min}^{1/\\rho}))^\\rho``.

References
----------
- Karras et al., *Elucidating the Design Space of Diffusion-Based
  Generative Models*, 2022 (arXiv:2206.00364).
"""

from __future__ import annotations



import torch

from nami.schedules.base import NoiseSchedule


class EDMSchedule(NoiseSchedule):
    """EDM rho-warped variance-exploding schedule.

    Parameters
    ----------
    sigma_min, sigma_max : float
        Endpoint noise levels.
    rho : float
        Warping exponent (Karras default 7.0).
    """

    def __init__(
        self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        if self.sigma_min <= 0 or self.sigma_max <= 0:
            msg = "sigma_min and sigma_max must be positive"
            raise ValueError(msg)
        if self.sigma_max <= self.sigma_min:
            msg = "sigma_max must be > sigma_min"
            raise ValueError(msg)
        if self.rho <= 0:
            msg = "rho must be positive"
            raise ValueError(msg)

    def _sigma_bounds(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(t)
        a = t.new_tensor(self.sigma_min ** (1.0 / self.rho))
        b = t.new_tensor(self.sigma_max ** (1.0 / self.rho))
        return a, b

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a, b = self._sigma_bounds(t)
        return (a + t * (b - a)) ** self.rho

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        a, b = self._sigma_bounds(t)
        base = a + t * (b - a)
        sigma = base**self.rho
        sigma_prime = self.rho * (base ** (self.rho - 1.0)) * (b - a)
        return torch.sqrt(torch.clamp(2.0 * sigma * sigma_prime, min=0.0))
