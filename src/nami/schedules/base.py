"""Abstract noise-schedule contract for Gaussian conditional paths.

Defines the four scalar fields a schedule must expose for the forward
process ``x_t = \\alpha(t) x_0 + \\sigma(t) \\epsilon`` and the associated
reverse-time SDE ``dx = [f(x,t) - g(t)^2 \\nabla \\log p_t(x)] dt + g(t) d\\bar W``.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456).
"""

from __future__ import annotations



import torch


class NoiseSchedule:
    """Base interface for ``(\\alpha(t), \\sigma(t))`` schedules and their SDE form."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Signal scale ``\\alpha(t)`` in ``x_t = \\alpha(t) x_0 + \\sigma(t) \\epsilon``."""
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise scale ``\\sigma(t)``."""
        raise NotImplementedError

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio ``\\alpha^2(t) / \\sigma^2(t)``."""
        a = self.alpha(t)
        s = self.sigma(t)
        return (a * a) / (s * s)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward-SDE drift ``f(x, t)``."""
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Forward-SDE diffusion coefficient ``g(t)``."""
        raise NotImplementedError
