r"""Abstract noise-schedule contract for Gaussian conditional paths.

Schedules define ``\alpha(t)``, ``\sigma(t)``, and the forward SDE
coefficients ``f(x,t)`` and ``g(t)``.  Diffusion processes use

.. math::

   x_t = \alpha(t)x_0 + \sigma(t)\epsilon,

while Gaussian interpolants may use the same scalar functions with
``x_noise`` and ``x_data`` as the two operands.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456).
"""

from __future__ import annotations

import torch


class NoiseSchedule:
    r"""Base interface for scalar Gaussian schedules."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluate ``\alpha(t)``.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Signal scale for diffusion-style paths.
        """
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluate ``\sigma(t)``.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Noise scale for diffusion-style paths.
        """
        raise NotImplementedError

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the signal-to-noise ratio.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: ``\alpha^2(t) / \sigma^2(t)``.
        """
        a = self.alpha(t)
        s = self.sigma(t)
        return (a * a) / (s * s)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the forward-SDE drift.

        Args:
            x (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Drift ``f(x,t)`` with the same shape as ``x``.
        """
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the forward-SDE diffusion coefficient.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Diffusion coefficient ``g(t)``.
        """
        raise NotImplementedError
