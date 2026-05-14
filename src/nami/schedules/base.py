"""Abstract noise-schedule contract for Gaussian conditional paths.

Defines the four scalar fields a schedule must expose. The functions
``\\alpha(t)`` and ``\\sigma(t)`` are defined by their numerical contracts:
``\\alpha(0)=1, \\alpha(1)=0`` and ``\\sigma(0)=0, \\sigma(1)=1`` (VP-style;
VE-style modifies ``\\sigma``). Two consumers reinterpret which operand
each multiplies:

* :class:`~nami.processes.diffusion.Diffusion` (diffusion convention,
  retained for the score-based reverse-time PF-ODE) reads
  ``x_t = \\alpha(t) x_0 + \\sigma(t) \\epsilon`` — ``\\alpha`` is the signal
  scale, ``\\sigma`` is the noise scale.  ``drift(x, t)`` and
  ``diffusion(t)`` describe the forward SDE
  ``dx = [f(x,t) - g(t)^2 \\nabla \\log p_t(x)] dt + g(t) d\\bar W`` in this
  convention.
* :class:`~nami.interpolants.gaussian.GaussianInterpolant` (FM
  convention, ``t=0`` noise → ``t=1`` data) reads
  ``x_t = \\alpha(t) \\epsilon + \\sigma(t) x_0`` — ``\\alpha`` is the noise
  coefficient and ``\\sigma`` is the data coefficient (operand swap, same
  numerical functions).

The legacy "signal scale" / "noise scale" naming below reflects the
diffusion-convention interpretation; FM-convention callers should
mentally swap the labels.

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
