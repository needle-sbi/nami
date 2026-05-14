"""Heun's method (improved-Euler / RK2) fixed-step ODE solver.

Second-order trapezoidal predictor-corrector — the deterministic
sampler from EDM (Karras et al., 2022) and a standard FM /
probability-flow integrator.

References
----------
- Karras et al., *Elucidating the Design Space of Diffusion-Based
  Generative Models* (EDM), 2022 (arXiv:2206.00364).
"""

from __future__ import annotations


import torch


class Heun:
    """Heun's method (RK2) ODE solver with optional augmented-state pass."""

    requires_steps = True
    supports_rsample = True
    is_sde = False

    def __init__(self, steps: int = 32):
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        self.steps = int(steps)

    def integrate(
        self,
        f,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = f(x, t)`` from ``t0`` to ``t1`` with ``steps`` RK2 steps."""
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        t = t0

        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + dt * k1, t + dt)
            x = x + (dt / 2.0) * (k1 + k2)
            t = t + dt

        return x

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint RK2 integration of state and log-density."""
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        logp = logp0
        t = t0

        for _ in range(steps):
            k1, l1 = f_aug(x, t)
            k2, l2 = f_aug(x + dt * k1, t + dt)

            x = x + (dt / 2.0) * (k1 + k2)
            logp = logp + (dt / 2.0) * (l1 + l2)
            t = t + dt

        return x, logp
