"""Classic fourth-order Runge-Kutta fixed-step ODE solver."""

from __future__ import annotations


import torch


class RK4:
    """Fourth-order Runge-Kutta solver with augmented-state log-density pass."""

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
        atol: float = 1e-6,  # unused
        rtol: float = 1e-5,  # unused
        steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = f(x, t)`` from ``t0`` to ``t1`` with RK4."""
        _ = atol, rtol
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        t = t0

        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
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
        atol: float = 1e-6,  # unused
        rtol: float = 1e-5,  # unused
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint RK4 integration of state and log-density."""
        _ = atol, rtol
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
            k2, l2 = f_aug(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3, l3 = f_aug(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4, l4 = f_aug(x + dt * k3, t + dt)

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            logp = logp + (dt / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
            t = t + dt

        return x, logp
