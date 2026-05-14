"""Euler-Maruyama fixed-step SDE solver.

Standard first-order scheme for Ito SDEs ``dx = b(x,t) dt + g(x,t) dW``.
"""

from __future__ import annotations



import inspect
import math

import torch


class EulerMaruyama:
    """Euler-Maruyama SDE integrator with explicit Brownian increments."""

    requires_steps = True
    is_sde = True

    def __init__(self, steps: int = 64):
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        self.steps = int(steps)

    def integrate(
        self,
        drift,
        diffusion,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx = b dt + g dW`` from ``t0`` to ``t1``."""
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        sqrt_dt = math.sqrt(abs(dt))
        x = x0
        t = t0

        for _ in range(steps):
            g = _call_diffusion(diffusion, x, t)
            noise = torch.randn_like(x)
            x = x + drift(x, t) * dt + g * sqrt_dt * noise
            t = t + dt

        return x


def _call_diffusion(diffusion, x: torch.Tensor, t: float) -> torch.Tensor:
    """
    Helper function to call the diffusion function with the correct arguments.
    Supports both ``diffusion(t)`` and general ``diffusion(x, t)`` calls."""
    try:
        signature = inspect.signature(diffusion)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        has_varargs = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL
            for param in signature.parameters.values()
        )
        if has_varargs or len(positional) >= 2:
            return diffusion(x, t)
        return diffusion(t)

    try:
        return diffusion(x, t)
    except TypeError:
        return diffusion(t)
