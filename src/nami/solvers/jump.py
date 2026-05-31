"""Tau-leaping sampler for pure-jump (CTMC) generators.

Fixed-step forward simulation of a jump process: it advances an integer state
from ``t0`` to ``t1`` in ``steps`` equal increments, delegating the per-step
state update to a caller-supplied transition function. For the masking CTMC the
transition reveals a fraction of the still-masked coordinates at each step (see
:meth:`~nami.generators.ctmc.CTMCGeneratorOperator.jump_step`).
"""

from __future__ import annotations

import torch


class TauLeapingSampler:
    """Fixed-step jump-process integrator.

    Args:
        steps (int): Number of equal time increments from ``t0`` to ``t1``.
    """

    requires_steps = True
    is_sde = False
    supports_rsample = False

    def __init__(self, steps: int = 64):
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        self.steps = int(steps)

    def integrate(
        self,
        transition,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Advance ``x0`` by repeatedly applying ``transition(x, t, dt)``.

        Args:
            transition: Callable ``(x, t, dt) -> x_next`` advancing the state
                over one increment.
            x0 (torch.Tensor): Initial (masked) state.
            t0 (float): Start time.
            t1 (float): End time.
            steps (int | None): Override for the configured step count.

        Returns:
            torch.Tensor: State at ``t1``.
        """
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        dt = (t1 - t0) / steps
        x = x0
        t = t0
        for _ in range(steps):
            x = transition(x, t, dt)
            t = t + dt
        return x
