from __future__ import annotations

import pytest
import torch

from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama


def test_rk4_rejects_invalid_init_and_step_overrides() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        RK4(steps=0)

    solver = RK4(steps=2)
    x0 = torch.ones(2, 3)

    def f(x, _t):
        return x

    with pytest.raises(ValueError, match="steps must be positive"):
        solver.integrate(f, x0, t0=0.0, t1=1.0, steps=-1)
    with pytest.raises(ValueError, match="steps must be positive"):
        solver.integrate_augmented(
            lambda x, _t: (x, torch.ones(x.shape[0])),
            x0,
            torch.zeros(2),
            t0=0.0,
            t1=1.0,
            steps=-1,
        )


def test_euler_maruyama_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        EulerMaruyama(steps=0)

    solver = EulerMaruyama(steps=2)
    x0 = torch.ones(2, 3)

    with pytest.raises(ValueError, match="steps must be positive"):
        solver.integrate(
            lambda x, _t: torch.zeros_like(x),
            lambda _x, _t: torch.ones_like(_x),
            x0,
            t0=0.0,
            t1=1.0,
            steps=-1,
        )


def test_euler_maruyama_accepts_time_only_diffusion() -> None:
    solver = EulerMaruyama(steps=2)
    x0 = torch.ones(2, 3)

    out = solver.integrate(
        lambda x, _t: torch.zeros_like(x),
        lambda _t: torch.zeros_like(x0),
        x0,
        t0=0.0,
        t1=1.0,
    )

    assert torch.equal(out, x0)
