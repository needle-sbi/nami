from __future__ import annotations

import pytest
import torch

from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama, _call_diffusion


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


def test_call_diffusion_handles_signatureless_two_arg_callable() -> None:
    """C builtins like ``max`` expose no inspect signature; the adapter
    falls back to trying ``diffusion(x, t)`` first."""
    out = _call_diffusion(max, torch.tensor(2.0), 0.5)

    assert float(out) == 2.0


def test_call_diffusion_handles_signatureless_time_only_callable() -> None:
    """A signatureless single-argument callable (``torch.sigmoid``) raises
    TypeError on ``(x, t)`` and the adapter retries with ``(t,)``."""
    t = torch.tensor(0.0)
    out = _call_diffusion(torch.sigmoid, torch.ones(3), t)

    assert torch.allclose(out, torch.tensor(0.5))
