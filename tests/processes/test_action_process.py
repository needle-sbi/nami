from __future__ import annotations

import pytest
import torch
from torch import nn

from nami import (
    Action,
    ActionMatching,
    FlowMatching,
    Heun,
    Parameterization,
    StandardNormal,
    Velocity,
    action_prediction,
)


class _LinearActionPotential(nn.Module):
    """``s(x) = v * x``; ``\\nabla_x s = v`` independent of ``(x, t)``."""

    event_ndim = 1

    def __init__(self, dim: int, v_value: float):
        super().__init__()
        self.dim = dim
        self.register_buffer("v", torch.full((dim,), v_value))

    def forward(self, x: torch.Tensor, _t: torch.Tensor, _c=None) -> torch.Tensor:
        return (x * self.v).sum(dim=-1)


class _ConstantVelocity(nn.Module):
    """Matching velocity field ``f(x, t) = v`` for the equivalence test."""

    event_ndim = 1

    def __init__(self, dim: int, v_value: float):
        super().__init__()
        self.dim = dim
        self.register_buffer("v", torch.full((dim,), v_value))

    def forward(self, x: torch.Tensor, _t: torch.Tensor, _c=None) -> torch.Tensor:
        return self.v.expand_as(x)


def test_action_matching_drift_is_grad_of_potential() -> None:
    """Samples from ``ActionMatching(s(x)=v*x, ...)`` match samples from
    ``FlowMatching(f(x,t)=v, ...)`` under the same RNG seed and solver —
    proves the integrator drift is exactly ``\\nabla_x s``.
    """
    dim = 3
    v_value = -0.7
    solver = Heun(steps=8)
    base = StandardNormal(event_shape=(dim,))

    torch.manual_seed(42)
    s_action = ActionMatching(
        _LinearActionPotential(dim, v_value),
        base=base,
        solver=solver,
    )().sample(sample_shape=(16,))

    torch.manual_seed(42)
    s_flow = FlowMatching(
        _ConstantVelocity(dim, v_value),
        base=base,
        solver=solver,
    )().sample(sample_shape=(16,))

    assert torch.allclose(s_action, s_flow, atol=1e-5, rtol=1e-5)


def test_action_matching_rejects_non_action_target() -> None:
    base = StandardNormal(event_shape=(3,))
    solver = Heun(steps=4)
    with pytest.raises(TypeError, match="Action"):
        ActionMatching(
            _LinearActionPotential(3, 0.0),
            base=base,
            solver=solver,
            parameterization=Parameterization(target=Velocity()),
        )()


def test_action_matching_uses_action_prediction_by_default() -> None:
    process = ActionMatching(
        _LinearActionPotential(3, 0.0),
        base=StandardNormal(event_shape=(3,)),
        solver=Heun(steps=2),
    )()
    assert isinstance(process._parameterization.target, Action)


def test_action_matching_log_prob_not_implemented() -> None:
    r"""``log_prob`` requires the Laplacian of ``s`` (\partial/\partialt log p = -\\nabla*\\nabla*s).
    Deferred until a real consumer drives that second-order autograd.
    """
    process = ActionMatching(
        _LinearActionPotential(3, 0.0),
        base=StandardNormal(event_shape=(3,)),
        solver=Heun(steps=2),
    )()
    x = torch.randn(4, 3)
    with pytest.raises(NotImplementedError, match="Laplacian"):
        process.log_prob(x)


def test_action_prediction_returns_uniform_weighting() -> None:
    """Pins the published convention used by ``action_matching_loss``."""
    p = action_prediction()
    t = torch.linspace(0.01, 0.99, 32)
    assert torch.equal(p.weighting(t), torch.ones_like(t))
