"""Action-matching loss + factory tests.

Pins:

* ``action_prediction`` returns a ``Parameterization`` with ``Action``
  target and the conventional uniform ω(t)=1 weighting.
* ``action_matching_loss`` regresses ``∇_x s`` against the interpolant
  velocity — a "perfect" scalar field whose gradient *is* the linear
  velocity should give zero loss.
* The loss is differentiable through the autograd plumbing
  (``create_graph=True``).
* Rejects non-``Action`` parameterisations with an explicit message.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nami import (
    Action,
    ActionHead,
    LinearInterpolant,
    Parameterization,
    StochasticLinearInterpolant,
    Velocity,
    action_matching_loss,
    action_prediction,
)


class _PerfectLinearAction(nn.Module):
    """Closed-form scalar potential whose gradient is the linear velocity.

    For ``LinearInterpolant`` the conditional velocity is the constant
    ``v = x_data - x_noise``.  The scalar potential ``s(x, t) = v * x``
    has ``∇_x s = v`` everywhere, so ``action_matching_loss`` should
    return exactly zero.

    The field must take ``v`` at construction (the loss only sees
    ``(x_t, t, c)`` at call time and the regression target depends on
    the path endpoints — wiring them as Parameters lets us test the
    autograd shape end-to-end).
    """

    event_ndim = 1

    def __init__(self, v: torch.Tensor):
        super().__init__()
        # Register as a buffer so it's part of the module state but not
        # treated as a trainable parameter (the test fixes ``v``).
        self.register_buffer("v", v)

    def forward(self, x: torch.Tensor, _t: torch.Tensor, _c=None) -> torch.Tensor:
        # s(x) = v * x  — broadcast over the batch.
        return (x * self.v).sum(dim=-1)


def test_action_prediction_factory_shape() -> None:
    p = action_prediction()
    assert isinstance(p, Parameterization)
    assert isinstance(p.target, Action)
    t = torch.linspace(0.05, 0.95, 8)
    # Published convention: uniform weighting.
    assert torch.equal(p.weighting(t), torch.ones_like(t))


def test_perfect_scalar_field_gives_zero_loss() -> None:
    """For the linear path, ``s(x) = v * x`` has ``∇_x s = v = u_t`` so
    the regression target is matched exactly.  This is the sanity
    pin that the autograd plumbing returns the right gradient.
    """
    torch.manual_seed(0)
    x_data = torch.randn(16, 3, dtype=torch.float64)
    x_noise = torch.randn(16, 3, dtype=torch.float64)
    # Broadcast: the linear velocity is per-sample but the perfect
    # potential is linear in x with a per-sample coefficient.  To keep
    # the test honest we use one shared v across the batch (test against
    # the *mean* velocity) — and verify it gives zero loss only when
    # x_data / x_noise are constant across the batch.
    x_data = x_data[:1].expand_as(x_data).contiguous()
    x_noise = x_noise[:1].expand_as(x_noise).contiguous()
    v = (x_data - x_noise)[0]
    field = _PerfectLinearAction(v).to(torch.float64)

    loss = action_matching_loss(
        field,
        x_data,
        x_noise,
        interpolant=LinearInterpolant(),
        parameterization=action_prediction(),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-20)


def test_action_loss_is_differentiable() -> None:
    """End-to-end: a trainable ``ActionHead`` produces a loss whose
    ``.backward()`` flows gradient into its parameters.  This is what
    ``create_graph=True`` in the inner ``autograd.grad`` exists for.
    """
    torch.manual_seed(0)
    head = ActionHead(dim=3, hidden=16, layers=1)
    x_data = torch.randn(8, 3)
    x_noise = torch.randn(8, 3)
    loss = action_matching_loss(
        head,
        x_data,
        x_noise,
        interpolant=LinearInterpolant(),
    )
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and (g.abs() > 0).any() for g in grads
    )


def test_action_loss_rejects_non_action_parameterization() -> None:
    head = ActionHead(dim=3, hidden=8, layers=1)
    x_data = torch.randn(4, 3)
    x_noise = torch.randn(4, 3)
    with pytest.raises(TypeError, match="Action"):
        action_matching_loss(
            head,
            x_data,
            x_noise,
            interpolant=LinearInterpolant(),
            parameterization=Parameterization(target=Velocity()),
        )


def test_action_loss_handles_stochastic_linear_interpolant_with_explicit_z() -> None:
    """``StochasticLinearInterpolant.target(Action)`` requires
    ``state.noise``; the loss forwards ``z`` to ``sample`` so a
    pre-supplied noise yields a deterministic loss value.
    """
    torch.manual_seed(0)
    head = ActionHead(dim=3, hidden=8, layers=1)
    x_data = torch.randn(8, 3)
    x_noise = torch.randn(8, 3)
    z = torch.randn(8, 3)
    t = 0.05 + 0.9 * torch.rand(8)

    loss1 = action_matching_loss(
        head,
        x_data,
        x_noise,
        t=t,
        interpolant=StochasticLinearInterpolant(),
        z=z,
    )
    loss2 = action_matching_loss(
        head,
        x_data,
        x_noise,
        t=t,
        interpolant=StochasticLinearInterpolant(),
        z=z,
    )
    assert torch.allclose(loss1, loss2, atol=1e-12, rtol=1e-12)
