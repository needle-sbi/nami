from __future__ import annotations

import pytest
import torch

from nami import (
    RK4,
    ConsistencyFlowMatching,
    ExactDivergence,
    StandardNormal,
)


class _PlainField(torch.nn.Module):
    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return torch.zeros_like(x)
        return (x * 0) + c.expand_as(x)


def test_cfm_sample_smoke():
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(3,)),
    )()

    sample = process.sample(sample_shape=(5,))

    assert sample.shape == (5, 3)
    assert torch.isfinite(sample).all()


def test_cfm_sample_without_solver():
    """Sampling does not require a solver (single-step)."""
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        solver=None,
    )()

    sample = process.sample(sample_shape=(4,))

    assert sample.shape == (4, 2)


def test_cfm_rsample_smoke():
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(3,)),
    )()

    sample = process.rsample(sample_shape=(5,))

    assert sample.shape == (5, 3)
    assert torch.isfinite(sample).all()


def test_cfm_log_prob_requires_solver():
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        solver=None,
    )()
    x = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="log_prob requires a solver"):
        process.log_prob(x)


def test_cfm_log_prob_smoke():
    context = torch.randn(5, 1)
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )(context)
    x = torch.zeros(5, 2)

    log_prob = process.log_prob(x, estimator=ExactDivergence(max_dim=8))

    assert log_prob.shape == (5,)
    assert torch.isfinite(log_prob).all()


def test_cfm_context_expands_over_samples():
    context = torch.randn(3, 1)
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
    )(context)

    sample = process.sample(sample_shape=(4,))

    assert sample.shape == (4, 3, 2)


def test_cfm_invert_smoke():
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(3,)),
    )()
    x = torch.randn(4, 3)

    z = process.invert(x)

    assert z.shape == (4, 3)
    assert torch.isfinite(z).all()


def test_cfm_invert_round_trip_with_identity_field():
    """With a zero velocity field, sample and invert are both identity maps,
    so round-tripping returns the original noise."""

    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(3,)),
    )()

    torch.manual_seed(42)
    z = torch.randn(8, 3)

    # With zero velocity: sample(z) = z + (0-1)*0 = z, invert(x) = x + (1-0)*0 = x
    x = z + (0.0 - 1.0) * torch.zeros_like(z)  # = z
    z_hat = process.invert(x)

    assert torch.allclose(z_hat, z, atol=1e-6)


class _LinearField(torch.nn.Module):
    """Predicts the true conditional velocity u_t = x_noise - x_data.

    For round-trip testing: if v is exact, then f(g(x)) = x and g(f(z)) = z.
    This field stores (x_data, x_noise) so it can return the exact velocity.
    """

    event_ndim = 1

    def __init__(self, x_data, x_noise):
        super().__init__()
        self._v = x_noise - x_data

    def forward(self, x, t, c=None):
        _ = t, c
        return self._v.expand_as(x)


def test_cfm_invert_round_trip_with_linear_field():
    """With the exact conditional velocity, sample → invert recovers the
    original noise (up to floating-point precision).

    sample: x = z + (t1 - t0) * v = z + (0 - 1) * v = z - v
    invert: z' = x + (t0 - t1) * v = x + (1 - 0) * v = x + v = z
    """
    x_data = torch.randn(6, 4)
    x_noise = torch.randn(6, 4)
    field = _LinearField(x_data, x_noise)

    process = ConsistencyFlowMatching(
        field,
        StandardNormal(event_shape=(4,)),
    )()

    z = torch.randn(6, 4)
    # Manual single-step sample: x = z - v
    x = z + (0.0 - 1.0) * field._v
    z_hat = process.invert(x)

    assert torch.allclose(z_hat, z, atol=1e-5)


# ---------------------------------------------------------------------------
# One-step log_prob via h_head
# ---------------------------------------------------------------------------


class _SimpleHHead(torch.nn.Module):
    """Minimal scalar head for testing."""

    event_ndim = 1

    def __init__(self, value: float = -1.0):
        super().__init__()
        self._value = value

    def forward(self, x, t, c=None):  # noqa: ARG002
        return torch.full(x.shape[:1], self._value)


def test_cfm_log_prob_one_step_with_h_head():
    """When h_head is provided, log_prob uses it (one-step, no solver)."""
    h_head = _SimpleHHead(value=-3.14)
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        solver=None,
        h_head=h_head,
    )()
    x = torch.randn(4, 2)

    log_p = process.log_prob(x)

    assert log_p.shape == (4,)
    assert torch.allclose(log_p, torch.tensor(-3.14))


def test_cfm_log_prob_ode_fallback_with_h_head():
    """With ode=True, log_prob uses ODE integration even if h_head exists."""
    h_head = _SimpleHHead(value=-999.0)
    context = torch.randn(5, 1)
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
        h_head=h_head,
    )(context)
    x = torch.zeros(5, 2)

    log_p = process.log_prob(x, ode=True, estimator=ExactDivergence(max_dim=8))

    assert log_p.shape == (5,)
    # Should NOT be the h_head constant value
    assert not torch.allclose(log_p, torch.tensor(-999.0))
    assert torch.isfinite(log_p).all()


def test_cfm_log_prob_no_h_head_no_solver_raises():
    """Without h_head or solver, log_prob raises."""
    process = ConsistencyFlowMatching(
        _PlainField(),
        StandardNormal(event_shape=(2,)),
        solver=None,
    )()
    x = torch.randn(4, 2)

    with pytest.raises(ValueError, match="log_prob requires a solver"):
        process.log_prob(x)
