"""``BrownianBridgeInterpolant`` structural and analytic-baseline tests.

Originally these tests pinned the new interpolant against the legacy
``BrownianBridgePath`` and ``BrownianGeneratorPath``.  Both legacy
classes were deleted in stage 5; this file's equivalence tests now
use **inline analytic baselines** computed from the bridge formulas
directly, so a future refactor of the interpolant body that breaks
the math fails against the formula itself, not against a stale
reference.
"""
from __future__ import annotations

import pytest
import torch

from nami import (
    BrownianBridgeInterpolant,
    GeneratorParams,
    ItoGeneratorOperator,
    LinearInterpolant,
    Score,
    Velocity,
)
from nami.parameterizations import X0, Epsilon

SIGMA = 0.7  # non-default to make sure sigma propagates correctly
EPS = 1e-5


@pytest.fixture
def interpolant() -> BrownianBridgeInterpolant:
    return BrownianBridgeInterpolant(sigma=SIGMA, eps=EPS)


def _bridge_xt(x_data, x_noise, t, z, *, sigma):
    """Closed-form bridge sample: (1-t) x_noise + t x_data + sigma sqrt(t(1-t)) z."""
    tt = t.reshape(t.shape + (1,) * (x_data.ndim - t.ndim))
    mu = (1.0 - tt) * x_noise + tt * x_data
    std = sigma * torch.sqrt(tt * (1.0 - tt))
    return mu + std * z


def _bridge_velocity(x_data, x_noise, t, xt, *, eps):
    """Closed-form bridge conditional velocity at xt.

    u_t(x_t) = (x_data - x_noise) + (2t-1)/(2 t(1-t)) * (x_t - mu_t)
    """
    tt = t.reshape(t.shape + (1,) * (x_data.ndim - t.ndim))
    mu = (1.0 - tt) * x_noise + tt * x_data
    denom = 2.0 * torch.clamp(tt * (1.0 - tt), min=eps)
    coeff = (2.0 * tt - 1.0) / denom
    return (x_data - x_noise) + coeff * (xt - mu)


def _bridge_score(x_data, x_noise, t, xt, *, sigma, eps):
    """Closed-form bridge conditional score: ∇ log p_t(x_t)."""
    tt = t.reshape(t.shape + (1,) * (x_data.ndim - t.ndim))
    mu = (1.0 - tt) * x_noise + tt * x_data
    var = sigma**2 * torch.clamp(tt * (1.0 - tt), min=eps)
    return (mu - xt) / var


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float64 batch with explicit noise so sample is deterministic."""
    torch.manual_seed(0)
    x_data = torch.randn(16, 4, dtype=torch.float64)
    x_noise = torch.randn(16, 4, dtype=torch.float64)
    # Avoid endpoints — both Velocity and Score divide by t(1-t).
    t = 0.05 + 0.9 * torch.rand(16, dtype=torch.float64)
    z = torch.randn(16, 4, dtype=torch.float64)
    return x_data, x_noise, t, z


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_matches_analytic_bridge_formula(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t, z = batch
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    expected = _bridge_xt(x_data, x_noise, t, z, sigma=SIGMA)
    assert torch.allclose(state.xt, expected, atol=1e-12, rtol=1e-12)


def test_sample_state_carries_noise(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t, z = batch
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    # Unlike LinearInterpolant, the bridge needs the noise z to define
    # x_t — so it lives on the state for downstream consumers.
    assert state.noise is not None
    assert torch.equal(state.noise, z)


def test_sample_with_no_noise_draws_internally(
    interpolant: BrownianBridgeInterpolant,
) -> None:
    """Calling sample() without noise produces a fresh draw — useful for
    the regression_loss path where t is auto-sampled and z is
    unsupplied.
    """
    torch.manual_seed(0)
    x_data = torch.randn(8, 3)
    x_noise = torch.randn(8, 3)
    t = 0.05 + 0.9 * torch.rand(8)

    state_a = interpolant.sample(x_data, x_noise, t)
    state_b = interpolant.sample(x_data, x_noise, t)
    # Different draws → different xt.
    assert not torch.allclose(state_a.xt, state_b.xt)


# ---------------------------------------------------------------------------
# Targets — equivalence with legacy path classes (the duplication-collapse claim)
# ---------------------------------------------------------------------------


def test_velocity_matches_analytic_bridge_velocity(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Pins: Velocity target = ``(x_data - x_noise) + (2t-1)/(2 t(1-t))*(x_t - mu_t)``."""
    x_data, x_noise, t, z = batch
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    expected = _bridge_velocity(x_data, x_noise, t, state.xt, eps=EPS)
    actual = interpolant.target(Velocity(), state)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_score_matches_analytic_bridge_score(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Pins: Score target = ``(mu_t - x_t) / (sigma2 * t(1-t))``."""
    x_data, x_noise, t, z = batch
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    expected = _bridge_score(x_data, x_noise, t, state.xt, sigma=SIGMA, eps=EPS)
    actual = interpolant.target(Score(), state)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "diffusion_mode", ["none", "diagonal"], ids=["drift_only", "drift_diffusion"]
)
def test_generator_params_drift_equals_velocity_target(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    diffusion_mode: str,
) -> None:
    """Pins: GeneratorParams drift ≡ analytic bridge velocity.

    The generator drift on the bridge IS the conditional velocity
    (for an Ito operator), plus an optional sigma-valued diagonal
    diffusion when the operator asks for one.  Pinned against the
    closed-form bridge velocity formula, decoupled from the
    interpolant body.
    """
    x_data, x_noise, t, z = batch
    operator = ItoGeneratorOperator(
        event_shape=x_data.shape[-1:], diffusion=diffusion_mode
    )
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    expected_drift = _bridge_velocity(x_data, x_noise, t, state.xt, eps=EPS)

    if diffusion_mode == "none":
        expected = operator.pack_params(drift=expected_drift)
    else:
        expected = operator.pack_params(
            drift=expected_drift,
            diffusion=torch.full_like(x_data, SIGMA),
        )

    actual = interpolant.target(GeneratorParams(operator=operator), state)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Targets — unsupported variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_cls", [Epsilon, X0], ids=["epsilon", "x0"])
def test_unsupported_targets_raise(
    interpolant: BrownianBridgeInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_cls: type,
) -> None:
    x_data, x_noise, t, z = batch
    state = interpolant.sample(x_data, x_noise, t, noise=z)
    with pytest.raises(NotImplementedError, match="Brownian-bridge increment"):
        interpolant.target(target_cls(), state)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_sigma_or_eps_rejected() -> None:
    with pytest.raises(ValueError, match="sigma"):
        BrownianBridgeInterpolant(sigma=-0.1)
    with pytest.raises(ValueError, match="eps"):
        BrownianBridgeInterpolant(eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        BrownianBridgeInterpolant(eps=0.6)


def test_linear_interpolant_generator_params_drift_equals_velocity() -> None:
    """Sibling check: linear-path drift target is just ``x_data - x_noise``,
    packed via the operator (with zero diffusion for the diagonal mode).

    The legacy ``LinearGeneratorPath`` was deleted in stage 3d; this
    test pins the analytic claim directly without that reference.
    """
    torch.manual_seed(0)
    x_data = torch.randn(16, 3, dtype=torch.float64)
    x_noise = torch.randn(16, 3, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64)

    interpolant = LinearInterpolant()
    state = interpolant.sample(x_data, x_noise, t)

    expected_drift = x_data - x_noise

    for diffusion_mode in ("none", "diagonal"):
        op = ItoGeneratorOperator(event_shape=(3,), diffusion=diffusion_mode)
        if diffusion_mode == "none":
            expected = op.pack_params(drift=expected_drift)
        else:
            expected = op.pack_params(
                drift=expected_drift,
                diffusion=torch.zeros_like(x_data),
            )
        actual = interpolant.target(GeneratorParams(operator=op), state)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12), (
            f"diffusion={diffusion_mode}: LinearInterpolant GeneratorParams "
            "drifted from analytic baseline"
        )
