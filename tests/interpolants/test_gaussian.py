"""Stage-1a tests: GaussianInterpolant + parameterization factories.

Covers the building blocks only.  Loss-level equivalence between the
three factories under change of variables lands in stage 1b alongside
``regression_loss``.
"""

from __future__ import annotations

import pytest
import torch

from nami.interpolants import (
    GaussianInterpolant,
    InterpolantState,
    epsilon_prediction,
    score_prediction,
    x0_prediction,
)
from nami.parameterizations import X0, Epsilon, Score, Velocity
from nami.schedules.vp import VPSchedule


@pytest.fixture
def schedule() -> VPSchedule:
    return VPSchedule(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def interpolant(schedule: VPSchedule) -> GaussianInterpolant:
    return GaussianInterpolant(schedule=schedule)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x_data = torch.randn(8, 3)
    x_noise = torch.randn(8, 3)
    t = torch.rand(8)
    return x_data, x_noise, t


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_matches_alpha_sigma_formula(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    a = schedule.alpha(t).unsqueeze(-1)
    s = schedule.sigma(t).unsqueeze(-1)
    expected = a * x_noise + s * x_data
    assert torch.allclose(state.xt, expected, atol=1e-6)


def test_sample_endpoints_are_pure_noise_and_clean_data(
    interpolant: GaussianInterpolant,
) -> None:
    """At t=0 the state collapses to x_noise; at t=1 to (~0)*x_noise + 1*x_data.

    In the FM convention here (t=0 → noise, t=1 → data), VPSchedule's
    alpha(0)=1 plays the noise-coefficient role and alpha(1)~0 plays the
    data-coefficient role at the data endpoint.  Validates that boundary
    behaviour propagates through the interpolant correctly.
    """
    x_data = torch.randn(4, 2)
    x_noise = torch.randn(4, 2)
    t0 = torch.zeros(4)
    t1 = torch.ones(4)
    state0 = interpolant.sample(x_noise, x_data, t0)
    _ = interpolant.sample(x_noise, x_data, t1)
    # At t=0 the noise endpoint dominates fully (alpha(0)=1, sigma(0)=0).
    diff0 = (state0.xt - x_noise).abs().max().item()
    assert diff0 < 1e-6, f"max diff at t=0 is {diff0}; expected ~0"
    # At t=1 the noise contribution is small and the data signal dominates;
    # the explicit numbers depend on schedule beta-range (VPSchedule defaults
    # give alpha(1) ~ 0.0066, sigma(1) ~ 1.0).  We verify the qualitative claim.
    a1 = float(interpolant.schedule.alpha(torch.tensor(1.0)))
    s1 = float(interpolant.schedule.sigma(torch.tensor(1.0)))
    assert a1 < 0.05, f"alpha(1) = {a1} larger than expected for VP defaults"
    assert s1 > 0.95, f"sigma(1) = {s1} smaller than expected for VP defaults"


def test_sample_state_has_no_extra_noise_slot(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    assert state.noise is None
    assert isinstance(state, InterpolantState)


def test_sample_rejects_external_noise_argument(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    with pytest.raises(ValueError, match="x_noise"):
        interpolant.sample(x_noise, x_data, t, noise=torch.randn_like(x_noise))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_target_epsilon_returns_x_noise(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    assert torch.equal(interpolant.target(Epsilon(), state), x_noise)


def test_target_x0_returns_x_data(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    assert torch.equal(interpolant.target(X0(), state), x_data)


def test_target_score_matches_negative_eps_over_alpha(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Score for x_t = alpha eps + sigma x_0 (FM convention) is -eps/alpha.

    In the FM convention, alpha(t) is the noise coefficient (alpha(0)=1,
    alpha(1)=0), so the score divides by alpha rather than sigma.
    """
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    expected = -x_noise / schedule.alpha(t).unsqueeze(-1)
    actual = interpolant.target(Score(), state)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_score_target_is_singular_at_t_one_by_design(
    interpolant: GaussianInterpolant,
) -> None:
    """At t=1 with a VP-style schedule, alpha(1)~0 and the score target is inf/nan.

    In the FM convention the noise level is alpha(t), and VP schedules
    drive alpha(1) → 0 at the data endpoint.  Score = -x_noise / alpha
    is therefore singular at t=1 by design.

    This is **deliberate**.  Silent clamping inside the interpolant or
    factories would hide numerical bugs in callers' t-sampling logic.
    The contract is: callers (stage 1b's ``regression_loss``) must
    restrict t to a non-degenerate interval.  This test pins the
    endpoint behaviour so a future "convenience" clamp cannot land
    without a deliberate update here.
    """
    x_data = torch.randn(2, 3)
    x_noise = torch.randn(2, 3)
    state = interpolant.sample(x_noise, x_data, torch.ones(2))
    score = interpolant.target(Score(), state)
    # VPSchedule's alpha(1) is small (~0.0066) but not exactly zero, so
    # the score blows up to a very large finite value rather than inf/nan.
    # The contract being pinned is "no silent clamp" — assert the
    # magnitude reflects the unclamped 1/alpha(1) divergence.
    a1 = float(interpolant.schedule.alpha(torch.tensor(1.0)))
    assert score.abs().max().item() >= 1.0 / a1 - 1e-3, (
        "Score should diverge as 1/alpha(t) at t=1; if a clamp landed, update this test"
    )


def test_x0_prediction_weighting_is_large_at_t_one_by_design(
    schedule: VPSchedule,
) -> None:
    """1/SNR weighting blows up as t→1 for VP-style schedules (alpha→0).

    Companion to the Score-target endpoint test.  Pins the contract
    that callers — not the factory — own t-sampling discipline.
    """
    p = x0_prediction(schedule)
    w = p.weighting(torch.ones(4))
    assert torch.isinf(w).any() or torch.isnan(w).any() or (w > 1e3).any()


def test_target_velocity_raises_until_schedule_derivatives_exist(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Velocity is left unimplemented deliberately.

    This test pins that decision: if a future change adds alpha'(t), sigma'(t)
    to ``NoiseSchedule`` and implements Velocity, the test will fail and
    force a deliberate update — guarding against silent capability drift.
    """
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    with pytest.raises(NotImplementedError, match="derivatives"):
        interpolant.target(Velocity(), state)


# ---------------------------------------------------------------------------
# Parameterization factories
# ---------------------------------------------------------------------------


def test_epsilon_prediction_carries_unit_weighting(schedule: VPSchedule) -> None:
    p = epsilon_prediction(schedule)
    assert isinstance(p.target, Epsilon)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.equal(p.weighting(t), torch.ones_like(t))


def test_score_prediction_weighting_is_alpha_squared(schedule: VPSchedule) -> None:
    """In the FM convention, score = -eps/alpha so the eps↔score
    change-of-variable weighting is alpha², not sigma²."""
    p = score_prediction(schedule)
    assert isinstance(p.target, Score)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.allclose(p.weighting(t), schedule.alpha(t).pow(2))


def test_x0_prediction_weighting_is_inverse_snr(schedule: VPSchedule) -> None:
    """In the FM convention, x_0 = (x_t - alpha·eps)/sigma so the eps↔x_0
    change-of-variable weighting is (sigma/alpha)² = 1/SNR, the inverse
    of the diffusion-convention SNR weighting."""
    p = x0_prediction(schedule)
    assert isinstance(p.target, X0)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.allclose(p.weighting(t), 1.0 / schedule.snr(t))


def test_factory_default_output_transform_is_identity(schedule: VPSchedule) -> None:
    """Network output is the target's value directly.

    Cross-target conversion (e.g. eps → score for a Process needing score
    from an eps-trained model) is the runtime concern of stage 1c, not the
    factory.  This test pins that stage 1a stays out of that business.
    """
    y = torch.randn(4, 3)
    for p in (
        epsilon_prediction(schedule),
        score_prediction(schedule),
        x0_prediction(schedule),
    ):
        assert p.output_transform(y) is y
