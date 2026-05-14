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
    x_target = torch.randn(8, 3)
    x_source = torch.randn(8, 3)
    t = torch.rand(8)
    return x_target, x_source, t


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_matches_alpha_sigma_formula(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
    a = schedule.alpha(t).unsqueeze(-1)
    s = schedule.sigma(t).unsqueeze(-1)
    expected = a * x_target + s * x_source
    assert torch.allclose(state.xt, expected, atol=1e-6)


def test_sample_endpoints_are_clean_data_and_pure_noise(
    interpolant: GaussianInterpolant,
) -> None:
    """At t=0 the state collapses to x_target; at t=1 to (~0)*x_target + 1*x_source.

    Validates that VPSchedule's alpha(0)=1, sigma(0)=0 and alpha(1)~0, sigma(1)~1
    boundary behaviour propagates through the interpolant correctly —
    the convention nami's docs state and the rest of the library
    depends on.
    """
    x_target = torch.randn(4, 2)
    x_source = torch.randn(4, 2)
    t0 = torch.zeros(4)
    t1 = torch.ones(4)
    state0 = interpolant.sample(x_target, x_source, t0)
    _ = interpolant.sample(x_target, x_source, t1)
    # At t=0 the data signal is fully present, noise contribution is zero.
    diff0 = (state0.xt - x_target).abs().max().item()
    assert diff0 < 1e-6, f"max diff at t=0 is {diff0}; expected ~0"
    # At t=1 the data signal is small and the noise contribution dominates;
    # the explicit numbers depend on schedule beta-range (VPSchedule defaults
    # give alpha(1) ~ 0.0066, sigma(1) ~ 1.0).  We verify the qualitative claim,
    # not the exact value.
    a1 = float(interpolant.schedule.alpha(torch.tensor(1.0)))
    s1 = float(interpolant.schedule.sigma(torch.tensor(1.0)))
    assert a1 < 0.05, f"alpha(1) = {a1} larger than expected for VP defaults"
    assert s1 > 0.95, f"sigma(1) = {s1} smaller than expected for VP defaults"


def test_sample_state_has_no_extra_noise_slot(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
    assert state.noise is None
    assert isinstance(state, InterpolantState)


def test_sample_rejects_external_noise_argument(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_target, x_source, t = batch
    with pytest.raises(ValueError, match="x_source"):
        interpolant.sample(x_target, x_source, t, noise=torch.randn_like(x_source))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_target_epsilon_returns_x_source(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
    assert torch.equal(interpolant.target(Epsilon(), state), x_source)


def test_target_x0_returns_x_target(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
    assert torch.equal(interpolant.target(X0(), state), x_target)


def test_target_score_matches_negative_eps_over_sigma(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Score for x_t = alpha x_0 + sigma eps is -eps/sigma.  This test pins that formula."""
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
    expected = -x_source / schedule.sigma(t).unsqueeze(-1)
    actual = interpolant.target(Score(), state)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_score_target_is_singular_at_t_zero_by_design(
    interpolant: GaussianInterpolant,
) -> None:
    """At t=0 with a VP-style schedule, sigma(0)=0 and the score target is inf/nan.

    This is **deliberate**.  Silent clamping inside the interpolant or
    factories would hide numerical bugs in callers' t-sampling logic.
    The contract is: callers (stage 1b's ``regression_loss``) must
    restrict t to a non-degenerate interval.  This test pins the
    endpoint behaviour so a future "convenience" clamp cannot land
    without a deliberate update here.
    """
    x_target = torch.randn(2, 3)
    x_source = torch.randn(2, 3)
    state = interpolant.sample(x_target, x_source, torch.zeros(2))
    score = interpolant.target(Score(), state)
    assert torch.isinf(score).any() or torch.isnan(score).any(), (
        "Score should be singular at t=0; if a clamp landed, update this test"
    )


def test_x0_prediction_weighting_is_singular_at_t_zero_by_design(
    schedule: VPSchedule,
) -> None:
    """SNR weighting diverges at t=0 for VP-style schedules.

    Companion to the Score-target endpoint test.  Pins the contract
    that callers — not the factory — own t-sampling discipline.
    """
    p = x0_prediction(schedule)
    w = p.weighting(torch.zeros(4))
    assert torch.isinf(w).any() or torch.isnan(w).any() or (w > 1e10).any()


def test_target_velocity_raises_until_schedule_derivatives_exist(
    interpolant: GaussianInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Velocity is left unimplemented deliberately.

    This test pins that decision: if a future change adds alpha'(t), sigma'(t)
    to ``NoiseSchedule`` and implements Velocity, the test will fail and
    force a deliberate update — guarding against silent capability drift.
    """
    x_target, x_source, t = batch
    state = interpolant.sample(x_target, x_source, t)
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


def test_score_prediction_weighting_is_sigma_squared(schedule: VPSchedule) -> None:
    p = score_prediction(schedule)
    assert isinstance(p.target, Score)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.allclose(p.weighting(t), schedule.sigma(t).pow(2))


def test_x0_prediction_weighting_is_snr(schedule: VPSchedule) -> None:
    p = x0_prediction(schedule)
    assert isinstance(p.target, X0)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.allclose(p.weighting(t), schedule.snr(t))


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
