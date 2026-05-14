from __future__ import annotations

from typing import get_args

import pytest
import torch

from nami import (
    BrownianBridgeInterpolant,
    CosineInterpolant,
    Diffusion,
    GaussianInterpolant,
    Heun,
    LinearInterpolant,
    LogDensityHead,
    Parameterization,
    StochasticLinearInterpolant,
    Velocity,
    VPrediction,
    epsilon_prediction,
    v_prediction,
)
from nami.diffusion import expand_like
from nami.fields.composite import TwoHeadField
from nami.lazy import UnconditionalField
from nami.parameterizations import Action, Target
from nami.schedules.vp import VPSchedule


def test_v_target_gaussian_matches_alpha_eps_minus_sigma_x0() -> None:
    """``v = alpha(t) eps - sigma(t) x0`` per Salimans & Ho's convention.

    With nami's ``eps = x_noise`` / ``x0 = x_data`` mapping, the
    closed form is ``v = alpha(t) x_noise - sigma(t) x_data`` — pinned
    here against the schedule's ``alpha`` / ``sigma`` functions.
    """
    torch.manual_seed(0)
    schedule = VPSchedule()
    interpolant = GaussianInterpolant(schedule=schedule)
    x_data = torch.randn(8, 3, dtype=torch.float64)
    x_noise = torch.randn(8, 3, dtype=torch.float64)
    t = 0.05 + 0.9 * torch.rand(8, dtype=torch.float64)

    state = interpolant.sample(x_data, x_noise, t)
    v_actual = interpolant.target(VPrediction(), state)

    a = schedule.alpha(t).unsqueeze(-1)
    s = schedule.sigma(t).unsqueeze(-1)
    v_expected = a * x_noise - s * x_data
    assert torch.allclose(v_actual, v_expected, atol=1e-12, rtol=1e-12)


def test_v_target_linear_unsupported() -> None:
    interpolant = LinearInterpolant()
    state = interpolant.sample(torch.randn(4, 3), torch.randn(4, 3), torch.rand(4))
    with pytest.raises(NotImplementedError, match="VPrediction"):
        interpolant.target(VPrediction(), state)


def test_v_target_brownian_bridge_unsupported() -> None:
    interpolant = BrownianBridgeInterpolant()
    state = interpolant.sample(
        torch.randn(4, 3), torch.randn(4, 3), 0.05 + 0.9 * torch.rand(4)
    )
    with pytest.raises(NotImplementedError, match="VPrediction"):
        interpolant.target(VPrediction(), state)


def test_v_prediction_factory_returns_parameterization_with_v_target() -> None:
    schedule = VPSchedule()
    p = v_prediction(schedule)
    assert isinstance(p, Parameterization)
    assert isinstance(p.target, VPrediction)
    # Default weighting is uniform.
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.equal(p.weighting(t), torch.ones_like(t))


def test_v_prediction_works_end_to_end_in_diffusion_process() -> None:
    """``Diffusion`` accepts ``v_prediction`` and produces samples
    consistent with ``epsilon_prediction`` when the field outputs are
    algebraically related (closing the inconsistency the stage-4
    review flagged: ``v_prediction`` was exported but ``Diffusion``
    rejected ``VPrediction``).
    """
    schedule = VPSchedule()
    solver = Heun(steps=12)

    # The "true" eps predicted by both networks (in their native
    # parameterisations).  Equivalence holds when the v-network emits
    # the v-target corresponding to the same underlying eps.
    torch.manual_seed(0)
    y_eps = float(0.3 * torch.randn(1))

    def eps_field(x, _t, _c=None):
        return torch.full_like(x, y_eps)

    def v_field(x, t, _c=None):
        # v = alpha eps - sigma x0, with eps = y_eps and x0 derived from x_t = alpha x0 + sigma eps
        # so x0 = (x - sigma y_eps) / alpha.
        alpha = expand_like(schedule.alpha(t), x)
        sigma = expand_like(schedule.sigma(t), x)
        x0 = (x - sigma * y_eps) / alpha
        return alpha * y_eps - sigma * x0

    common = {"schedule": schedule, "solver": solver, "event_shape": (), "t1": 1e-3}

    torch.manual_seed(42)
    s_eps = Diffusion(
        UnconditionalField(eps_field),
        parameterization=epsilon_prediction(schedule),
        **common,
    )().sample(sample_shape=(8, 3))

    torch.manual_seed(42)
    s_v = Diffusion(
        UnconditionalField(v_field),
        parameterization=v_prediction(schedule),
        **common,
    )().sample(sample_shape=(8, 3))

    assert torch.allclose(s_eps, s_v, atol=1e-5, rtol=1e-5), (
        "v_prediction in Diffusion diverged from epsilon_prediction — "
        "v_to_eps conversion is wrong"
    )


def test_log_density_head_constructs_and_runs() -> None:
    head = LogDensityHead(dim=3)
    x = torch.randn(4, 3)
    t = torch.rand(4)
    out = head(x, t)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_two_head_field_protocol_is_runtime_checkable() -> None:
    """Protocol is decorated ``@runtime_checkable`` so callers can
    ``isinstance``-test it.  Pins that a stub conforming to the
    declared ``__call__`` + ``event_ndim`` shape passes the check.
    """

    class _Stub:
        event_ndim = 1

        def __call__(self, x, t, c=None):  # noqa: ARG002
            return torch.zeros_like(x)

    stub = _Stub()
    assert isinstance(stub, TwoHeadField)
    out = stub(torch.zeros(2, 3), torch.zeros(2))
    assert out.shape == (2, 3)


def test_two_head_field_rejects_non_conformer() -> None:
    """Pins runtime-checkable behaviour: a class missing
    ``event_ndim`` must fail the ``isinstance`` check.
    """

    class _MissingEventNdim:
        def __call__(self, x, t, c=None):  # noqa: ARG002
            return torch.zeros_like(x)

    assert not isinstance(_MissingEventNdim(), TwoHeadField)


def _state_for(interpolant) -> object:
    x_data = torch.randn(4, 3)
    x_noise = torch.randn(4, 3)
    t = 0.05 + 0.9 * torch.rand(4)
    return interpolant.sample(x_data, x_noise, t)


@pytest.mark.parametrize(
    ("interpolant", "name"),
    [
        (LinearInterpolant(), "LinearInterpolant"),
        (StochasticLinearInterpolant(), "StochasticLinearInterpolant"),
        (CosineInterpolant(), "CosineInterpolant"),
        (BrownianBridgeInterpolant(), "BrownianBridgeInterpolant"),
    ],
    ids=["linear", "stochastic_linear", "cosine", "bridge"],
)
def test_action_target_returns_conditional_velocity(interpolant, name) -> None:  # noqa: ARG001
    """The action target *is* the conditional velocity nabla_x s should match.

    Pinned bit-exactly against the ``Velocity`` arm for the deterministic
    interpolants; the stochastic ones get a shape check (the velocity
    carries gamma*z / bridge-correction terms whose value depends on the
    sampled noise, but ``state.noise`` is shared by both calls so the
    return must still match).
    """
    state = _state_for(interpolant)
    v = interpolant.target(Velocity(), state)
    a = interpolant.target(Action(), state)
    assert a.shape == v.shape
    assert torch.allclose(a, v, atol=1e-12, rtol=1e-12)


def test_action_target_gaussian_still_raises() -> None:
    """``GaussianInterpolant`` cannot express the conditional velocity
    without alpha'(t), sigma'(t); ``Action`` raises for the same reason
    ``Velocity`` does there.
    """
    interpolant = GaussianInterpolant(schedule=VPSchedule())
    state = _state_for(interpolant)
    with pytest.raises(NotImplementedError, match="Action"):
        interpolant.target(Action(), state)


def test_action_is_a_target_union_member() -> None:
    """Pins that ``Action`` is part of the ``Target`` sum type, so
    static checkers flag any consumer that forgets to add a
    ``case Action():`` arm.
    """
    assert Action in get_args(Target)


def test_action_parameterization_constructs_with_default_weighting() -> None:
    """Pins that ``Action`` slots into ``Parameterization`` like the
    other variants — the factory + loss arrive in the follow-up MR but
    the variant must already compose with the existing class today.
    """
    p = Parameterization(target=Action())
    t = torch.linspace(0.05, 0.95, 8)
    assert torch.equal(p.weighting(t), torch.ones_like(t))
    assert isinstance(p.target, Action)
