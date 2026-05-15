from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.diffusion import expand_like
from nami.interpolants import (
    GaussianInterpolant,
    epsilon_prediction,
    score_prediction,
    x0_prediction,
)
from nami.losses import regression_loss
from nami.parameterizations import Epsilon, Parameterization, Score
from nami.schedules.vp import VPSchedule

# ---------------------------------------------------------------------------
# Fake fields that emit consistent predictions across parameterisations.
# ---------------------------------------------------------------------------
# Each field corresponds to the *same underlying eps prediction* y_eps,
# expressed in the parameterisation's native target space:
#
#   eps-prediction:    y_eps                                  (identity)
#   score:           -y_eps / sigma(t)                          (eps_to_score)
#   x0-prediction:   (x_t - sigma(t) y_eps) / alpha(t)              (eps_to_x0)
#
# When the three losses agree, they confirm that the algebraic
# identities and the factory weightings together preserve the loss
# value across parameterisations.


class _ConsistentEpsField(nn.Module):
    event_ndim = 1

    def __init__(self, y_eps: torch.Tensor):
        super().__init__()
        self.register_buffer("y_eps", y_eps)

    def forward(self, x, t, c=None):  # noqa: ARG002
        return self.y_eps


class _ConsistentScoreField(nn.Module):
    event_ndim = 1

    def __init__(self, y_eps: torch.Tensor, schedule: VPSchedule):
        super().__init__()
        self.register_buffer("y_eps", y_eps)
        self.schedule = schedule

    def forward(self, x, t, c=None):  # noqa: ARG002
        # FM convention: score = -eps / alpha(t) (alpha is the noise level).
        alpha = expand_like(self.schedule.alpha(t), self.y_eps)
        return -self.y_eps / alpha


class _ConsistentX0Field(nn.Module):
    event_ndim = 1

    def __init__(self, y_eps: torch.Tensor, schedule: VPSchedule):
        super().__init__()
        self.register_buffer("y_eps", y_eps)
        self.schedule = schedule

    def forward(self, x, t, c=None):  # noqa: ARG002
        # FM convention: x_t = alpha(t) * eps + sigma(t) * x_0,
        # so x_0 = (x_t - alpha(t) * eps) / sigma(t).
        alpha = expand_like(self.schedule.alpha(t), x)
        sigma = expand_like(self.schedule.sigma(t), x)
        return (x - alpha * self.y_eps) / sigma


@pytest.fixture
def schedule() -> VPSchedule:
    return VPSchedule(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def interpolant(schedule: VPSchedule) -> GaussianInterpolant:
    return GaussianInterpolant(schedule=schedule)


@pytest.fixture
def fixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float64 batch with explicit t in (0.05, 0.95) for numerical stability."""
    torch.manual_seed(0)
    x_data = torch.randn(64, 3, dtype=torch.float64)
    x_noise = torch.randn(64, 3, dtype=torch.float64)
    # Avoid endpoints — see GaussianInterpolant docstring.
    t = 0.05 + 0.9 * torch.rand(64, dtype=torch.float64)
    y_eps = x_noise + 0.7 * torch.randn(64, 3, dtype=torch.float64)
    return x_data, x_noise, t, y_eps


# ---------------------------------------------------------------------------
# Headline: factory equivalence
# ---------------------------------------------------------------------------


def test_three_parameterizations_produce_equal_weighted_losses(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    fixed_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """All three factories yield the same per-sample weighted loss on
    consistent network outputs.

    This pins the structural fix: ω(t) carried with the Target choice
    means the *effective* loss is invariant to parameterisation —
    silent re-weighting bugs become impossible because changing target
    *is* changing the Parameterization, not toggling a flag.
    """
    x_data, x_noise, t, y_eps = fixed_batch

    eps_field = _ConsistentEpsField(y_eps).to(dtype=torch.float64)
    score_field = _ConsistentScoreField(y_eps, schedule).to(dtype=torch.float64)
    x0_field = _ConsistentX0Field(y_eps, schedule).to(dtype=torch.float64)

    common = {
        "x_data": x_data,
        "x_noise": x_noise,
        "t": t,
        "interpolant": interpolant,
        "reduction": "none",
    }

    loss_eps = regression_loss(
        eps_field, parameterization=epsilon_prediction(schedule), **common
    )
    loss_score = regression_loss(
        score_field, parameterization=score_prediction(schedule), **common
    )
    loss_x0 = regression_loss(
        x0_field, parameterization=x0_prediction(schedule), **common
    )

    assert torch.allclose(loss_eps, loss_score, atol=1e-9, rtol=1e-7), (
        "eps / score factories disagree — algebraic identity broken"
    )
    assert torch.allclose(loss_eps, loss_x0, atol=1e-9, rtol=1e-7), (
        "eps / x0 factories disagree — algebraic identity broken"
    )


def test_breaking_weighting_target_binding_breaks_equivalence(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
    fixed_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """If you keep eps's weighting (ω=1) but switch the target to Score, the
    losses no longer agree.

    Demonstrates the failure mode the factory bindings are designed to
    prevent.  Without the weighting carried by ``Parameterization``, a
    user could pick ``Score`` as a training target and silently change
    the effective objective by leaving ``ω`` at its old value — exactly
    the Arruda Eq. 57-61 bug.  The factories close that gap; this test
    proves the gap was real.
    """
    x_data, x_noise, t, y_eps = fixed_batch

    eps_field = _ConsistentEpsField(y_eps).to(dtype=torch.float64)
    score_field = _ConsistentScoreField(y_eps, schedule).to(dtype=torch.float64)

    common = {
        "x_data": x_data,
        "x_noise": x_noise,
        "t": t,
        "interpolant": interpolant,
        "reduction": "none",
    }

    loss_eps = regression_loss(
        eps_field, parameterization=epsilon_prediction(schedule), **common
    )
    # Score target with eps's ω=1 — the silent-reweighting bug shape.
    bad = Parameterization(target=Score(), weighting=torch.ones_like)
    loss_score_unweighted = regression_loss(score_field, parameterization=bad, **common)

    # We want to assert *inequality*; the difference should be substantial,
    # not a numerical near-miss.
    rel_diff = (loss_eps - loss_score_unweighted).abs() / loss_eps.clamp_min(1e-12)
    assert rel_diff.max() > 1e-2, (
        f"Expected substantial divergence with broken weighting; "
        f"max rel diff = {rel_diff.max().item()}"
    )


# ---------------------------------------------------------------------------
# t-sampling discipline (the contract from stage 1a)
# ---------------------------------------------------------------------------


def test_default_eps_t_keeps_auto_sampled_t_off_endpoints(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
) -> None:
    """The loss owns endpoint discipline; the interpolant stays pure."""
    torch.manual_seed(0)
    x_data = torch.randn(2048, 3)
    x_noise = torch.randn(2048, 3)

    field = _ConsistentEpsField(torch.zeros_like(x_data))
    captured_t: list[torch.Tensor] = []

    def capture(xt, t, c=None):  # noqa: ARG001
        captured_t.append(t.clone())
        return torch.zeros_like(xt)

    field.forward = capture  # type: ignore[method-assign]

    regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        interpolant=interpolant,
        parameterization=epsilon_prediction(schedule),
        eps_t=1e-3,
    )

    t = captured_t[0]
    assert t.min() >= 1e-3
    assert t.max() <= 1.0 - 1e-3


def test_explicit_t_is_not_silently_clamped(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
) -> None:
    """User-supplied t is the user's contract.  We do not silently
    rescale — that would hide bugs in the caller's sampling logic.
    """
    torch.manual_seed(0)
    x_data = torch.randn(8, 3)
    x_noise = torch.randn(8, 3)
    t_at_zero = torch.zeros(8)  # deliberately at the singularity

    field = _ConsistentEpsField(torch.zeros_like(x_data))

    # Loss should *run* (eps-prediction has no singularity) and use t=0
    # without clamping.  The score-prediction analogue would NaN; that
    # divergence is the user's responsibility to avoid.
    loss = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t_at_zero,
        interpolant=interpolant,
        parameterization=epsilon_prediction(schedule),
    )
    assert torch.isfinite(loss)


def test_invalid_eps_t_raises(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
) -> None:
    x_data = torch.randn(4, 3)
    x_noise = torch.randn(4, 3)
    field = _ConsistentEpsField(torch.zeros_like(x_data))
    for bad in (-0.1, 0.5, 1.0):
        with pytest.raises(ValueError, match="eps_t"):
            regression_loss(
                field,
                x_data=x_data,
                x_noise=x_noise,
                interpolant=interpolant,
                parameterization=epsilon_prediction(schedule),
                eps_t=bad,
            )


# ---------------------------------------------------------------------------
# Smoke / shape / reduction
# ---------------------------------------------------------------------------


def test_reduction_modes(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
) -> None:
    torch.manual_seed(0)
    x_data = torch.randn(16, 3)
    x_noise = torch.randn(16, 3)
    field = _ConsistentEpsField(torch.zeros_like(x_data))
    p = epsilon_prediction(schedule)

    common = {
        "x_data": x_data,
        "x_noise": x_noise,
        "interpolant": interpolant,
        "parameterization": p,
        "t": 0.05 + 0.9 * torch.rand(16),  # fix t for determinism across calls
    }
    none = regression_loss(field, reduction="none", **common)
    mean = regression_loss(field, reduction="mean", **common)
    summ = regression_loss(field, reduction="sum", **common)

    assert none.shape == (16,)
    assert torch.allclose(none.mean(), mean)
    assert torch.allclose(none.sum(), summ)


def test_output_transform_is_applied(
    interpolant: GaussianInterpolant,
    schedule: VPSchedule,
) -> None:
    """A non-identity output_transform changes the loss as expected.

    Pins that ``output_transform`` is in the call path and not silently
    short-circuited.
    """
    torch.manual_seed(0)
    x_data = torch.randn(32, 3, dtype=torch.float64)
    x_noise = torch.randn(32, 3, dtype=torch.float64)
    t = 0.05 + 0.9 * torch.rand(32, dtype=torch.float64)
    field = _ConsistentEpsField(torch.zeros_like(x_data)).to(dtype=torch.float64)

    p_id = epsilon_prediction(schedule)
    p_neg = Parameterization(
        target=Epsilon(),
        weighting=p_id.weighting,
        output_transform=lambda y: -y,
    )

    common = {
        "x_data": x_data,
        "x_noise": x_noise,
        "t": t,
        "interpolant": interpolant,
        "reduction": "none",
    }
    loss_id = regression_loss(field, parameterization=p_id, **common)
    loss_neg = regression_loss(field, parameterization=p_neg, **common)

    # field emits 0 → p_id has prediction=0; p_neg has prediction=-0=0 too.
    # That's degenerate.  Use a non-zero field to make the test load-bearing:
    nz_field = _ConsistentEpsField(torch.ones_like(x_data)).to(dtype=torch.float64)
    loss_id_nz = regression_loss(nz_field, parameterization=p_id, **common)
    loss_neg_nz = regression_loss(nz_field, parameterization=p_neg, **common)
    # prediction = 1 vs prediction = -1 against the same target should differ
    assert not torch.allclose(loss_id_nz, loss_neg_nz), (
        "output_transform appears to be ignored"
    )
    # And the zero-field case is correctly equal because ±0 are the same.
    assert torch.allclose(loss_id, loss_neg)
