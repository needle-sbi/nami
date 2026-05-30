"""LinearInterpolant + analytic linear-path velocity-matching baseline."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.generators.operators import ItoGeneratorOperator
from nami.interpolants import (
    InterpolantState,
    LinearInterpolant,
    StochasticLinearInterpolant,
    velocity_prediction,
)
from nami.losses import regression_loss
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Velocity,
    VPrediction,
)


@pytest.fixture
def interpolant() -> LinearInterpolant:
    return LinearInterpolant()


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


def test_sample_matches_linear_formula(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    expected = (1.0 - t.unsqueeze(-1)) * x_noise + t.unsqueeze(-1) * x_data
    assert torch.allclose(state.xt, expected, atol=1e-6)


def test_sample_endpoints(interpolant: LinearInterpolant) -> None:
    x_data = torch.randn(4, 2)
    x_noise = torch.randn(4, 2)
    state0 = interpolant.sample(x_noise, x_data, torch.zeros(4))
    state1 = interpolant.sample(x_noise, x_data, torch.ones(4))
    assert torch.allclose(state0.xt, x_noise, atol=1e-6)
    assert torch.allclose(state1.xt, x_data, atol=1e-6)


def test_sample_state_has_no_noise(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    assert state.noise is None
    assert isinstance(state, InterpolantState)


def test_sample_rejects_external_noise(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    with pytest.raises(ValueError, match="deterministic"):
        interpolant.sample(x_noise, x_data, t, noise=torch.randn_like(x_noise))


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_target_velocity_is_constant_difference(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """u_t = x_data - x_noise — independent of x_t and t."""
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    velocity = interpolant.target(Velocity(), state)
    assert torch.equal(velocity, x_data - x_noise)


@pytest.mark.parametrize(
    "target_cls",
    [Score, Epsilon, X0],
    ids=["score", "epsilon", "x0"],
)
def test_stochastic_targets_are_unsupported(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_cls: type,
) -> None:
    """LinearInterpolant deliberately rejects stochastic targets.

    The conditional density is a delta function; score / eps / x0
    targets have no clean meaning here.  Pinned by test so the
    decision is explicit and a future "convenience" implementation
    can't sneak in.
    """
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)
    with pytest.raises(NotImplementedError, match="deterministic"):
        interpolant.target(target_cls(), state)


def test_velocity_prediction_factory_uses_unit_weighting() -> None:
    p = velocity_prediction()
    assert isinstance(p.target, Velocity)
    t = torch.linspace(0.0, 1.0, 16)
    assert torch.equal(p.weighting(t), torch.ones_like(t))


# ---------------------------------------------------------------------------
# Headline: analytic linear-path velocity-matching baseline
# ---------------------------------------------------------------------------


def _analytic_linear_velocity_loss(
    field: nn.Module,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """Reference implementation of linear-path FM loss.

    Computed from first principles: the conditional velocity for the
    linear path ``x_t = (1-t) x_noise + t x_data`` is the constant
    ``u_t = x_data - x_noise``, the FM weighting is ω(t)=1, and
    per-sample MSE is the mean over event dimensions.  This is the
    formula the deleted ``fm_loss`` encoded; ``regression_loss`` with
    ``LinearInterpolant + Velocity`` must reproduce it.
    """
    tt = t.reshape(t.shape + (1,) * (x_data.ndim - t.ndim))
    xt = (1.0 - tt) * x_noise + tt * x_data
    ut = x_data - x_noise
    vt = field(xt, t)
    per_sample = (vt - ut).pow(2).reshape(*t.shape, -1).mean(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    msg = f"unknown reduction: {reduction}"
    raise ValueError(msg)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_regression_loss_matches_analytic_linear_path_formula(
    interpolant: LinearInterpolant, reduction: str
) -> None:
    """``regression_loss(LinearInterpolant, Velocity)`` reproduces the
    closed-form linear-path velocity-matching loss bit-exactly.

    The analytic baseline is independent of any other ``nami`` code,
    so this test pins what ``regression_loss`` *means* for this
    interpolant: not "what some other function in the library
    returns", but the mathematical formula directly.  If a future
    refactor of the loss primitives changes the answer, the divergence
    is from the formula itself, not from a stale reference.
    """

    class _Field(nn.Module):
        event_ndim = 1

        def forward(self, x, t, c=None):  # noqa: ARG002
            # Non-trivial but deterministic to keep the loss non-zero.
            return 0.5 * x + t.unsqueeze(-1) * 0.1

    torch.manual_seed(0)
    x_data = torch.randn(32, 4, dtype=torch.float64)
    x_noise = torch.randn(32, 4, dtype=torch.float64)
    t = torch.rand(32, dtype=torch.float64)
    field = _Field().to(dtype=torch.float64)

    expected = _analytic_linear_velocity_loss(
        field, x_data, x_noise, t, reduction=reduction
    )
    actual = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=velocity_prediction(),
        reduction=reduction,
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12), (
        "regression_loss has drifted from the analytic linear-path "
        "velocity-matching formula"
    )


def test_bare_parameterization_works_too(
    interpolant: LinearInterpolant,
) -> None:
    """Users can pass ``Parameterization(target=Velocity())`` directly.

    The factory is a convenience; the bare form (with default unit
    weighting and identity output_transform) is equivalent and pinned
    here so future factory changes don't accidentally silently change
    the bare-form semantics.
    """

    class _Field(nn.Module):
        event_ndim = 1

        def forward(self, x, t, c=None):  # noqa: ARG002
            return torch.zeros_like(x)

    torch.manual_seed(2)
    x_data = torch.randn(8, 2)
    x_noise = torch.randn(8, 2)
    t = torch.rand(8)
    field = _Field()

    factory = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=velocity_prediction(),
        reduction="none",
    )
    bare = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=Parameterization(target=Velocity()),
        reduction="none",
    )
    assert torch.allclose(factory, bare)


def test_linear_target_action_matches_velocity(
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = interpolant.sample(x_noise, x_data, t)

    assert torch.equal(interpolant.target(Action(), state), x_data - x_noise)


@pytest.mark.parametrize("diffusion", ["none", "diagonal"])
def test_linear_target_generator_params_packs_drift_and_zero_diffusion(
    diffusion: str,
    interpolant: LinearInterpolant,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    operator = ItoGeneratorOperator((3,), diffusion=diffusion)
    state = interpolant.sample(x_noise, x_data, t)

    params = interpolant.target(GeneratorParams(operator=operator), state)

    drift = operator.drift(state.xt, t, params)
    assert torch.equal(drift, x_data - x_noise)
    if diffusion == "diagonal":
        assert torch.equal(
            operator.diffusion(state.xt, t, params), torch.zeros_like(x_data)
        )


def test_stochastic_linear_sample_draws_noise_when_not_provided(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    interpolant = StochasticLinearInterpolant()

    state = interpolant.sample(x_noise, x_data, t)

    assert state.noise is not None
    assert state.noise.shape == x_data.shape
    assert state.xt.shape == x_data.shape


@pytest.mark.parametrize("target", [Velocity(), Action()], ids=["velocity", "action"])
def test_stochastic_linear_velocity_like_targets_require_noise(target) -> None:
    interpolant = StochasticLinearInterpolant()
    state = InterpolantState(
        xt=torch.zeros(2, 3),
        x_data=torch.ones(2, 3),
        x_noise=torch.zeros(2, 3),
        t=torch.full((2,), 0.5),
        noise=None,
    )

    with pytest.raises(ValueError, match=r"requires .*noise"):
        interpolant.target(target, state)


def test_stochastic_linear_action_matches_velocity_with_shared_noise(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    noise = torch.randn_like(x_data)
    interpolant = StochasticLinearInterpolant()
    state = interpolant.sample(x_noise, x_data, t, noise=noise)

    assert torch.allclose(
        interpolant.target(Action(), state),
        interpolant.target(Velocity(), state),
    )


@pytest.mark.parametrize(
    "target",
    [
        Score(),
        Epsilon(),
        X0(),
        VPrediction(),
        GeneratorParams(ItoGeneratorOperator((3,))),
    ],
    ids=["score", "epsilon", "x0", "v", "generator"],
)
def test_stochastic_linear_rejects_unsupported_targets(
    target,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    interpolant = StochasticLinearInterpolant()
    state = interpolant.sample(x_noise, x_data, t, noise=torch.randn_like(x_data))

    with pytest.raises(NotImplementedError):
        interpolant.target(target, state)
