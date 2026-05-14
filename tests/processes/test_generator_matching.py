from __future__ import annotations

import math

import pytest
import torch

from nami import (
    RK4,
    EulerMaruyama,
    GeneratorMatching,
    Heun,
    ItoGeneratorOperator,
    Parameterization,
    Velocity,
    generator_prediction,
)
from nami.fields.generator import GeneratorField
from nami.generators.base import GeneratorOperator


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class _ZeroDrift(torch.nn.Module):
    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t, c
        return torch.zeros_like(x)


class _ConstantDiagonalDiffusion(torch.nn.Module):
    event_ndim = 1

    def __init__(self, scale: float):
        super().__init__()
        self.raw_scale = float(_inv_softplus(scale))

    def forward(self, x, t, c=None):
        _ = t, c
        drift = torch.zeros_like(x)
        diffusion = torch.full_like(x, self.raw_scale)
        return torch.stack((drift, diffusion), dim=-2)


class _ContextDrift(torch.nn.Module):
    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return torch.zeros_like(x)
        return torch.zeros_like(x) + c[..., :1].expand_as(x)


class _JumpOperator(GeneratorOperator):
    def __init__(self):
        super().__init__(runtime_kind="jump")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (2,)

    def pack_params(self, *, drift, diffusion=None):
        _ = diffusion
        return drift

    def drift(self, x, t, params):
        _ = x, t
        return params

    def diffusion(self, x, t, params):
        _ = x, t, params
        raise NotImplementedError


def test_generator_matching_ode_smoke():
    operator = ItoGeneratorOperator((3,))
    process = GeneratorMatching(
        _ZeroDrift(),
        RK4(steps=4),
        parameterization=generator_prediction(operator),
        event_shape=(3,),
    )()

    sample = process.sample(sample_shape=(5,))

    assert sample.shape == (5, 3)
    assert torch.isfinite(sample).all()


def test_generator_matching_sde_smoke():
    operator = ItoGeneratorOperator((2,), diffusion="diagonal")
    process = GeneratorMatching(
        _ConstantDiagonalDiffusion(scale=0.2),
        EulerMaruyama(steps=4),
        parameterization=generator_prediction(operator),
        event_shape=(2,),
    )()

    sample = process.sample(sample_shape=(6,))

    assert sample.shape == (6, 2)
    assert torch.isfinite(sample).all()


def test_generator_matching_context_expands_over_samples():
    operator = ItoGeneratorOperator((2,))
    context = torch.randn(3, 1)
    process = GeneratorMatching(
        _ContextDrift(),
        Heun(steps=2),
        parameterization=generator_prediction(operator),
        event_shape=(2,),
    )(context)

    sample = process.sample(sample_shape=(4,))

    assert sample.shape == (4, 3, 2)


def test_generator_matching_jump_runtime_requires_simulator():
    process = GeneratorMatching(
        _ZeroDrift(),
        RK4(steps=2),
        parameterization=generator_prediction(_JumpOperator()),
        event_shape=(2,),
        validate_args=False,
    )()

    with pytest.raises(
        NotImplementedError, match="jump runtime requires a compatible simulator"
    ):
        process.sample(sample_shape=(3,))


def test_generator_matching_rsample_rejects_sde():
    operator = ItoGeneratorOperator((2,), diffusion="diagonal")
    process = GeneratorMatching(
        _ConstantDiagonalDiffusion(scale=0.2),
        EulerMaruyama(steps=4),
        parameterization=generator_prediction(operator),
        event_shape=(2,),
    )()

    with pytest.raises(
        NotImplementedError, match="rsample is supported only for ODE generators"
    ):
        process.rsample(sample_shape=(3,))


def test_generator_matching_validates_operator_shape():
    operator = ItoGeneratorOperator((2,))

    with pytest.raises(ValueError, match=r"operator\.event_shape does not match"):
        GeneratorMatching(
            _ZeroDrift(),
            RK4(steps=2),
            parameterization=generator_prediction(operator),
            event_shape=(3,),
        )()


def test_generator_matching_rejects_legacy_operator_kwarg():
    """Pins the API break for stage 3c: passing the legacy ``operator``
    positional fails clearly rather than silently consuming it as
    ``solver``.
    """
    operator = ItoGeneratorOperator((2,))

    with pytest.raises(TypeError):
        # Old-style positional call: (field, operator, solver).
        # New signature: (field, solver, *, parameterization=...).
        # The middle positional now binds to `solver`, but
        # parameterization is missing, so construction fails at __init__.
        GeneratorMatching(
            _ZeroDrift(),
            operator,
            RK4(steps=2),
            event_shape=(2,),
        )


def test_generator_matching_rejects_non_generator_params_target():
    """The Process supports only the GeneratorParams target.

    Pins that a future caller passing e.g. ``Velocity`` here gets a
    clear migration message — pattern-match fallthrough errors would
    appear too late, deep inside _projected_params.
    """
    with pytest.raises(TypeError, match="GeneratorParams"):
        GeneratorMatching(
            _ZeroDrift(),
            RK4(steps=2),
            parameterization=Parameterization(target=Velocity()),
            event_shape=(2,),
        )()


def test_generator_field_matches_operator_shape():
    operator = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = GeneratorField(
        (4,), operator=operator, condition_dim=2, hidden=32, layers=2
    )
    x = torch.randn(5, 4)
    t = torch.rand(5)
    c = torch.randn(5, 2)

    params = field(x, t, c)

    assert params.shape == (5, 2, 4)


# ---------------------------------------------------------------------------
# Stage-3c headline: sample equivalence between custom output_transform paths
# ---------------------------------------------------------------------------


class _RawDriftDiffusion(torch.nn.Module):
    """Field that emits raw (drift, raw_diffusion) — exercises projection."""

    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t, c
        drift = -0.2 * x
        raw_diffusion = torch.full_like(x, _inv_softplus(0.3))
        return torch.stack((drift, raw_diffusion), dim=-2)


def test_default_projection_path_is_op_project():
    """``generator_prediction(op)``'s ``output_transform`` is ``op.project``,
    so the unified Process produces identical samples to one constructed
    with a hand-rolled ``Parameterization`` whose ``output_transform``
    explicitly calls ``op.project``.

    The point: the factory is an honest synonym for the underlying
    composition, not a different code path.
    """
    operator = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = _RawDriftDiffusion()

    factory_p = generator_prediction(operator)
    explicit_p = Parameterization(
        target=factory_p.target,
        weighting=factory_p.weighting,
        output_transform=operator.project,
    )

    torch.manual_seed(0)
    s_factory = GeneratorMatching(
        field,
        EulerMaruyama(steps=8),
        parameterization=factory_p,
        event_shape=(4,),
    )().sample(sample_shape=(8,))

    torch.manual_seed(0)
    s_explicit = GeneratorMatching(
        field,
        EulerMaruyama(steps=8),
        parameterization=explicit_p,
        event_shape=(4,),
    )().sample(sample_shape=(8,))

    assert torch.allclose(s_factory, s_explicit, atol=1e-12, rtol=1e-12)


def test_custom_output_transform_changes_samples():
    """A non-``op.project`` ``output_transform`` (here, op.project then
    a sign flip on diffusion) measurably changes samples — pins that
    the Process actually consults
    ``parameterization.output_transform`` rather than calling
    ``op.project`` directly behind it.
    """
    operator = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = _RawDriftDiffusion()

    def flipped_diffusion(raw):
        # Apply normal projection, then reduce diffusion magnitude
        # by half (still positive, still valid for the operator).
        params = operator.project(raw)
        drift, diff = torch.unbind(params, dim=-2)
        return torch.stack((drift, 0.5 * diff), dim=-2)

    custom = Parameterization(
        target=generator_prediction(operator).target,
        output_transform=flipped_diffusion,
    )

    torch.manual_seed(0)
    s_default = GeneratorMatching(
        field,
        EulerMaruyama(steps=8),
        parameterization=generator_prediction(operator),
        event_shape=(4,),
    )().sample(sample_shape=(8,))

    torch.manual_seed(0)
    s_custom = GeneratorMatching(
        field,
        EulerMaruyama(steps=8),
        parameterization=custom,
        event_shape=(4,),
    )().sample(sample_shape=(8,))

    assert not torch.allclose(s_default, s_custom), (
        "output_transform appears not to be applied — the Process is "
        "calling op.project directly behind the parameterization"
    )
