from __future__ import annotations

import pytest
import torch

import nami
from nami import (
    BrownianBridgeInterpolant,
    GeneratorField,
    ItoGeneratorOperator,
    KLDivergence,
    LinearInterpolant,
    Parameterization,
    SquaredL2,
    Velocity,
    cgm_loss,
    generator_prediction,
    regression_loss,
)


def _field(dim, op):
    torch.manual_seed(0)
    return GeneratorField(dim, operator=op, hidden=16, layers=2)


def test_cgm_drift_only_matches_regression_loss():
    """Single-component (ODE) Itô generator: the default CGM loss is exactly the
    MSE regression loss."""
    op = ItoGeneratorOperator((4,), diffusion="none")
    field = _field(4, op)
    interp = LinearInterpolant()
    param = generator_prediction(op)
    xn, xd, t = torch.randn(32, 4), torch.randn(32, 4), torch.rand(32)

    reg = regression_loss(
        field, x_noise=xn, x_data=xd, t=t, interpolant=interp, parameterization=param
    )
    cgm = cgm_loss(
        field, x_noise=xn, x_data=xd, t=t, interpolant=interp, parameterization=param
    )
    assert torch.allclose(reg, cgm, atol=1e-6)


def test_cgm_diffusion_sums_per_component():
    """With drift + diffusion, the CGM loss sums the two equal-size per-component
    MSEs, which is exactly twice ``regression_loss``'s single flat MSE over the
    packed (2, d) tensor (shared interpolant noise)."""
    op = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = _field(4, op)
    interp = BrownianBridgeInterpolant(sigma=0.3)
    param = generator_prediction(op)
    xn, xd, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    z = torch.randn(16, 4)

    reg = regression_loss(
        field,
        x_noise=xn,
        x_data=xd,
        t=t,
        interpolant=interp,
        parameterization=param,
        z=z,
    )
    cgm = cgm_loss(
        field,
        x_noise=xn,
        x_data=xd,
        t=t,
        interpolant=interp,
        parameterization=param,
        z=z,
    )
    assert torch.allclose(cgm, 2.0 * reg, rtol=1e-5)


def test_cgm_rejects_non_generator_params_target():
    field = _field(4, ItoGeneratorOperator((4,)))
    with pytest.raises(TypeError, match="GeneratorParams"):
        cgm_loss(
            field,
            x_noise=torch.randn(8, 4),
            x_data=torch.randn(8, 4),
            interpolant=LinearInterpolant(),
            parameterization=Parameterization(target=Velocity()),
        )


def test_cgm_per_component_divergence_mapping():
    """A dict ``divergence`` lets each component carry its own Bregman; here we
    zero out the diffusion arm by mapping it to a divergence of itself."""
    op = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = _field(4, op)
    interp = BrownianBridgeInterpolant(sigma=0.3)
    param = generator_prediction(op)
    xn, xd, t = torch.randn(16, 4), torch.randn(16, 4), torch.rand(16)
    z = torch.randn(16, 4)

    full = cgm_loss(
        field,
        x_noise=xn,
        x_data=xd,
        t=t,
        interpolant=interp,
        parameterization=param,
        z=z,
        divergence={"drift": SquaredL2(), "diffusion": SquaredL2()},
    )
    default = cgm_loss(
        field,
        x_noise=xn,
        x_data=xd,
        t=t,
        interpolant=interp,
        parameterization=param,
        z=z,
    )
    assert torch.allclose(full, default, atol=1e-6)


def test_cgm_dict_divergence_missing_component_raises():
    op = ItoGeneratorOperator((4,), diffusion="diagonal")
    field = _field(4, op)
    with pytest.raises(KeyError):
        cgm_loss(
            field,
            x_noise=torch.randn(8, 4),
            x_data=torch.randn(8, 4),
            interpolant=BrownianBridgeInterpolant(sigma=0.3),
            parameterization=generator_prediction(op),
            divergence={"drift": SquaredL2()},  # missing "diffusion"
        )


def test_cgm_loss_is_differentiable():
    op = ItoGeneratorOperator((3,), diffusion="diagonal")
    field = _field(3, op)
    loss = cgm_loss(
        field,
        x_noise=torch.randn(8, 3),
        x_data=torch.randn(8, 3),
        interpolant=BrownianBridgeInterpolant(sigma=0.2),
        parameterization=generator_prediction(op),
    )
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cgm_reductions(reduction):
    op = ItoGeneratorOperator((3,))
    field = _field(3, op)
    out = cgm_loss(
        field,
        x_noise=torch.randn(8, 3),
        x_data=torch.randn(8, 3),
        interpolant=LinearInterpolant(),
        parameterization=generator_prediction(op),
        reduction=reduction,
    )
    if reduction == "none":
        assert out.shape == (8,)
    else:
        assert out.ndim == 0


def test_cgm_loss_exported_at_top_level():
    assert nami.cgm_loss is cgm_loss
    assert hasattr(nami, "KLDivergence")
    assert nami.KLDivergence is KLDivergence
