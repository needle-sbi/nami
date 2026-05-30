from __future__ import annotations

import pytest
import torch

from nami.diffusion import eps_to_x0, score_to_x0, x0_to_score
from nami.divergence.base import DivergenceEstimator
from nami.divergence.exact import ExactDivergence
from nami.divergence.hutchinson import HutchinsonDivergence
from nami.fields.base import VectorField
from nami.generators.base import GeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
from nami.lazy import LazyDistribution, LazyField, LazyProcess, UnconditionalField
from nami.schedules.base import NoiseSchedule


def test_lazy_base_classes_raise_for_unimplemented_forward() -> None:
    with pytest.raises(NotImplementedError):
        LazyDistribution()(None)
    with pytest.raises(NotImplementedError):
        LazyProcess()(None)
    with pytest.raises(NotImplementedError):
        LazyField()(None)


def test_lazy_field_default_event_ndim_and_unconditional_adapter() -> None:
    class _Field(torch.nn.Module):
        event_ndim = 2

        def forward(self, x, t, c=None):
            _ = t, c
            return x

    field = _Field()
    wrapped = UnconditionalField(field)

    assert LazyField().event_ndim is None
    assert wrapped.event_ndim == 2
    assert wrapped(torch.randn(1)) is field


def test_noise_schedule_base_methods_raise() -> None:
    schedule = NoiseSchedule()
    t = torch.ones(2)
    x = torch.ones(2, 3)

    with pytest.raises(NotImplementedError):
        schedule.alpha(t)
    with pytest.raises(NotImplementedError):
        schedule.sigma(t)
    with pytest.raises(NotImplementedError):
        schedule.drift(x, t)
    with pytest.raises(NotImplementedError):
        schedule.diffusion(t)


def test_vector_field_and_divergence_base_methods_raise() -> None:
    field = VectorField()
    x = torch.ones(2, 3)
    t = torch.ones(2)

    with pytest.raises(NotImplementedError):
        _ = field.event_ndim
    with pytest.raises(NotImplementedError):
        field(x, t)
    with pytest.raises(NotImplementedError):
        field.call_and_divergence(x, t)
    with pytest.raises(NotImplementedError):
        DivergenceEstimator()(field, x, t, None)


def test_divergence_estimators_validate_field_contracts() -> None:
    class _IdentityField(torch.nn.Identity):
        event_ndim = 1

    field = torch.nn.Identity()
    x = torch.ones(2, 3)
    t = torch.ones(2)

    with pytest.raises(ValueError, match="event_ndim"):
        ExactDivergence()(field, x, t, None)
    with pytest.raises(ValueError, match="max_dim"):
        ExactDivergence(max_dim=2)(_IdentityField(), x, t, None)
    with pytest.raises(ValueError, match="event_ndim"):
        HutchinsonDivergence()(field, x, t, None)
    with pytest.raises(ValueError, match="probe"):
        HutchinsonDivergence(probe="bad")


def test_diffusion_clean_endpoint_conversions_are_inverse() -> None:
    x0 = torch.tensor([[0.5, -1.0], [1.5, 0.25]])
    eps = torch.tensor([[0.1, 0.3], [-0.2, 0.4]])
    alpha = torch.tensor([0.8, 0.6])
    sigma = torch.tensor([0.2, 0.5])
    x = alpha.unsqueeze(-1) * x0 + sigma.unsqueeze(-1) * eps

    assert torch.allclose(eps_to_x0(x, eps, alpha, sigma), x0)
    score = x0_to_score(x, x0, alpha, sigma)
    assert torch.allclose(score_to_x0(x, score, alpha, sigma), x0)


def test_generator_operator_base_contract_and_ito_validation() -> None:
    with pytest.raises(ValueError, match="runtime_kind"):
        GeneratorOperator(runtime_kind="bad")

    op = GeneratorOperator(runtime_kind="ode")
    x = torch.ones(2, 3)
    t = torch.ones(2)

    assert op.runtime_kind == "ode"
    with pytest.raises(NotImplementedError):
        _ = op.event_shape
    with pytest.raises(NotImplementedError):
        _ = op.parameter_shape
    with pytest.raises(NotImplementedError):
        op.pack_params(drift=x)
    with pytest.raises(NotImplementedError):
        op.drift(x, t, x)
    with pytest.raises(NotImplementedError):
        op.diffusion(x, t, x)

    ito = ItoGeneratorOperator((3,))
    with pytest.raises(ValueError, match="not used"):
        ito.pack_params(drift=x, diffusion=x)
    with pytest.raises(NotImplementedError, match="unavailable"):
        ito.diffusion(x, t, x)

    ito_sde = ItoGeneratorOperator((3,), diffusion="diagonal")
    with pytest.raises(ValueError, match="required"):
        ito_sde.pack_params(drift=x)
