from __future__ import annotations

import math

import pytest
import torch

from nami import CosineInterpolant, GeneratorParams, ItoGeneratorOperator
from nami.parameterizations import X0, Action, Epsilon, Score, Velocity


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(6, 3), torch.randn(6, 3), torch.rand(6)


def test_cosine_sample_matches_closed_form(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    interpolant = CosineInterpolant()

    state = interpolant.sample(x_noise, x_data, t)

    a = torch.cos(t * math.pi / 2).unsqueeze(-1)
    s = torch.sin(t * math.pi / 2).unsqueeze(-1)
    assert torch.allclose(state.xt, a * x_noise + s * x_data)
    assert state.noise is None


def test_cosine_sample_rejects_external_noise(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch

    with pytest.raises(ValueError, match="deterministic"):
        CosineInterpolant().sample(x_noise, x_data, t, noise=torch.randn_like(x_data))


@pytest.mark.parametrize("target", [Velocity(), Action()], ids=["velocity", "action"])
def test_cosine_velocity_like_targets_match_derivative(
    target,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    interpolant = CosineInterpolant()
    state = interpolant.sample(x_noise, x_data, t)

    actual = interpolant.target(target, state)

    ap = (-(math.pi / 2) * torch.sin(t * math.pi / 2)).unsqueeze(-1)
    sp = ((math.pi / 2) * torch.cos(t * math.pi / 2)).unsqueeze(-1)
    assert torch.allclose(actual, ap * x_noise + sp * x_data)


@pytest.mark.parametrize(
    "target",
    [
        Score(),
        Epsilon(),
        X0(),
        GeneratorParams(ItoGeneratorOperator((3,))),
    ],
    ids=["score", "epsilon", "x0", "generator"],
)
def test_cosine_rejects_unsupported_targets(
    target,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    x_data, x_noise, t = batch
    state = CosineInterpolant().sample(x_noise, x_data, t)

    with pytest.raises(NotImplementedError):
        CosineInterpolant().target(target, state)
