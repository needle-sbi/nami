from __future__ import annotations

import pytest
import torch

from nami.fields import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
    TwoHeadField,
)
from nami.interpolants.gamma import BrownianGamma


class _ConstantField(torch.nn.Module):
    event_ndim = 1

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x, t, c=None):  # noqa: ARG002
        return torch.full_like(x, self.value)


def test_drift_from_velocity_score_matches_formula() -> None:
    velocity = _ConstantField(2.0)
    score = _ConstantField(3.0)
    gamma = BrownianGamma()
    field = DriftFromVelocityScore(velocity, score, gamma)
    x = torch.zeros(5, 4)
    t = torch.linspace(0.1, 0.9, 5)

    actual = field(x, t)
    coeff = gamma.gamma_gamma_dot(t).unsqueeze(-1)
    expected = torch.full_like(x, 2.0) - coeff * torch.full_like(x, 3.0)

    assert isinstance(field, TwoHeadField)
    assert torch.allclose(actual, expected)


def test_markovization_drift_from_velocity_score_matches_formula() -> None:
    velocity = _ConstantField(2.0)
    score = _ConstantField(3.0)
    gamma = BrownianGamma()
    field = MarkovizationDriftFromVelocityScore(
        velocity,
        score,
        gamma,
        diffusion2=0.4,
    )
    x = torch.zeros(5, 4)
    t = torch.linspace(0.1, 0.9, 5)

    actual = field(x, t)
    coeff = (-gamma.gamma_gamma_dot(t) + 0.5 * 0.4).unsqueeze(-1)
    expected = torch.full_like(x, 2.0) + coeff * torch.full_like(x, 3.0)

    assert isinstance(field, TwoHeadField)
    assert torch.allclose(actual, expected)


def test_markovization_accepts_callable_diffusion2() -> None:
    velocity = _ConstantField(0.0)
    score = _ConstantField(1.0)
    gamma = BrownianGamma()
    field = MarkovizationDriftFromVelocityScore(
        velocity,
        score,
        gamma,
        diffusion2=lambda t: 2.0 * t,
    )
    x = torch.zeros(3, 2)
    t = torch.tensor([0.2, 0.4, 0.6])

    actual = field(x, t)
    coeff = (-gamma.gamma_gamma_dot(t) + t).unsqueeze(-1)
    assert torch.allclose(actual, coeff.expand_as(x))


def test_composite_fields_expose_velocity_head_event_ndim() -> None:
    """Both composites report the velocity head's event_ndim for shape checks."""
    velocity = _ConstantField(0.0)
    score = _ConstantField(0.0)
    gamma = BrownianGamma()

    drift = DriftFromVelocityScore(velocity, score, gamma)
    markov = MarkovizationDriftFromVelocityScore(velocity, score, gamma, diffusion2=1.0)

    assert drift.event_ndim == 1
    assert markov.event_ndim == 1


def test_two_head_field_rejects_event_ndim_mismatch() -> None:
    velocity = _ConstantField(0.0)
    score = _ConstantField(0.0)
    score.event_ndim = 2

    with pytest.raises(ValueError, match="event_ndim"):
        DriftFromVelocityScore(velocity, score, BrownianGamma())
