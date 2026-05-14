from __future__ import annotations

import pytest
import torch

from nami import (
    RK4,
    FlowMatching,
    LinearInterpolant,
    StandardNormal,
    TransformerVelocityField,
    regression_loss,
    velocity_prediction,
)


def test_transformer_velocity_field_supports_tuple_event_shape():
    field = TransformerVelocityField((2, 3), model_dim=16, depth=2, num_heads=4)
    x = torch.randn(4, 5, 2, 3)
    t = torch.rand(4, 5)

    out = field(x, t)

    assert out.shape == x.shape
    assert field.event_ndim == 2


def test_transformer_velocity_field_requires_context_when_conditional():
    field = TransformerVelocityField(
        6,
        model_dim=16,
        depth=2,
        num_heads=4,
        condition_dim=3,
    )
    x = torch.randn(4, 6)
    t = torch.rand(4)

    with pytest.raises(ValueError, match="conditioning input"):
        field(x, t)


def test_transformer_velocity_field_rejects_context_when_unconditional():
    field = TransformerVelocityField(6, model_dim=16, depth=2, num_heads=4)
    x = torch.randn(4, 6)
    t = torch.rand(4)
    c = torch.randn(4, 3)

    with pytest.raises(ValueError, match="condition_dim is 0"):
        field(x, t, c)


def test_transformer_velocity_field_validates_context_shape():
    field = TransformerVelocityField(
        6,
        model_dim=16,
        depth=2,
        num_heads=4,
        condition_dim=3,
    )
    x = torch.randn(2, 4, 6)
    t = torch.rand(2, 4)
    c = torch.randn(4, 3)

    with pytest.raises(ValueError, match="context shape mismatch"):
        field(x, t, c)


def test_transformer_velocity_field_supports_valid_context():
    field = TransformerVelocityField(
        6,
        model_dim=16,
        depth=2,
        num_heads=4,
        condition_dim=3,
    )
    x = torch.randn(4, 6)
    t = torch.rand(4)
    c = torch.randn(4, 3)

    out = field(x, t, c)

    assert out.shape == x.shape


def test_transformer_velocity_field_supports_flow_matching_process():
    field = TransformerVelocityField(8, model_dim=16, depth=2, num_heads=4)
    loss = regression_loss(
        field,
        x_data=torch.randn(16, 8),
        x_noise=torch.randn(16, 8),
        interpolant=LinearInterpolant(),
        parameterization=velocity_prediction(),
        eps_t=0.0,
    )

    process = FlowMatching(field, StandardNormal(8), RK4(steps=4))()
    samples = process.sample((5,))

    assert loss.ndim == 0
    assert samples.shape == (5, 8)
