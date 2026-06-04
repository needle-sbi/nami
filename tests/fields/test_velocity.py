from __future__ import annotations

import pytest
import torch

from nami import VelocityField
from nami.core.specs import TensorSpec


def test_velocity_field_supports_tuple_event_shape():
    field = VelocityField((2, 3), hidden=16, layers=1)
    x = torch.randn(4, 5, 2, 3)
    t = torch.rand(4, 5)

    out = field(x, t)

    assert out.shape == x.shape
    assert field.event_ndim == 2


def test_velocity_field_layer_norm_supports_multiple_leading_dims():
    field = VelocityField(3, hidden=16, layers=2, layer_norm=True)
    x = torch.randn(2, 4, 5, 3)
    t = torch.rand(2, 4, 5)

    out = field(x, t)

    assert out.shape == x.shape


def test_velocity_field_rejects_context_when_unconditional():
    field = VelocityField(3, hidden=16, layers=1)
    x = torch.randn(4, 3)
    t = torch.rand(4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="condition_dim is 0"):
        field(x, t, c)


def test_velocity_field_requires_context_when_conditional():
    field = VelocityField(3, condition_dim=2, hidden=16, layers=1)
    x = torch.randn(4, 3)
    t = torch.rand(4)

    with pytest.raises(ValueError, match="conditioning input"):
        field(x, t)


def test_velocity_field_validates_context_dimension():
    field = VelocityField((2, 2), condition_dim=3, hidden=16, layers=1)
    x = torch.randn(4, 2, 2)
    t = torch.rand(4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="context dimension mismatch"):
        field(x, t, c)


def test_velocity_field_supports_valid_context():
    field = VelocityField((2, 2), condition_dim=3, hidden=16, layers=1)
    x = torch.randn(4, 2, 2)
    t = torch.rand(4)
    c = torch.randn(4, 3)

    out = field(x, t, c)

    assert out.shape == x.shape


def test_velocity_field_validates_context_leading_shape():
    field = VelocityField(3, condition_dim=2, hidden=16, layers=1)
    x = torch.randn(2, 4, 3)
    t = torch.rand(2, 4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="context shape mismatch"):
        field(x, t, c)


def test_velocity_field_rejects_negative_condition_dim():
    with pytest.raises(ValueError, match="condition_dim must be non-negative"):
        VelocityField(3, condition_dim=-1, hidden=16, layers=1)


def test_velocity_field_exposes_tensor_spec():
    # The spec is the single source of shape truth; the legacy attributes
    # must stay consistent with it.
    field = VelocityField((2, 3), hidden=16, layers=1)

    assert isinstance(field.spec, TensorSpec)
    assert field.spec.event_shape == (2, 3)
    assert field.event_shape == (2, 3)
    assert field.event_ndim == field.spec.event_ndim == 2
    assert field.flat_dim == field.spec.numel == 6
