from __future__ import annotations

import pytest
import torch

from nami import AdaLNVelocityField


def test_adaln_velocity_field_unconditional_forward_shape():
    field = AdaLNVelocityField(3, hidden=16, layers=2, time_dim=8, cond_hidden=8)
    x = torch.randn(4, 5, 3)
    t = torch.rand(4, 5)

    out = field(x, t)

    assert out.shape == x.shape
    assert field.event_ndim == 1


def test_adaln_velocity_field_conditional_forward_shape():
    field = AdaLNVelocityField(
        3, condition_dim=2, hidden=16, layers=2, time_dim=8, cond_hidden=8
    )
    x = torch.randn(4, 5, 3)
    t = torch.rand(4, 5)
    c = torch.randn(4, 5, 2)

    out = field(x, t, c)

    assert out.shape == x.shape


def test_adaln_velocity_field_supports_tuple_event_shape():
    field = AdaLNVelocityField((2, 3), hidden=16, layers=1, time_dim=8, cond_hidden=8)
    x = torch.randn(4, 5, 2, 3)
    t = torch.rand(4, 5)

    out = field(x, t)

    assert out.shape == x.shape
    assert field.event_ndim == 2


def test_adaln_velocity_field_rejects_context_when_unconditional():
    field = AdaLNVelocityField(3, hidden=16, layers=1, time_dim=8, cond_hidden=8)
    x = torch.randn(4, 3)
    t = torch.rand(4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="condition_dim is 0"):
        field(x, t, c)


def test_adaln_velocity_field_requires_context_when_conditional():
    field = AdaLNVelocityField(
        3, condition_dim=2, hidden=16, layers=1, time_dim=8, cond_hidden=8
    )
    x = torch.randn(4, 3)
    t = torch.rand(4)

    with pytest.raises(ValueError, match="conditioning input"):
        field(x, t)


def test_adaln_velocity_field_validates_context_dimension():
    field = AdaLNVelocityField(
        3, condition_dim=3, hidden=16, layers=1, time_dim=8, cond_hidden=8
    )
    x = torch.randn(4, 3)
    t = torch.rand(4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="context dimension mismatch"):
        field(x, t, c)


def test_adaln_velocity_field_validates_context_leading_shape():
    field = AdaLNVelocityField(
        3, condition_dim=2, hidden=16, layers=1, time_dim=8, cond_hidden=8
    )
    x = torch.randn(2, 4, 3)
    t = torch.rand(2, 4)
    c = torch.randn(4, 2)

    with pytest.raises(ValueError, match="context shape mismatch"):
        field(x, t, c)


def test_adaln_velocity_field_rejects_negative_condition_dim():
    with pytest.raises(ValueError, match="condition_dim must be non-negative"):
        AdaLNVelocityField(3, condition_dim=-1)


def test_adaln_velocity_field_rejects_unknown_activation():
    with pytest.raises(ValueError, match="unknown activation"):
        AdaLNVelocityField(3, activation="banana")


def test_adaln_velocity_field_outputs_zero_at_initialisation():
    """adaLN-zero plus zero-init output projection should give a quiet start."""
    torch.manual_seed(0)
    field = AdaLNVelocityField(
        4, condition_dim=2, hidden=16, layers=2, time_dim=8, cond_hidden=8
    )
    x = torch.randn(3, 4)
    t = torch.rand(3)
    c = torch.randn(3, 2)

    out = field(x, t, c)

    assert torch.allclose(out, torch.zeros_like(out))


def test_adaln_velocity_field_breaks_zero_after_training_step():
    """A single optimiser step should be enough to move the output off zero."""
    torch.manual_seed(0)
    field = AdaLNVelocityField(
        4, condition_dim=2, hidden=16, layers=2, time_dim=8, cond_hidden=8
    )
    x = torch.randn(3, 4)
    t = torch.rand(3)
    c = torch.randn(3, 2)
    target = torch.randn(3, 4)

    opt = torch.optim.SGD(field.parameters(), lr=1e-1)
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(field(x, t, c), target)
    loss.backward()
    opt.step()

    out = field(x, t, c)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_adaln_velocity_field_gradients_flow_to_context():
    field = AdaLNVelocityField(
        3, condition_dim=2, hidden=16, layers=2, time_dim=8, cond_hidden=8
    )
    x = torch.randn(4, 3)
    t = torch.rand(4)
    c = torch.randn(4, 2, requires_grad=True)

    # nudge parameters off the zero-init so gradients are non-trivial
    for p in field.cond_mlp[-1].parameters():
        p.data.add_(torch.randn_like(p) * 1e-2)
    for p in field.output_proj.parameters():
        p.data.add_(torch.randn_like(p) * 1e-2)

    out = field(x, t, c)
    out.sum().backward()

    assert c.grad is not None
    assert torch.isfinite(c.grad).all()
    assert c.grad.abs().sum().item() > 0
