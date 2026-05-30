from __future__ import annotations

import pytest
import torch

from nami.core.broadcast import broadcast


def test_broadcast_expands_x_t_c_to_joint_leading_shape() -> None:
    x = torch.randn(2, 1, 3, 4)
    t = torch.randn(1, 5)
    c = torch.randn(1, 5, 7)

    x_out, t_out, c_out = broadcast(x, t, c, event_ndim=2)

    assert x_out.shape == (2, 5, 3, 4)
    assert t_out is not None
    assert c_out is not None
    assert t_out.shape == (2, 5)
    assert c_out.shape == (2, 5, 7)
    torch.testing.assert_close(x_out, x.expand(2, 5, 3, 4))
    torch.testing.assert_close(t_out, t.expand(2, 5))
    torch.testing.assert_close(c_out, c.expand(2, 5, 7))


def test_broadcast_handles_missing_optional_tensors() -> None:
    x = torch.randn(2, 3, 4)

    x_out, t_out, c_out = broadcast(x, t=None, c=None, event_ndim=2)

    assert x_out.shape == x.shape
    assert t_out is None
    assert c_out is None


def test_broadcast_supports_scalar_event_ndim_zero() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([[0.5], [1.5]])
    c = torch.tensor([[2.0, 4.0]])

    x_out, t_out, c_out = broadcast(x, t, c, event_ndim=0)

    assert x_out.shape == (2, 3)
    assert t_out is not None
    assert c_out is not None
    assert t_out.shape == (2, 3)
    assert c_out.shape == (2, 3, 2)
    torch.testing.assert_close(x_out, x.expand(2, 3))
    torch.testing.assert_close(t_out, t.expand(2, 3))
    torch.testing.assert_close(c_out, c.expand(2, 3, 2))


def test_broadcast_rejects_scalar_context_tensor() -> None:
    x = torch.randn(2, 3)
    t = torch.randn(2)
    c = torch.tensor(1.0)

    with pytest.raises(ValueError, match="context tensor must have at least 1 dim"):
        broadcast(x, t, c, event_ndim=1)


def test_broadcast_raises_on_incompatible_shapes() -> None:
    x = torch.randn(2, 4)
    t = torch.randn(3)

    with pytest.raises(ValueError, match="failed to broadcast"):
        broadcast(x, t, c=None, event_ndim=1)


def test_broadcast_reraises_runtime_error_when_validation_disabled() -> None:
    x = torch.randn(2, 4)
    t = torch.randn(3)

    with pytest.raises(RuntimeError):
        broadcast(x, t, c=None, event_ndim=1, validate_args=False)
