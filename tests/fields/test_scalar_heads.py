"""Scalar heads (ActionHead, LogDensityHead) and shared field helpers."""

from __future__ import annotations

import pytest
import torch

from nami.fields import ActionHead, LogDensityHead
from nami.fields._common import require_event_ndim


@pytest.mark.parametrize("head_cls", [ActionHead, LogDensityHead])
def test_scalar_head_conditions_on_context(head_cls) -> None:
    """A conditional head consumes ``c`` and the output depends on it."""
    torch.manual_seed(0)
    head = head_cls(dim=3, condition_dim=2)
    x = torch.randn(5, 3)
    t = torch.rand(5)

    out_a = head(x, t, torch.zeros(5, 2))
    out_b = head(x, t, torch.ones(5, 2))

    assert out_a.shape == (5,)
    assert not torch.allclose(out_a, out_b), "context should change the output"


def test_require_event_ndim_rejects_field_without_attribute() -> None:
    def bare_field(x, t, c=None):
        _ = t, c
        return x

    with pytest.raises(ValueError, match="event_ndim is required"):
        require_event_ndim(bare_field)
