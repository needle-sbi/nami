from __future__ import annotations

import pytest
import torch
from torch import nn

from nami import ExactDivergence, FlowMatching, RK4, StandardNormal
from nami.fields.base import VectorField


class PlainField(nn.Module):
    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return torch.zeros_like(x)
        return (x * 0) + c.expand_as(x)


class BaseVectorField(VectorField):
    @property
    def event_ndim(self):
        return 1

    def forward(self, x, t, c=None):
        _ = t, c
        return x


def test_log_prob_without_divergence_path_raises_clean_error():
    process = FlowMatching(
        PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()
    x = torch.zeros(4, 2)

    with pytest.raises(
        TypeError,
        match=r"log_prob requires either `estimator=\.\.\.` or a field implementing "
        r"`call_and_divergence\(x, t, c\)`",
    ):
        process.log_prob(x)


def test_log_prob_hides_internal_not_implemented_cause():
    process = FlowMatching(
        BaseVectorField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()
    x = torch.zeros(4, 2)

    with pytest.raises(
        TypeError,
        match=r"log_prob requires either `estimator=\.\.\.` or a field implementing "
        r"`call_and_divergence\(x, t, c\)`",
    ) as excinfo:
        process.log_prob(x)

    assert excinfo.value.__cause__ is None


def test_conditional_log_prob_supports_plain_module_with_estimator():
    context = torch.randn(5, 1)
    process = FlowMatching(
        PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )(context)
    x = torch.zeros(5, 2)

    log_prob = process.log_prob(x, estimator=ExactDivergence(max_dim=8))

    assert log_prob.shape == (5,)
    assert torch.isfinite(log_prob).all()
