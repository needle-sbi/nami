from __future__ import annotations

import torch
from torch.distributions import Independent, Normal

from nami.distributions.normal import DiagonalNormal, StandardNormal


def test_standard_normal_mean_variance_and_expand() -> None:
    dist = StandardNormal(event_shape=(2, 3), batch_shape=(4,))

    assert dist.batch_shape == torch.Size([4])
    assert dist.event_shape == torch.Size([2, 3])
    torch.testing.assert_close(dist.mean, torch.zeros(4, 2, 3))
    torch.testing.assert_close(dist.variance, torch.ones(4, 2, 3))

    expanded = dist.expand(torch.Size([5, 4]))
    assert expanded.batch_shape == torch.Size([5, 4])
    assert expanded.event_shape == torch.Size([2, 3])
    assert expanded.sample().shape == (5, 4, 2, 3)


def test_diagonal_normal_sample_rsample_log_prob_and_expand() -> None:
    loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    scale = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float64)
    dist = DiagonalNormal(loc=loc, scale=scale, event_ndim=1)
    reference = Independent(Normal(loc, scale), 1)

    sample = dist.sample()
    rsample = dist.rsample()
    value = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float64)

    assert sample.shape == (2, 2)
    assert rsample.shape == (2, 2)
    torch.testing.assert_close(dist.log_prob(value), reference.log_prob(value))
    torch.testing.assert_close(dist.mean, loc)
    torch.testing.assert_close(dist.variance, scale.square())

    expanded = dist.expand(torch.Size([3, 2]))
    assert expanded.batch_shape == torch.Size([3, 2])
    assert expanded.event_shape == torch.Size([2])
    assert expanded.sample().shape == (3, 2, 2)
