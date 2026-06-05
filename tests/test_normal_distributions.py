from __future__ import annotations

import pytest
import torch
from torch.distributions import Independent, Normal

from nami.core.specs import TensorSpec
from nami.distributions.mask import AllMask
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


def test_standard_normal_accepts_spec() -> None:
    dist = StandardNormal(spec=TensorSpec((2, 3), dtype=torch.float64))

    assert dist.event_shape == torch.Size([2, 3])
    assert dist.sample().dtype == torch.float64
    assert dist.spec.event_shape == (2, 3)


def test_standard_normal_rejects_spec_with_explicit_args() -> None:
    with pytest.raises(ValueError, match="not both"):
        StandardNormal((2,), spec=TensorSpec((2,)))
    with pytest.raises(ValueError, match="not both"):
        StandardNormal(spec=TensorSpec((2,)), dtype=torch.float64)
    with pytest.raises(ValueError, match="required"):
        StandardNormal()


def test_diagonal_normal_spec_survives_expand() -> None:
    loc = torch.zeros(2, 3)
    dist = DiagonalNormal(loc=loc, scale=torch.ones(2, 3), event_ndim=1)

    assert dist.spec.event_shape == (3,)
    assert dist.event_ndim == 1

    expanded = dist.expand(torch.Size([5, 2]))
    assert expanded.event_ndim == 1
    assert expanded.spec.event_shape == (3,)


def test_all_mask_accepts_spec() -> None:
    dist = AllMask(spec=TensorSpec((4,)), mask_index=5)

    assert dist.sample().shape == (4,)
    assert (dist.sample() == 5).all()
    assert dist.spec.event_shape == (4,)
    with pytest.raises(ValueError, match="not both"):
        AllMask((4,), spec=TensorSpec((4,)), mask_index=5)
