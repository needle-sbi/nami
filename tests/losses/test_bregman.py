from __future__ import annotations

import math

import pytest
import torch

from nami import ItakuraSaito, KLDivergence, SquaredL2
from nami.losses._common import per_sample_mse


def test_squared_l2_matches_per_sample_mse():
    pred = torch.randn(8, 5)
    target = torch.randn(8, 5)
    lead = (8,)
    assert torch.allclose(
        SquaredL2()(pred, target, lead), per_sample_mse(pred, target, lead)
    )


@pytest.mark.parametrize(
    "divergence",
    [SquaredL2(), KLDivergence(), ItakuraSaito()],
)
def test_divergence_is_zero_at_coincidence(divergence):
    a = torch.rand(6, 4) + 0.1  # strictly positive: valid for all three domains
    a = a / a.sum(dim=-1, keepdim=True)  # also a valid simplex point for KL
    value = divergence(a.clone(), a.clone(), (6,))
    assert torch.allclose(value, torch.zeros(6), atol=1e-6)


@pytest.mark.parametrize(
    "divergence",
    [SquaredL2(), KLDivergence(), ItakuraSaito()],
)
def test_divergence_is_non_negative(divergence):
    torch.manual_seed(0)
    a = torch.rand(16, 4) + 0.05
    b = torch.rand(16, 4) + 0.05
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    value = divergence(b, a, (16,))  # prediction=b, target=a
    assert (value >= -1e-6).all()


def test_kl_matches_closed_form():
    target = torch.tensor([[0.7, 0.3]])
    pred = torch.tensor([[0.5, 0.5]])
    expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    got = KLDivergence()(pred, target, (1,))
    assert torch.allclose(got, torch.tensor([expected]), atol=1e-6)


def test_itakura_saito_matches_closed_form():
    target = torch.tensor([[2.0, 1.0]])
    pred = torch.tensor([[1.0, 4.0]])
    # sum_i (a/b - log(a/b) - 1), then mean over the 2 elements.
    terms = [
        (2.0 / 1.0) - math.log(2.0 / 1.0) - 1.0,
        (1.0 / 4.0) - math.log(1.0 / 4.0) - 1.0,
    ]
    expected = sum(terms) / 2
    got = ItakuraSaito()(pred, target, (1,))
    assert torch.allclose(got, torch.tensor([expected]), atol=1e-6)


def test_kl_gradient_equals_cross_entropy_gradient():
    """KL and cross-entropy differ only by the prediction-independent entropy of
    the target, so their gradients w.r.t. the prediction coincide — the property
    that lets CGM use either for rate matching."""
    target = torch.tensor([[0.6, 0.4]])
    raw = torch.randn(1, 2, requires_grad=True)

    pred = torch.softmax(raw, dim=-1)
    kl = KLDivergence()(pred, target, (1,)).sum()
    (kl_grad,) = torch.autograd.grad(kl, raw, retain_graph=True)

    pred2 = torch.softmax(raw, dim=-1)
    ce = -(target * pred2.clamp_min(1e-8).log()).sum()
    (ce_grad,) = torch.autograd.grad(ce, raw)

    assert torch.allclose(kl_grad, ce_grad, atol=1e-6)
