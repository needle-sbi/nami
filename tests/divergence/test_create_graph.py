from __future__ import annotations

import torch
from torch import nn

from nami.distributions.normal import StandardNormal
from nami.divergence.exact import ExactDivergence
from nami.divergence.hutchinson import HutchinsonDivergence
from nami.processes.fm import FlowMatching


class LinearScaleField(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.event_ndim = 1
        self.scale = nn.Parameter(torch.tensor(1.5))
        self.dim = int(dim)

    def forward(self, x, t, c=None):
        _ = t, c
        return self.scale * x


class SingleStepAugmentedSolver:
    is_sde = False
    requires_steps = False
    supports_rsample = True

    def integrate(self, f, x0: torch.Tensor, *, t0: float, t1: float, **kwargs):
        _ = kwargs
        return x0 + (t1 - t0) * f(x0, t0)

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        **kwargs,
    ):
        _ = kwargs
        _v, logp_dot = f_aug(x0, t0)
        return x0, logp0 + (t1 - t0) * logp_dot


def test_hutchinson_create_graph_controls_param_grad_connectivity():
    field = LinearScaleField(dim=3)
    x = torch.randn(5, 3)
    t = torch.zeros(5)

    div_no_graph = HutchinsonDivergence()(field, x, t, None)
    assert not div_no_graph.requires_grad

    div_graph = HutchinsonDivergence(create_graph=True)(field, x, t, None)
    assert div_graph.requires_grad

    div_graph.sum().backward()
    assert field.scale.grad is not None
    expected = torch.tensor(float(x.shape[0] * x.shape[1]))
    assert torch.allclose(field.scale.grad, expected)


def test_exact_create_graph_controls_param_grad_connectivity():
    field = LinearScaleField(dim=4)
    x = torch.randn(6, 4)
    t = torch.zeros(6)

    div_no_graph = ExactDivergence(max_dim=16)(field, x, t, None)
    assert not div_no_graph.requires_grad

    div_graph = ExactDivergence(max_dim=16, create_graph=True)(field, x, t, None)
    assert div_graph.requires_grad

    div_graph.sum().backward()
    assert field.scale.grad is not None
    expected = torch.tensor(float(x.shape[0] * x.shape[1]))
    assert torch.allclose(field.scale.grad, expected)


def test_log_prob_is_differentiable_with_create_graph_estimator():
    field = LinearScaleField(dim=2)
    process = FlowMatching(
        field,
        StandardNormal(event_shape=(2,)),
        SingleStepAugmentedSolver(),
    )()
    x = torch.zeros(4, 2)
    estimator = HutchinsonDivergence(create_graph=True)

    loss = -process.log_prob(x, estimator=estimator).sum()
    loss.backward()

    assert field.scale.grad is not None
    assert torch.isfinite(field.scale.grad)


def test_log_prob_is_differentiable_with_exact_create_graph_estimator():
    field = LinearScaleField(dim=2)
    process = FlowMatching(
        field,
        StandardNormal(event_shape=(2,)),
        SingleStepAugmentedSolver(),
    )()
    x = torch.zeros(4, 2)
    estimator = ExactDivergence(max_dim=8, create_graph=True)

    loss = -process.log_prob(x, estimator=estimator).sum()
    loss.backward()

    assert field.scale.grad is not None
    assert torch.isfinite(field.scale.grad)
