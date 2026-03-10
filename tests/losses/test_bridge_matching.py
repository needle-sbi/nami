from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.losses.bridge import bridge_matching_loss
from nami.paths.bridge import BrownianBridgePath


class _Field(nn.Module):
    def __init__(self, event_ndim: int = 1):
        super().__init__()
        self._event_ndim = event_ndim
        self.last_c = None
        self.last_t = None

    @property
    def event_ndim(self) -> int:
        return self._event_ndim

    def forward(self, x, t, c=None):
        self.last_t = t
        self.last_c = c
        return torch.zeros_like(x)


class _PerfectFlowField(nn.Module):
    """Returns exact flow targets (requires path and paired samples)."""

    def __init__(self, path, x_target, x_source):
        super().__init__()
        self.path = path
        self.x_target = x_target
        self.x_source = x_source

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, xt, t, c=None):
        _ = c
        return self.path.target_ut(self.x_target, self.x_source, t, xt=xt)


class _PerfectScoreField(nn.Module):
    """Returns exact score targets (requires path and paired samples)."""

    def __init__(self, path, x_target, x_source):
        super().__init__()
        self.path = path
        self.x_target = x_target
        self.x_source = x_source

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, xt, t, c=None):
        _ = c
        return self.path.score_target(self.x_target, self.x_source, t, xt=xt)


class _LinearField(nn.Module):
    """Minimal learnable field for gradient testing."""

    def __init__(self, dim: int, event_ndim: int = 1):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self._event_ndim = event_ndim

    @property
    def event_ndim(self) -> int:
        return self._event_ndim

    def forward(self, x, t, c=None):
        _ = t, c
        return self.linear(x)


class TestBridgeMatchingLoss:
    def test_gradients_flow_to_parameters(self):
        """Loss should produce gradients for both flow and score model parameters."""
        torch.manual_seed(0)
        flow = _LinearField(dim=3)
        score = _LinearField(dim=3)
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)

        loss = bridge_matching_loss(flow, score, x_target, x_source, t=t)
        loss.backward()

        assert flow.linear.weight.grad is not None
        assert score.linear.weight.grad is not None
        assert flow.linear.weight.grad.abs().sum() > 0
        assert score.linear.weight.grad.abs().sum() > 0

    def test_reductions(self):
        torch.manual_seed(0)
        flow = _Field()
        score = _Field()
        x_target = torch.randn(5, 3)
        x_source = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_target)

        loss_none = bridge_matching_loss(
            flow, score, x_target, x_source, t=t, z=z, reduction="none"
        )
        loss_sum = bridge_matching_loss(
            flow, score, x_target, x_source, t=t, z=z, reduction="sum"
        )
        loss_mean = bridge_matching_loss(
            flow, score, x_target, x_source, t=t, z=z, reduction="mean"
        )

        assert loss_none.shape == (5,)
        assert loss_sum.shape == ()
        assert loss_mean.shape == ()
        assert torch.isclose(loss_sum, loss_none.sum())
        assert torch.isclose(loss_mean, loss_none.mean())

    def test_invalid_z_shape_raises(self):
        flow = _Field()
        score = _Field()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4)
        z = torch.randn(4, 2)

        with pytest.raises(ValueError, match="z must match the shape"):
            bridge_matching_loss(flow, score, x_target, x_source, t=t, z=z)

    def test_event_ndim_mismatch_raises(self):
        flow = _Field(event_ndim=1)
        score = _Field(event_ndim=2)
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)

        with pytest.raises(ValueError, match="event_ndim"):
            bridge_matching_loss(flow, score, x_target, x_source)

    def test_flow_weight_zero(self):
        """With flow_weight=0, loss depends only on score."""
        torch.manual_seed(0)
        flow = _Field()
        score = _Field()
        x_target = torch.randn(5, 3)
        x_source = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_target)

        loss_both = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=0.0,
            score_weight=1.0,
            reduction="none",
        )
        # Score-only loss should not depend on flow field output
        # Since ZeroField returns zeros, score_mse = ||0 - score_target||^2
        # Changing flow_weight to 0 should remove the flow contribution
        loss_with_flow = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=1.0,
            reduction="none",
        )
        loss_flow_only = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=0.0,
            reduction="none",
        )

        # loss_both = flow_mse + score_mse
        # loss_with_flow_zero = score_mse
        # loss_flow_only = flow_mse
        assert torch.allclose(loss_with_flow, loss_both + loss_flow_only)

    def test_score_weight_zero(self):
        """With score_weight=0, loss depends only on flow."""
        torch.manual_seed(0)
        flow = _Field()
        score = _Field()
        x_target = torch.randn(5, 3)
        x_source = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_target)

        loss_score_zero = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=0.0,
            reduction="none",
        )
        loss_flow_zero = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=0.0,
            score_weight=1.0,
            reduction="none",
        )
        loss_both = bridge_matching_loss(
            flow,
            score,
            x_target,
            x_source,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=1.0,
            reduction="none",
        )

        assert torch.allclose(loss_both, loss_score_zero + loss_flow_zero)

    def test_context_forwarding(self):
        """Context c should be forwarded to both fields."""
        flow = _Field()
        score = _Field()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)
        c = torch.randn(4, 2)

        bridge_matching_loss(flow, score, x_target, x_source, t=t, c=c)

        assert flow.last_c is c
        assert score.last_c is c

    def test_perfect_fields_zero_loss(self):
        """Fields returning exact targets should give zero loss."""
        torch.manual_seed(42)
        path = BrownianBridgePath(sigma=1.0)
        x_target = torch.randn(6, 4)
        x_source = torch.randn(6, 4)
        t = torch.rand(6).clamp(0.05, 0.95)
        z = torch.randn_like(x_target)

        flow = _PerfectFlowField(path, x_target, x_source)
        score = _PerfectScoreField(path, x_target, x_source)

        loss = bridge_matching_loss(
            flow, score, x_target, x_source, t=t, z=z, path=path
        )

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-10)

    def test_deterministic_with_fixed_z_and_t(self):
        """Fixed z and t give deterministic loss values."""
        flow = _Field()
        score = _Field()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.tensor([0.25, 0.5, 0.5, 0.75])
        z = torch.randn_like(x_target)

        loss1 = bridge_matching_loss(flow, score, x_target, x_source, t=t, z=z)
        loss2 = bridge_matching_loss(flow, score, x_target, x_source, t=t, z=z)

        assert torch.allclose(loss1, loss2)

    def test_samples_t_within_path_eps_when_t_is_none(self):
        flow = _Field()
        score = _Field()
        path = BrownianBridgePath(eps=0.2)
        x_target = torch.randn(16, 3)
        x_source = torch.randn(16, 3)

        _ = bridge_matching_loss(flow, score, x_target, x_source, path=path)

        assert flow.last_t is not None
        assert score.last_t is not None
        assert torch.all(flow.last_t >= path.eps)
        assert torch.all(flow.last_t <= 1.0 - path.eps)
        assert torch.all(score.last_t >= path.eps)
        assert torch.all(score.last_t <= 1.0 - path.eps)
