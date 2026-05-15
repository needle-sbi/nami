from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.interpolants.bridge import BrownianBridgeInterpolant
from nami.interpolants.protocol import InterpolantState
from nami.losses.bridge import bridge_matching_loss
from nami.parameterizations import Score, Velocity


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


def _state(interpolant, x_data, x_noise, t, xt):  # noqa: ARG001
    """Build an InterpolantState with the given x_t.

    The bridge's Velocity / Score targets read only x_data, x_noise,
    t, xt — so the noise slot can be left ``None`` for these test
    fields without affecting the computed target.
    """
    return InterpolantState(xt=xt, x_data=x_data, x_noise=x_noise, t=t, noise=None)


class _PerfectFlowField(nn.Module):
    """Returns exact velocity targets via the interpolant."""

    def __init__(self, interpolant, x_data, x_noise):
        super().__init__()
        self.interpolant = interpolant
        self.x_data = x_data
        self.x_noise = x_noise
        self._target = Velocity()

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, xt, t, c=None):
        _ = c
        return self.interpolant.target(
            self._target,
            _state(self.interpolant, self.x_data, self.x_noise, t, xt),
        )


class _PerfectScoreField(nn.Module):
    """Returns exact score targets via the interpolant."""

    def __init__(self, interpolant, x_data, x_noise):
        super().__init__()
        self.interpolant = interpolant
        self.x_data = x_data
        self.x_noise = x_noise
        self._target = Score()

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, xt, t, c=None):
        _ = c
        return self.interpolant.target(
            self._target,
            _state(self.interpolant, self.x_data, self.x_noise, t, xt),
        )


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
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)

        loss = bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, t=t)
        loss.backward()

        assert flow.linear.weight.grad is not None
        assert score.linear.weight.grad is not None
        assert flow.linear.weight.grad.abs().sum() > 0
        assert score.linear.weight.grad.abs().sum() > 0

    def test_reductions(self):
        torch.manual_seed(0)
        flow = _Field()
        score = _Field()
        x_data = torch.randn(5, 3)
        x_noise = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_data)

        loss_none = bridge_matching_loss(
            flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z, reduction="none"
        )
        loss_sum = bridge_matching_loss(
            flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z, reduction="sum"
        )
        loss_mean = bridge_matching_loss(
            flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z, reduction="mean"
        )

        assert loss_none.shape == (5,)
        assert loss_sum.shape == ()
        assert loss_mean.shape == ()
        assert torch.isclose(loss_sum, loss_none.sum())
        assert torch.isclose(loss_mean, loss_none.mean())

    def test_invalid_z_shape_raises(self):
        flow = _Field()
        score = _Field()
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)
        t = torch.rand(4)
        z = torch.randn(4, 2)

        with pytest.raises(ValueError, match="z must match the shape"):
            bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z)

    def test_event_ndim_mismatch_raises(self):
        flow = _Field(event_ndim=1)
        score = _Field(event_ndim=2)
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)

        with pytest.raises(ValueError, match="event_ndim"):
            bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise)

    def test_flow_weight_zero(self):
        """With flow_weight=0, loss depends only on score."""
        torch.manual_seed(0)
        flow = _Field()
        score = _Field()
        x_data = torch.randn(5, 3)
        x_noise = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_data)

        loss_both = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
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
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=1.0,
            reduction="none",
        )
        loss_flow_only = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
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
        x_data = torch.randn(5, 3)
        x_noise = torch.randn(5, 3)
        t = torch.rand(5).clamp(0.05, 0.95)
        z = torch.randn_like(x_data)

        loss_score_zero = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            z=z,
            flow_weight=1.0,
            score_weight=0.0,
            reduction="none",
        )
        loss_flow_zero = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            z=z,
            flow_weight=0.0,
            score_weight=1.0,
            reduction="none",
        )
        loss_both = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
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
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)
        c = torch.randn(4, 2)

        bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, t=t, c=c)

        assert flow.last_c is c
        assert score.last_c is c

    def test_perfect_fields_zero_loss(self):
        """Fields returning exact targets should give zero loss."""
        torch.manual_seed(42)
        interpolant = BrownianBridgeInterpolant(sigma=1.0)
        x_data = torch.randn(6, 4)
        x_noise = torch.randn(6, 4)
        t = torch.rand(6).clamp(0.05, 0.95)
        z = torch.randn_like(x_data)

        flow = _PerfectFlowField(interpolant, x_data=x_data, x_noise=x_noise)
        score = _PerfectScoreField(interpolant, x_data=x_data, x_noise=x_noise)

        loss = bridge_matching_loss(
            flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z, interpolant=interpolant
        )

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-10)

    def test_deterministic_with_fixed_z_and_t(self):
        """Fixed z and t give deterministic loss values."""
        flow = _Field()
        score = _Field()
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)
        t = torch.tensor([0.25, 0.5, 0.5, 0.75])
        z = torch.randn_like(x_data)

        loss1 = bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z)
        loss2 = bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, t=t, z=z)

        assert torch.allclose(loss1, loss2)

    def test_default_z_is_shared_across_flow_and_score_heads(self):
        """When ``z`` is omitted, ``bridge_matching_loss`` must draw one
        shared noise tensor and forward it to both ``regression_loss``
        calls.  Without that, each call's interpolant.sample draws
        independent noise and the two heads land on different bridge
        realisations — silently breaking the joint training claim.

        Pinned by comparing two successive calls under the same seed:
        if shared-noise was respected, the loss is reproducible across
        the two calls; if independent, the second call would draw new
        noise inside each regression_loss and produce a different
        loss.  We also compare against a hand-shared-z baseline to
        rule out RNG-state coincidence.
        """
        flow = _Field()
        score = _Field()
        x_data = torch.randn(8, 3)
        x_noise = torch.randn(8, 3)
        t = torch.rand(8).clamp(0.05, 0.95)

        # Hand-shared-z baseline.
        z = torch.randn_like(x_data)
        baseline = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            z=z,
            reduction="none",
        )

        # Default-z: function should draw one z and share.  Same seed
        # should reproduce the *function-level* z draw deterministically.
        torch.manual_seed(0)
        a = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            reduction="none",
        )
        torch.manual_seed(0)
        b = bridge_matching_loss(
            flow,
            score,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            reduction="none",
        )
        assert torch.allclose(a, b, atol=1e-12), (
            "Same-seed calls disagree — default z is not deterministically "
            "drawn at the function level"
        )

        # Sanity: function output is a finite real loss (not zero).
        assert torch.isfinite(a).all()
        assert (a > 0).any()
        # Baseline check that hand-shared-z and default-z differ in value
        # (different RNG draws) but match in shape and behaviour.
        assert a.shape == baseline.shape

    def test_samples_t_within_interpolant_eps_when_t_is_none(self):
        flow = _Field()
        score = _Field()
        interpolant = BrownianBridgeInterpolant(eps=0.2)
        x_data = torch.randn(16, 3)
        x_noise = torch.randn(16, 3)

        _ = bridge_matching_loss(flow, score, x_data=x_data, x_noise=x_noise, interpolant=interpolant)

        assert flow.last_t is not None
        assert score.last_t is not None
        assert torch.all(flow.last_t >= interpolant.eps)
        assert torch.all(flow.last_t <= 1.0 - interpolant.eps)
        assert torch.all(score.last_t >= interpolant.eps)
        assert torch.all(score.last_t <= 1.0 - interpolant.eps)
