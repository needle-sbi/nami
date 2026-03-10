from __future__ import annotations

import math

import pytest
import torch

from nami.paths.bridge import BrownianBridgePath


class TestBrownianBridgeValidation:
    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            BrownianBridgePath(sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            BrownianBridgePath(sigma=-1.0)

    def test_non_positive_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            BrownianBridgePath(eps=0.0)

        with pytest.raises(ValueError, match="eps must be positive"):
            BrownianBridgePath(eps=-1e-6)

    @pytest.mark.parametrize("eps", [0.5, 0.9], ids=["half", "large"])
    def test_large_eps_raises(self, eps):
        with pytest.raises(ValueError, match="eps must be < 0.5"):
            BrownianBridgePath(eps=eps)


class TestBrownianBridgeSampleXt:
    """Tests for BrownianBridgePath.sample_xt method."""

    def test_t0_returns_target(self):
        """At t=0, xt should equal x_target (with z=0)."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.zeros(4)
        z = torch.zeros_like(x_target)

        xt = path.sample_xt(x_target, x_source, t, z=z)

        assert torch.allclose(xt, x_target)

    def test_t1_returns_source(self):
        """At t=1, xt should equal x_source (with z=0)."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.ones(4)
        z = torch.zeros_like(x_target)

        xt = path.sample_xt(x_target, x_source, t, z=z)

        assert torch.allclose(xt, x_source)

    def test_midpoint_mean(self):
        """At t=0.5, mean should be the midpoint of x_target and x_source."""
        path = BrownianBridgePath()
        x_target = torch.ones(4, 3)
        x_source = torch.zeros(4, 3)
        t = torch.full((4,), 0.5)
        z = torch.zeros_like(x_target)

        xt = path.sample_xt(x_target, x_source, t, z=z)

        expected = 0.5 * x_target + 0.5 * x_source
        assert torch.allclose(xt, expected)

    def test_midpoint_std(self):
        """At t=0.5, std should be sigma/2."""
        path = BrownianBridgePath(sigma=1.0)
        x_target = torch.zeros(4, 3)
        x_source = torch.zeros(4, 3)
        t = torch.full((4,), 0.5)
        z = torch.ones_like(x_target)

        xt = path.sample_xt(x_target, x_source, t, z=z)

        # std = sigma * sqrt(0.5 * 0.5) = sigma * 0.5
        expected = 0.5 * z
        assert torch.allclose(xt, expected)

    def test_deterministic_with_fixed_z(self):
        """Fixed z gives reproducible results."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4)
        z = torch.randn_like(x_target)

        xt1 = path.sample_xt(x_target, x_source, t, z=z)
        xt2 = path.sample_xt(x_target, x_source, t, z=z)

        assert torch.allclose(xt1, xt2)

    def test_broadcasting(self):
        """Time tensor should broadcast correctly over batch dimensions."""
        path = BrownianBridgePath()
        x_target = torch.randn(2, 3, 4)
        x_source = torch.randn(2, 3, 4)
        t = torch.tensor([0.0, 1.0])
        z = torch.zeros_like(x_target)

        xt = path.sample_xt(x_target, x_source, t, z=z)

        assert xt.shape == x_target.shape
        assert torch.allclose(xt[0], x_target[0])
        assert torch.allclose(xt[1], x_source[1])


class TestBrownianBridgeTargetUt:
    """Tests for BrownianBridgePath.target_ut method."""

    def test_without_xt_returns_linear(self):
        """Without xt, target_ut returns x_source - x_target (same as LinearPath)."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4)

        ut = path.target_ut(x_target, x_source, t)

        assert torch.allclose(ut, x_source - x_target)

    def test_with_xt_at_mean_correction_vanishes(self):
        """When xt equals the mean, the stochastic correction vanishes."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)  # avoid boundary

        t_expanded = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t_expanded) * x_target + t_expanded * x_source
        ut = path.target_ut(x_target, x_source, t, xt=mu)

        expected = x_source - x_target
        assert torch.allclose(ut, expected, atol=1e-5)

    def test_known_values_t025(self):
        """Hand-computed values at t=0.25 with fixed z."""
        path = BrownianBridgePath(sigma=1.0)
        x_target = torch.tensor([[1.0]])
        x_source = torch.tensor([[0.0]])
        t = torch.tensor([0.25])
        z = torch.tensor([[1.0]])

        xt = path.sample_xt(x_target, x_source, t, z=z)
        ut = path.target_ut(x_target, x_source, t, xt=xt)

        # mu = 0.75, std = sqrt(0.25*0.75) = sqrt(3)/4
        # xt = 0.75 + sqrt(3)/4
        # coeff = (1-0.5)/(2*0.25*0.75) = 0.5/0.375 = 4/3
        # ut = (0-1) + (4/3)*sqrt(3)/4 = -1 + sqrt(3)/3
        expected = -1.0 + math.sqrt(3) / 3.0
        assert torch.allclose(ut, torch.tensor([[expected]]), atol=1e-5)


class TestBrownianBridgeScoreTarget:
    """Tests for BrownianBridgePath.score_target method."""

    def test_at_mean_returns_zero(self):
        """When xt equals the mean, score is zero."""
        path = BrownianBridgePath()
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)

        t_expanded = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t_expanded) * x_target + t_expanded * x_source
        score = path.score_target(x_target, x_source, t, xt=mu)

        assert torch.allclose(score, torch.zeros_like(score), atol=1e-5)

    def test_sign_points_toward_mean(self):
        """Score should point from xt toward mu."""
        path = BrownianBridgePath()
        x_target = torch.zeros(4, 3)
        x_source = torch.zeros(4, 3)
        t = torch.full((4,), 0.5)
        # xt above zero -> score should be negative (pointing toward mu=0)
        xt = torch.ones(4, 3)

        score = path.score_target(x_target, x_source, t, xt=xt)

        assert (score < 0).all()

    def test_scales_with_sigma(self):
        """Doubling sigma should quarter the score magnitude."""
        path1 = BrownianBridgePath(sigma=1.0)
        path2 = BrownianBridgePath(sigma=2.0)
        x_target = torch.randn(4, 3)
        x_source = torch.randn(4, 3)
        t = torch.rand(4).clamp(0.05, 0.95)

        t_expanded = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t_expanded) * x_target + t_expanded * x_source
        xt = mu + torch.randn_like(mu)

        score1 = path1.score_target(x_target, x_source, t, xt=xt)
        score2 = path2.score_target(x_target, x_source, t, xt=xt)

        assert torch.allclose(score2, score1 / 4.0, atol=1e-5)


class TestBrownianBridgeGammaSchedule:
    def test_gamma_schedule_matches_sigma_and_eps(self):
        sigma = 1.7
        eps = 1e-4
        path = BrownianBridgePath(sigma=sigma, eps=eps)
        schedule = path.gamma_schedule()
        t = torch.linspace(0.1, 0.9, 9)

        assert schedule.scale == pytest.approx(sigma**2)
        assert schedule.eps == pytest.approx(eps)

        expected_gg = 0.5 * (sigma**2) * (1.0 - 2.0 * t)
        assert torch.allclose(
            schedule.gamma_gamma_dot(t), expected_gg, atol=1e-6, rtol=1e-6
        )
