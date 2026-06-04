from __future__ import annotations

import pytest
import torch

from nami.interpolants.gamma import (
    BrownianGamma,
    GammaSchedule,
    ScaledBrownianGamma,
    ZeroGamma,
)


class _AffineGamma(GammaSchedule):
    """Minimal subclass relying on the base-class default product."""

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class TestGammaScheduleDefault:
    def test_default_product_multiplies_gamma_and_gamma_dot(self):
        """Subclasses without a closed-form override get gamma * gamma_dot."""
        schedule = _AffineGamma()
        t = torch.linspace(0.1, 0.9, 9)

        assert torch.allclose(schedule.gamma_gamma_dot(t), t)


class TestZeroGamma:
    def test_outputs_zero(self):
        schedule = ZeroGamma()
        t = torch.rand(8)

        assert torch.all(schedule.gamma(t) == 0)
        assert torch.all(schedule.gamma_dot(t) == 0)
        assert torch.all(schedule.gamma_gamma_dot(t) == 0)


class TestBrownianGamma:
    def test_gamma_product_matches_explicit_product(self):
        schedule = BrownianGamma()
        t = torch.linspace(0.1, 0.9, 9)

        lhs = schedule.gamma(t) * schedule.gamma_dot(t)
        rhs = schedule.gamma_gamma_dot(t)

        assert torch.allclose(lhs, rhs, rtol=1e-6, atol=1e-6)


class TestScaledBrownianGamma:
    @pytest.mark.parametrize("scale", [0.0, -1.0], ids=["zero", "negative"])
    def test_invalid_scale_raises(self, scale):
        with pytest.raises(ValueError, match="scale must be positive"):
            ScaledBrownianGamma(scale=scale)

    def test_reduces_to_brownian_when_scale_one(self):
        scaled = ScaledBrownianGamma(scale=1.0)
        base = BrownianGamma()
        t = torch.linspace(0.1, 0.9, 9)

        assert torch.allclose(scaled.gamma(t), base.gamma(t), rtol=1e-6, atol=1e-6)
        assert torch.allclose(
            scaled.gamma_dot(t),
            base.gamma_dot(t),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize("sigma", [0.0, -1.0], ids=["zero", "negative"])
    def test_from_sigma_invalid_raises(self, sigma):
        with pytest.raises(ValueError, match="sigma must be positive"):
            ScaledBrownianGamma.from_sigma(sigma=sigma)

    def test_from_sigma_matches_scale_parameterization(self):
        sigma = 1.7
        eps = 1e-5
        t = torch.linspace(0.1, 0.9, 9)

        from_sigma = ScaledBrownianGamma.from_sigma(sigma=sigma, eps=eps)
        from_scale = ScaledBrownianGamma(scale=sigma**2, eps=eps)

        assert torch.allclose(
            from_sigma.gamma(t), from_scale.gamma(t), atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            from_sigma.gamma_dot(t), from_scale.gamma_dot(t), atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            from_sigma.gamma_gamma_dot(t),
            from_scale.gamma_gamma_dot(t),
            atol=1e-6,
            rtol=1e-6,
        )
