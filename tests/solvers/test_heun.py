from __future__ import annotations

import pytest
import torch

from nami.solvers.heun import Heun


class TestHeunInit:
    """Tests for Heun solver initialization."""

    def test_basic_init(self):
        heun = Heun(steps=10)
        assert heun.steps == 10
        assert heun.requires_steps
        assert heun.supports_rsample
        assert not heun.is_sde

    @pytest.mark.parametrize("steps", [0, -1, -10], ids=["zero", "neg1", "neg10"])
    def test_invalid_steps(self, steps):
        with pytest.raises(ValueError, match="steps must be positive"):
            Heun(steps=steps)


class TestHeunIntegrate:
    """Tests for Heun.integrate method."""

    def test_constant_field(self, sample_tensor_2d):
        """Zero velocity field should leave state unchanged."""

        def f(x, _t):
            return torch.zeros_like(x)

        heun = Heun(steps=10)
        x1 = heun.integrate(f, sample_tensor_2d, t0=0.0, t1=1.0)
        assert torch.allclose(x1, sample_tensor_2d)

    def test_linear_decay(self):
        """Test dx/dt = -x, solution is x(t) = x0 * exp(-t)."""

        def f(x, _t):
            return -x

        x0 = torch.ones(5)

        heun = Heun(steps=100)
        x1 = heun.integrate(f, x0, t0=0.0, t1=1.0)

        expected = torch.exp(torch.tensor(-1.0)) * x0
        assert torch.allclose(x1, expected, rtol=1e-2)

    def test_oscillator(self):
        """Test simple harmonic oscillator returns to initial state after one period."""

        def f(x, _t):
            v = x[..., 1:]
            dx_dt = -v
            dv_dt = x[..., :1]
            return torch.cat([dx_dt, dv_dt], dim=-1)

        x0 = torch.tensor([1.0, 0.0])

        heun = Heun(steps=200)
        x1 = heun.integrate(f, x0, t0=0.0, t1=2 * 3.14159)

        assert torch.allclose(x1, x0, atol=5e-2)

    def test_custom_steps_override(self, sample_tensor_2d):
        """Steps parameter in integrate should override default."""

        def f(x, _t):
            return torch.zeros_like(x)

        heun = Heun(steps=5)
        x1 = heun.integrate(f, sample_tensor_2d, t0=0.0, t1=1.0, steps=10)

        assert torch.allclose(x1, sample_tensor_2d)

    @pytest.mark.parametrize(
        ("steps", "rtol"),
        [
            (10, 5e-2),
            (100, 1e-2),
            (500, 5e-3),
        ],
        ids=["coarse", "medium", "fine"],
    )
    def test_accuracy_vs_steps(self, steps, rtol):
        """More steps should give better accuracy for linear decay."""

        def f(x, _t):
            return -x

        x0 = torch.ones(5)

        heun = Heun(steps=steps)
        x1 = heun.integrate(f, x0, t0=0.0, t1=1.0)

        expected = torch.exp(torch.tensor(-1.0)) * x0
        assert torch.allclose(x1, expected, rtol=rtol)


class TestHeunIntegrateAugmented:
    """Tests for Heun.integrate_augmented method."""

    def test_constant_augmented(self, sample_tensor_2d):
        """Zero velocity and divergence should leave state unchanged."""

        def f_aug(x, _t):
            v = torch.zeros_like(x)
            div = torch.zeros_like(x[..., 0])
            return v, div

        logp0 = torch.randn(3)

        heun = Heun(steps=10)
        x1, logp1 = heun.integrate_augmented(
            f_aug, sample_tensor_2d, logp0, t0=0.0, t1=1.0
        )

        assert torch.allclose(x1, sample_tensor_2d)
        assert torch.allclose(logp1, logp0)

    def test_linear_augmented(self, sample_tensor_2d):
        """Test augmented integration with linear decay and constant divergence."""

        def f_aug(x, _t):
            v = -x
            div = -torch.ones_like(x[..., 0])
            return v, div

        logp0 = torch.zeros(3)

        heun = Heun(steps=10)
        x1, logp1 = heun.integrate_augmented(
            f_aug, sample_tensor_2d, logp0, t0=0.0, t1=1.0
        )

        expected_x = torch.exp(torch.tensor(-1.0)) * sample_tensor_2d
        expected_logp = -1.0 * torch.ones(3)

        assert torch.allclose(x1, expected_x, rtol=1e-2)
        assert torch.allclose(logp1, expected_logp, rtol=1e-2)


class TestHeunStepOverrides:
    """Step-count overrides passed to the integrate calls."""

    def test_integrate_rejects_negative_steps_override(self):
        heun = Heun(steps=4)

        with pytest.raises(ValueError, match="steps must be positive"):
            heun.integrate(
                lambda x, _t: torch.zeros_like(x),
                torch.ones(3),
                t0=0.0,
                t1=1.0,
                steps=-1,
            )

    def test_integrate_augmented_rejects_negative_steps_override(self):
        heun = Heun(steps=4)

        with pytest.raises(ValueError, match="steps must be positive"):
            heun.integrate_augmented(
                lambda x, _t: (torch.zeros_like(x), torch.zeros(x.shape[0])),
                torch.ones(3, 2),
                torch.zeros(3),
                t0=0.0,
                t1=1.0,
                steps=-1,
            )
