from __future__ import annotations

import pytest
import torch

from nami.schedules.vp import VPSchedule
from nami.solvers.dpm import DPMSolverPP


class TestDPMSolverPPInit:
    def test_basic_init(self):
        solver = DPMSolverPP(steps=12, order=2, skip_type="time_uniform")
        assert solver.steps == 12
        assert solver.order == 2
        assert solver.skip_type == "time_uniform"
        assert solver.requires_steps
        assert solver.supports_rsample
        assert not solver.is_sde

    def test_defaults(self):
        solver = DPMSolverPP()
        assert solver.steps == 20
        assert solver.order == 2
        assert solver.skip_type == "time_uniform"
        assert solver.sigma_min == 1e-12

    @pytest.mark.parametrize("steps", [0, -1], ids=["zero", "neg"])
    def test_invalid_steps(self, steps):
        with pytest.raises(ValueError, match="steps must be positive"):
            DPMSolverPP(steps=steps)

    @pytest.mark.parametrize("order", [0, 3], ids=["zero", "three"])
    def test_invalid_order(self, order):
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            DPMSolverPP(order=order)

    def test_invalid_skip_type(self):
        with pytest.raises(ValueError, match="skip_type must be"):
            DPMSolverPP(skip_type="bad")

    @pytest.mark.parametrize("sigma_min", [0.0, -1e-6], ids=["zero", "neg"])
    def test_invalid_sigma_min(self, sigma_min):
        with pytest.raises(ValueError, match="sigma_min must be positive"):
            DPMSolverPP(sigma_min=sigma_min)


class TestDPMSolverPPIntegrate:
    def test_integrate_constant_field(self, sample_tensor_2d):
        def f(x, _t):
            return torch.zeros_like(x)

        solver = DPMSolverPP(steps=8)
        x1 = solver.integrate(f, sample_tensor_2d, t0=0.0, t1=1.0)
        assert torch.allclose(x1, sample_tensor_2d)

    def test_integrate_linear_field(self):
        """Heun fallback should integrate dx/dt = 1 exactly: x(1) = x(0) + 1."""
        x0 = torch.zeros(4, 2)

        def f(_x, _t):
            return torch.ones_like(_x)

        solver = DPMSolverPP(steps=10)
        x1 = solver.integrate(f, x0, t0=0.0, t1=1.0)
        assert torch.allclose(x1, torch.ones_like(x0), atol=1e-5)

    def test_integrate_explicit_steps_override(self, sample_tensor_2d):
        def f(x, _t):
            return torch.zeros_like(x)

        solver = DPMSolverPP(steps=100)
        x1 = solver.integrate(f, sample_tensor_2d, t0=0.0, t1=1.0, steps=4)
        assert torch.allclose(x1, sample_tensor_2d)

    def test_integrate_augmented_constant(self, sample_tensor_2d):
        def f_aug(x, _t):
            v = torch.zeros_like(x)
            div = torch.zeros_like(x[..., 0])
            return v, div

        solver = DPMSolverPP(steps=8)
        logp0 = torch.randn(sample_tensor_2d.shape[0])
        x1, logp1 = solver.integrate_augmented(
            f_aug, sample_tensor_2d, logp0, t0=0.0, t1=1.0
        )
        assert torch.allclose(x1, sample_tensor_2d)
        assert torch.allclose(logp1, logp0)

    def test_integrate_augmented_linear(self):
        """Augmented Heun integrating dx/dt=1, dlogp/dt=1 over [0,1]."""
        x0 = torch.zeros(3, 2)
        logp0 = torch.zeros(3)

        def f_aug(_x, _t):
            return torch.ones_like(_x), torch.ones(_x.shape[0])

        solver = DPMSolverPP(steps=10)
        x1, logp1 = solver.integrate_augmented(f_aug, x0, logp0, t0=0.0, t1=1.0)
        assert torch.allclose(x1, torch.ones_like(x0), atol=1e-5)
        assert torch.allclose(logp1, torch.ones_like(logp0), atol=1e-5)

    def test_integrate_augmented_explicit_steps(self, sample_tensor_2d):
        def f_aug(x, _t):
            return torch.zeros_like(x), torch.zeros(x.shape[0])

        solver = DPMSolverPP(steps=100)
        logp0 = torch.zeros(sample_tensor_2d.shape[0])
        x1, logp1 = solver.integrate_augmented(
            f_aug, sample_tensor_2d, logp0, t0=0.0, t1=1.0, steps=4
        )
        assert torch.allclose(x1, sample_tensor_2d)
        assert torch.allclose(logp1, logp0)


class TestDPMSolverPPDiffusion:
    @pytest.mark.parametrize("skip_type", ["time_uniform", "logsnr"])
    def test_integrate_diffusion_smoke(self, skip_type):
        solver = DPMSolverPP(steps=10, order=2, skip_type=skip_type)
        schedule = VPSchedule()
        x0 = torch.randn(4, 3)

        def predict_eps(x, _t):
            return torch.zeros_like(x)

        x1 = solver.integrate_diffusion(
            predict_eps, schedule, x0, t0=1.0, t1=1e-3, steps=10
        )

        assert x1.shape == x0.shape
        assert torch.isfinite(x1).all()

    def test_integrate_diffusion_order1(self):
        solver = DPMSolverPP(steps=10, order=1)
        schedule = VPSchedule()
        x0 = torch.randn(4, 3)

        def predict_eps(x, _t):
            return torch.zeros_like(x)

        x1 = solver.integrate_diffusion(
            predict_eps, schedule, x0, t0=1.0, t1=1e-3, steps=10
        )
        assert x1.shape == x0.shape
        assert torch.isfinite(x1).all()

    def test_integrate_diffusion_explicit_steps(self):
        solver = DPMSolverPP(steps=100, order=2)
        schedule = VPSchedule()
        x0 = torch.randn(4, 3)

        def predict_eps(x, _t):
            return torch.zeros_like(x)

        x1 = solver.integrate_diffusion(
            predict_eps, schedule, x0, t0=1.0, t1=1e-3, steps=5
        )
        assert x1.shape == x0.shape
        assert torch.isfinite(x1).all()

    def test_order1_vs_order2_differ(self):
        """Order-2 should give different results from order-1 with a non-trivial model."""
        schedule = VPSchedule()
        torch.manual_seed(42)
        x0 = torch.randn(4, 3)

        def predict_eps(x, _t):
            return 0.1 * x

        x1_o1 = DPMSolverPP(steps=10, order=1).integrate_diffusion(
            predict_eps, schedule, x0, t0=1.0, t1=1e-3
        )
        x1_o2 = DPMSolverPP(steps=10, order=2).integrate_diffusion(
            predict_eps, schedule, x0, t0=1.0, t1=1e-3
        )
        assert not torch.allclose(x1_o1, x1_o2, atol=1e-6)

    def test_integrate_diffusion_3d_input(self):
        """Diffusion integration should handle higher-dimensional inputs."""
        solver = DPMSolverPP(steps=5, order=2)
        schedule = VPSchedule()
        x0 = torch.randn(2, 4, 3)

        def predict_eps(x, _t):
            return torch.zeros_like(x)

        x1 = solver.integrate_diffusion(predict_eps, schedule, x0, t0=1.0, t1=1e-3)
        assert x1.shape == x0.shape
        assert torch.isfinite(x1).all()

    def test_logsnr_increasing_lambda(self):
        """logsnr skip with a schedule where lambda increases over [lo, hi]."""

        class _IncreasingLambdaSchedule:
            """alpha increases, sigma decreases => lambda = log(alpha/sigma) increases."""

            def alpha(self, t):
                return 0.1 + 0.9 * t

            def sigma(self, t):
                return 1.0 - 0.9 * t

        solver = DPMSolverPP(steps=8, order=1, skip_type="logsnr")
        schedule = _IncreasingLambdaSchedule()
        x0 = torch.randn(4, 3)

        def predict_eps(x, _t):
            return torch.zeros_like(x)

        x1 = solver.integrate_diffusion(predict_eps, schedule, x0, t0=0.1, t1=0.9)
        assert x1.shape == x0.shape
        assert torch.isfinite(x1).all()
