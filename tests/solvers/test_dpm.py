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


class TestDPMSolverPPIntegrate:
    def test_integrate_constant_field(self, sample_tensor_2d):
        def f(x, _t):
            return torch.zeros_like(x)

        solver = DPMSolverPP(steps=8)
        x1 = solver.integrate(f, sample_tensor_2d, t0=0.0, t1=1.0)
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
