from __future__ import annotations

import pytest
import torch

from nami import EDMSchedule, VESchedule, VPSchedule
from nami.lazy import UnconditionalField
from nami.processes.diffusion import Diffusion
from nami.solvers import RK4, DPMSolverPP, EulerMaruyama, Heun


def zero_field(x, _t, _c=None):
    return torch.zeros_like(x)


def context_field(x, _t, c):
    return torch.zeros_like(x) + c.sum(dim=-1)


class TestDiffusionProcesses:
    # ---------------------------------------------------------
    # Diffusion process class tests and its integration with various schedules and solvers
    # initial test to check if instantiation and sampling works correctly
    def test_diffusion_smoke(self):
        """Test basic diffusion instantiation and sampling."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        sample = process.sample(sample_shape=(10, 2))
        assert sample.shape == (10, 2)
        assert torch.isfinite(sample).all()

    # ---------------------------------------------------------
    # test different versions of the diffusion process (noise, score, x0)
    @pytest.mark.parametrize("parameterization", ["eps", "score", "x0"])
    def test_diffusion_parameterizations(self, parameterization):
        """Test different diffusion param."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=parameterization,
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        sample = process.sample(sample_shape=(5, 3))
        assert sample.shape == (5, 3)

    # ---------------------------------------------------------
    # test the different noise schedules to make sure shapes are preserved
    @pytest.mark.parametrize("schedule_cls", [EDMSchedule, VESchedule, VPSchedule])
    def test_diffusion_schedules(self, schedule_cls):
        """Test diffusion with different noise schedules."""
        model = UnconditionalField(zero_field)
        schedule = schedule_cls()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        sample = process.sample(sample_shape=(3, 4))
        assert sample.shape == (3, 4)

    # ---------------------------------------------------------
    # test different solvers to make sure shapes are preserved
    @pytest.mark.parametrize("solver_cls", [RK4, EulerMaruyama, Heun, DPMSolverPP])
    def test_diffusion_solvers(self, solver_cls):
        """Test diffusion with different solvers."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = solver_cls()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        sample = process.sample(sample_shape=(4, 2))
        assert sample.shape == (4, 2)

    # ---------------------------------------------------------
    # test diffusion with conditional context to make sure shapes are preserved
    def test_diffusion_with_context(self):
        """Test diffusion with conditional context."""
        model = context_field
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        context = torch.randn(3, 2)
        process = diffusion(context)

        # sample_shape matches context batch size (3) in first dim
        sample = process.sample(sample_shape=(3, 4))
        # expect (sample_shape) + (batch_shape) = (3, 4) + (3,) = (3, 4, 3)
        assert sample.shape == (3, 4, 3)

    # ---------------------------------------------------------
    # test error handling for invalid parameterization to make sure the error is raised
    def test_diffusion_invalid_parameterization(self):
        """Test error handling for invalid param."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="invalid", event_shape=()
        )

        with pytest.raises(ValueError, match="parameterization must be"):
            diffusion()

    # ---------------------------------------------------------
    # test error handling for missing event_shape to make sure the error is raised
    def test_diffusion_missing_event_shape(self):
        """Test error handling for missing event_shape."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(model, schedule, solver, parameterization="eps")
        with pytest.raises(
            ValueError, match="event_shape is required when base is None"
        ):
            diffusion()

    # ---------------------------------------------------------
    # test error handling for shape mismatches to make sure the error is raised
    def test_diffusion_event_shape_mismatch(self):
        """Test error handling for shape mismatches."""
        # model expects 2D but give base of 1D

        class MockModel(torch.nn.Module):
            event_ndim = 2

            def forward(self, x, _t, _c=None):
                return x

        model = UnconditionalField(MockModel())
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(2,)
        )
        # event_shape has len 1, event_ndim is 2 -> Mismatch

        with pytest.raises(
            ValueError, match=r"model\.event_ndim does not match base\.event_shape"
        ):
            diffusion()

    # ---------------------------------------------------------
    # test rsample method for ODE solvers to make sure shapes are preserved
    def test_diffusion_rsample_ode(self):
        """Test rsample method for ODE solvers."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        # rsample should work for ODE solvers
        sample = process.rsample(sample_shape=(5, 2))
        assert sample.shape == (5, 2)

    # ---------------------------------------------------------
    # test rsample fails for non-ODE solvers to make sure the error is raised
    def test_diffusion_rsample_non_ode_fails(self):
        """Test rsample fails for non-ODE solvers."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = EulerMaruyama()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        with pytest.raises(
            NotImplementedError, match="rsample is supported only for ODE solvers"
        ):
            process.rsample(sample_shape=(3, 2))

    # ---------------------------------------------------------
    # test edge case with minimal steps to make sure shapes are preserved
    def test_diffusion_zero_steps(self):
        """Test edge case with minimal steps."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun(steps=1)

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        sample = process.sample(sample_shape=(2, 3))
        assert sample.shape == (2, 3)

    # ---------------------------------------------------------
    # guidance_fn coverage
    def test_sample_with_guidance_fn_ode(self):
        """guidance_fn should be called during ODE sampling."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):
            calls.append(1)
            return eps

        sample = process.sample(sample_shape=(4, 2), guidance_fn=guidance)
        assert sample.shape == (4, 2)
        assert len(calls) > 0

    def test_sample_with_guidance_fn_sde(self):
        """guidance_fn should be called during SDE sampling."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = EulerMaruyama()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):
            calls.append(1)
            return eps

        sample = process.sample(sample_shape=(4, 2), guidance_fn=guidance)
        assert sample.shape == (4, 2)
        assert len(calls) > 0

    def test_sample_with_guidance_fn_dpm(self):
        """guidance_fn should be called via integrate_diffusion fast path."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = DPMSolverPP(steps=5)

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):
            calls.append(1)
            return eps

        sample = process.sample(sample_shape=(4, 2), guidance_fn=guidance)
        assert sample.shape == (4, 2)
        assert len(calls) > 0

    # ---------------------------------------------------------
    # rsample coverage
    def test_rsample_with_dpm_solver(self):
        """rsample should work through DPMSolverPP integrate_diffusion path."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = DPMSolverPP(steps=5)

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        sample = process.rsample(sample_shape=(3, 2))
        assert sample.shape == (3, 2)

    def test_rsample_with_guidance_fn(self):
        """rsample should forward guidance_fn."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):
            calls.append(1)
            return eps

        sample = process.rsample(sample_shape=(3, 2), guidance_fn=guidance)
        assert sample.shape == (3, 2)
        assert len(calls) > 0

    def test_rsample_no_rsample_solver_fails(self):
        """rsample should fail if solver doesn't support rsample."""

        class _NoRsampleSolver:
            is_sde = False
            supports_rsample = False
            requires_steps = True
            steps = 5

            def integrate(self, f, x0, *, t0, t1, **kw):
                return x0

        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = _NoRsampleSolver()

        diffusion = Diffusion(
            model, schedule, solver, parameterization="eps", event_shape=(), t1=1e-3
        )
        process = diffusion()

        with pytest.raises(NotImplementedError, match="solver does not support rsample"):
            process.rsample(sample_shape=(3, 2))
