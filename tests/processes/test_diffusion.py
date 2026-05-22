from __future__ import annotations

import pytest
import torch

from nami import EDMSchedule, VESchedule, VPSchedule
from nami.diffusion import expand_like
from nami.interpolants import epsilon_prediction, score_prediction, x0_prediction
from nami.lazy import UnconditionalField
from nami.parameterizations import Parameterization, Velocity
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        sample = process.sample(sample_shape=(10, 2))
        assert sample.shape == (10, 2)
        assert torch.isfinite(sample).all()

    # ---------------------------------------------------------
    # test different versions of the diffusion process (noise, score, x0)
    @pytest.mark.parametrize(
        "factory",
        [epsilon_prediction, score_prediction, x0_prediction],
        ids=["epsilon", "score", "x0"],
    )
    def test_diffusion_parameterizations(self, factory):
        """Sampling works under all three target choices.

        Factories replace the legacy ``parameterization="eps"|"score"|"x0"``
        flag.  The Process pattern-matches on ``parameterization.target``
        to dispatch the right eps-conversion.
        """
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=factory(schedule),
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        context = torch.randn(3, 2)
        process = diffusion(context)

        # sample_shape matches context batch size (3) in first dim
        sample = process.sample(sample_shape=(3, 4))
        # expect (sample_shape) + (batch_shape) = (3, 4) + (3,) = (3, 4, 3)
        assert sample.shape == (3, 4, 3)

    # ---------------------------------------------------------
    def test_diffusion_rejects_legacy_string_flag(self):
        """The legacy string flag was removed; passing one fails clearly.

        Pins the API break: callers who pass ``parameterization="eps"``
        get an explicit migration message rather than a silent
        ``isinstance`` False that propagates to a confusing later error.
        """
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization="eps",  # type: ignore[arg-type]
            event_shape=(),
        )

        with pytest.raises(TypeError, match="Parameterization"):
            diffusion()

    def test_diffusion_rejects_unsupported_target(self):
        """Diffusion supports only Epsilon / Score / X0 targets.

        Velocity and GeneratorParams targets must fail at construction
        time with a clear message — pattern-match fallthrough errors
        appear too late.
        """
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=Parameterization(target=Velocity()),
            event_shape=(),
        )

        with pytest.raises(TypeError, match="Diffusion supports targets"):
            diffusion()

    # ---------------------------------------------------------
    # test error handling for missing event_shape to make sure the error is raised
    def test_diffusion_missing_event_shape(self):
        """Test error handling for missing event_shape."""
        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model, schedule, solver, parameterization=epsilon_prediction(schedule)
        )
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(2,),
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):  # noqa: ARG001
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):  # noqa: ARG001
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):  # noqa: ARG001
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
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
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        calls = []

        def guidance(x, t, eps):  # noqa: ARG001
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

            def integrate(self, f, x0, *, t0, t1, **kw):  # noqa: ARG002
                return x0

        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = _NoRsampleSolver()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        with pytest.raises(
            NotImplementedError, match="solver does not support rsample"
        ):
            process.rsample(sample_shape=(3, 2))

    def test_rsample_no_rsample_base_fails(self):
        """rsample should fail if base distribution lacks rsample."""

        class _NoRsampleDist(torch.distributions.Distribution):
            has_rsample = False

            def __init__(self):
                super().__init__(batch_shape=(), event_shape=(), validate_args=False)

            def sample(self, sample_shape=()):
                return torch.randn(sample_shape)

        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = Heun()
        base = _NoRsampleDist()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            base=base,
            t1=1e-3,
            validate_args=False,
        )
        process = diffusion()

        with pytest.raises(
            NotImplementedError, match="base distribution does not support rsample"
        ):
            process.rsample(sample_shape=(3, 2))

    def test_ode_solver_requires_steps_but_has_none(self):
        """ODE solver with requires_steps=True but no steps attr should raise."""

        class _SteplessSolver:
            is_sde = False
            requires_steps = True

            def integrate(self, f, x0, *, t0, t1, **kw):  # noqa: ARG002
                return x0

        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = _SteplessSolver()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        with pytest.raises(ValueError, match="solver requires steps"):
            process.sample(sample_shape=(2,))

    def test_sde_solver_requires_steps_but_has_none(self):
        """SDE solver without steps attr should raise."""

        class _SteplessSDE:
            is_sde = True
            requires_steps = False

            def integrate(self, drift, diffusion, x0, *, t0, t1, steps):  # noqa: ARG002
                return x0

        model = UnconditionalField(zero_field)
        schedule = VPSchedule()
        solver = _SteplessSDE()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        with pytest.raises(ValueError, match="sde solver requires steps"):
            process.sample(sample_shape=(2,))

    def test_diffusion_with_custom_base_and_context(self):
        """Custom base distribution with context should expand correctly."""
        model = context_field
        schedule = VPSchedule()
        solver = Heun()
        base = torch.distributions.Normal(torch.zeros(3), torch.ones(3))

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            base=base,
            t1=1e-3,
        )
        context = torch.randn(3, 2)
        process = diffusion(context)

        sample = process.sample(sample_shape=(4,))
        assert torch.isfinite(sample).all()

    def test_diffusion_with_nn_module_model(self):
        """nn.Module model should have device/dtype detected from parameters."""

        class _ZeroModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x, _t, _c=None):
                return torch.zeros_like(x)

        model = UnconditionalField(_ZeroModule())
        schedule = VPSchedule()
        solver = Heun()

        diffusion = Diffusion(
            model,
            schedule,
            solver,
            parameterization=epsilon_prediction(schedule),
            event_shape=(),
            t1=1e-3,
        )
        process = diffusion()

        sample = process.sample(sample_shape=(4, 2))
        assert sample.shape == (4, 2)
        assert torch.isfinite(sample).all()

    def test_three_target_choices_produce_identical_samples(self):
        schedule = VPSchedule()
        solver = Heun(steps=20)

        # The "true" eps predicted by all three networks (in their native
        # parameterisations).  Constant across (x, t) so we can express
        # each field's output as a closed-form transform.
        torch.manual_seed(0)
        y_eps_value = 0.3 * torch.randn(1)  # broadcast scalar
        y_eps = float(y_eps_value)

        def eps_field(x, _t, _c=None):
            return torch.full_like(x, y_eps)

        def score_field(x, t, _c=None):
            sigma = expand_like(schedule.sigma(t), x)
            return torch.full_like(x, y_eps).neg().div_(sigma)

        def x0_field(x, t, _c=None):
            alpha = expand_like(schedule.alpha(t), x)
            sigma = expand_like(schedule.sigma(t), x)
            return (x - sigma * y_eps) / alpha

        common = {
            "schedule": schedule,
            "solver": solver,
            "event_shape": (),
            "t1": 1e-3,
        }

        torch.manual_seed(42)
        s_eps = Diffusion(
            UnconditionalField(eps_field),
            parameterization=epsilon_prediction(schedule),
            **common,
        )().sample(sample_shape=(8, 3))

        torch.manual_seed(42)
        s_score = Diffusion(
            UnconditionalField(score_field),
            parameterization=score_prediction(schedule),
            **common,
        )().sample(sample_shape=(8, 3))

        torch.manual_seed(42)
        s_x0 = Diffusion(
            UnconditionalField(x0_field),
            parameterization=x0_prediction(schedule),
            **common,
        )().sample(sample_shape=(8, 3))

        # The three trajectories share the same eps prediction at every
        # integration step, so with the same RNG seed they must produce
        # the same final samples up to floating-point reordering.
        assert torch.allclose(s_eps, s_score, atol=1e-5, rtol=1e-5), (
            "Score parameterisation diverged from Epsilon — score_to_eps "
            "or the dispatch broke"
        )
        assert torch.allclose(s_eps, s_x0, atol=1e-5, rtol=1e-5), (
            "X0 parameterisation diverged from Epsilon — x0_to_eps "
            "or the dispatch broke"
        )
