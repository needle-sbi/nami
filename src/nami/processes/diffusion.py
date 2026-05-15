"""Diffusion process: schedule-driven sampling from Score / Epsilon / X0 / V targets.

Probability-flow ODE and reverse-time SDE samplers built on top of a
:class:`NoiseSchedule` and a :class:`Parameterization`. Target
dispatch is exhaustive over the diffusion target sum-type.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456).
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020
  (arXiv:2006.11239).
- Salimans & Ho, *Progressive Distillation for Fast Sampling of
  Diffusion Models*, 2022 (arXiv:2202.00512) — v-prediction.
"""

from __future__ import annotations

import torch

from nami.diffusion import (
    eps_to_score,
    expand_like,
    score_to_eps,
    v_to_eps,
    x0_to_eps,
)
from nami.distributions.base import expand_distribution, has_rsample
from nami.distributions.normal import StandardNormal
from nami.lazy import (
    LazyDistribution,
    LazyField,
    LazyProcess,
    UnconditionalDistribution,
    UnconditionalField,
)
from nami.parameterizations import X0, Epsilon, Parameterization, Score, VPrediction
from nami.processes._common import (
    ProcessRuntimeMixin,
    cast_time,
    model_device_dtype,
    validate_base_event_ndim,
)


class Diffusion(LazyProcess):
    r"""Diffusion process driven by a :class:`Parameterization`.

    Args:
        model: Field emitting the target specified by ``parameterization``.
        schedule: Noise schedule providing ``\alpha(t)``, ``\sigma(t)``, drift,
            and diffusion coefficients.
        solver: ODE or SDE solver.
        parameterization (Parameterization): Diffusion target and output
            projection. Supported targets are ``Epsilon``, ``Score``, ``X0``,
            and ``VPrediction``.
        t0 (float): Initial integration time, usually the noise endpoint.
        t1 (float): Final integration time, usually the data endpoint.
        base (LazyDistribution | torch.distributions.Distribution | None):
            Optional base distribution.
        event_shape (tuple[int, ...] | None): Event shape used when creating a
            default base distribution.
        validate_args (bool): Whether to validate target and event-shape
            compatibility.

    The process converts the model output into ``\epsilon`` before applying the
    probability-flow ODE or reverse-time SDE update.
    """

    def __init__(
        self,
        model,
        schedule,
        solver,
        *,
        parameterization: Parameterization,
        # Diffusion uses the schedule's forward-SDE convention:
        # t0=1.0 is the noise endpoint and t1=0.0 is the data endpoint.
        t0: float = 1.0,
        t1: float = 0.0,
        base: LazyDistribution | torch.distributions.Distribution | None = None,
        event_shape: tuple[int, ...] | None = None,
        validate_args: bool = True,
    ):
        super().__init__()
        self.model = (
            model if isinstance(model, LazyField) else UnconditionalField(model)
        )
        self.schedule = schedule
        self.solver = solver
        self.parameterization = parameterization
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.base = (
            base
            if base is None or isinstance(base, LazyDistribution)
            else UnconditionalDistribution(base)
        )
        self.event_shape = event_shape
        self.validate_args = bool(validate_args)

    def forward(self, c: torch.Tensor | None = None) -> DiffusionProcess:
        model = self.model(c)
        base = self.base(c) if self.base is not None else None

        if base is None:
            if self.event_shape is None:
                msg = "event_shape is required when base is None"
                raise ValueError(msg)
            device, dtype = model_device_dtype(model)
            batch_shape = tuple(c.shape[:-1]) if c is not None else ()
            base = StandardNormal(
                self.event_shape, batch_shape=batch_shape, device=device, dtype=dtype
            )
            base_scale = self.schedule.sigma(
                torch.as_tensor(self.t0, device=device, dtype=dtype)
            )
        else:
            if c is not None:
                base = expand_distribution(base, tuple(c.shape[:-1]))
            base_scale = None

        event_ndim = getattr(model, "event_ndim", None)
        if self.validate_args:
            if not isinstance(self.parameterization, Parameterization):
                msg = (
                    "parameterization must be a nami.parameterizations.Parameterization "
                    "instance; use epsilon_prediction(schedule), score_prediction(schedule), "
                    "or x0_prediction(schedule)"
                )
                raise TypeError(msg)
            if not isinstance(
                self.parameterization.target, (Epsilon, Score, X0, VPrediction)
            ):
                msg = (
                    "Diffusion supports targets Epsilon, Score, X0, VPrediction; got "
                    f"{type(self.parameterization.target).__name__}"
                )
                raise TypeError(msg)
            if event_ndim is not None:
                validate_base_event_ndim(
                    base,
                    int(event_ndim),
                    message="model.event_ndim does not match base.event_shape",
                )

        return DiffusionProcess(
            model=model,
            schedule=self.schedule,
            solver=self.solver,
            parameterization=self.parameterization,
            t0=self.t0,
            t1=self.t1,
            base=base,
            base_scale=base_scale,
            context=c,
            validate_args=self.validate_args,
        )


class DiffusionProcess(ProcessRuntimeMixin):
    def __init__(
        self,
        model,
        schedule,
        solver,
        *,
        parameterization: Parameterization,
        # See note on Diffusion.__init__ above — diffusion retains the
        # diffusion-convention t-direction.
        t0: float = 1.0,
        t1: float = 0.0,
        base: torch.distributions.Distribution,
        base_scale: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        validate_args: bool = True,
    ):
        self._model = model
        self._schedule = schedule
        self._solver = solver
        self._parameterization = parameterization
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._base = base
        self._base_scale = base_scale
        self._context = context
        self._validate_args = bool(validate_args)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._base.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._base.batch_shape)

    def _is_ode(self) -> bool:
        # Explicit check: SDE solvers must declare is_sde=True
        # This avoids conflating "can reparameterize" with "is ODE vs SDE"
        return not getattr(self._solver, "is_sde", False)

    def _steps(self) -> int | None:
        if hasattr(self._solver, "steps"):
            return int(self._solver.steps)
        return None

    def _predict_eps(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lead = x.shape[: -len(self.event_shape)] if self.event_shape else x.shape
        tt = t.expand(lead)
        alpha = self._schedule.alpha(tt)
        sigma = self._schedule.sigma(tt)

        raw = self._model(x, tt, context)
        out = self._parameterization.output_transform(raw)

        # Pattern-match dispatch on the typed target.  Adding a diffusion
        # target requires adding one conversion arm here.
        match self._parameterization.target:
            case Epsilon():
                eps = out
            case Score():
                eps = score_to_eps(out, sigma)
            case X0():
                eps = x0_to_eps(x, out, alpha, sigma)
            case VPrediction():
                eps = v_to_eps(x, out, alpha, sigma)
            case _:  # pragma: no cover — Diffusion.forward validates
                msg = (
                    f"Diffusion supports targets Epsilon, Score, X0, VPrediction; got "
                    f"{type(self._parameterization.target).__name__}"
                )
                raise TypeError(msg)

        return eps, alpha, sigma

    def _apply_base_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self._base_scale is None:
            return x
        return x * self._base_scale

    def _integrate_ode(
        self,
        x0: torch.Tensor,
        *,
        context: torch.Tensor | None,
        guidance_fn,
    ) -> torch.Tensor:
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = self._steps()
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps

        if hasattr(self._solver, "integrate_diffusion"):

            def predict_eps(x, t):
                tt = cast_time(t, x)
                eps, _, _ = self._predict_eps(x, tt, context)
                if guidance_fn is not None:
                    eps = guidance_fn(x, tt, eps)
                return eps

            return self._solver.integrate_diffusion(
                predict_eps,
                self._schedule,
                x0,
                t0=self._t0,
                t1=self._t1,
                **kwargs,
            )

        def drift(x, t):
            tt = cast_time(t, x)
            eps, _, sigma = self._predict_eps(x, tt, context)
            if guidance_fn is not None:
                eps = guidance_fn(x, tt, eps)
            score = eps_to_score(eps, sigma)
            g = expand_like(self._schedule.diffusion(tt), x)
            f = self._schedule.drift(x, tt)
            return f - 0.5 * (g**2) * score

        return self._solver.integrate(drift, x0, t0=self._t0, t1=self._t1, **kwargs)

    def sample(self, sample_shape=(), *, guidance_fn=None) -> torch.Tensor:
        x0 = self._base.sample(sample_shape)
        x0 = self._apply_base_scale(x0)
        context = self._expand_context(self._context, x0)

        if self._is_ode():
            return self._integrate_ode(x0, context=context, guidance_fn=guidance_fn)

        def drift(x, t):
            tt = cast_time(t, x)
            eps, _, sigma = self._predict_eps(x, tt, context)
            if guidance_fn is not None:
                eps = guidance_fn(x, tt, eps)
            score = eps_to_score(eps, sigma)
            g = expand_like(self._schedule.diffusion(tt), x)
            f = self._schedule.drift(x, tt)
            return f - (g**2) * score

        def diffusion(t):
            return expand_like(self._schedule.diffusion(cast_time(t, x0)), x0)

        steps = self._steps()
        if steps is None:
            msg = "sde solver requires steps"
            raise ValueError(msg)
        return self._solver.integrate(
            drift, diffusion, x0, t0=self._t0, t1=self._t1, steps=steps
        )

    def rsample(self, sample_shape=(), *, guidance_fn=None) -> torch.Tensor:
        if not self._is_ode():
            msg = "rsample is supported only for ODE solvers"
            raise NotImplementedError(msg)
        if not has_rsample(self._base):
            msg = "base distribution does not support rsample"
            raise NotImplementedError(msg)
        if not getattr(self._solver, "supports_rsample", False):
            msg = "solver does not support rsample"
            raise NotImplementedError(msg)

        x0 = self._base.rsample(sample_shape)
        x0 = self._apply_base_scale(x0)
        context = self._expand_context(self._context, x0)
        return self._integrate_ode(x0, context=context, guidance_fn=guidance_fn)
