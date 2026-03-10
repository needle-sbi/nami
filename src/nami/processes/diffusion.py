from __future__ import annotations

import torch

from ..distributions.base import expand_distribution, has_rsample
from ..distributions.normal import StandardNormal
from ..fields.diffusion import _expand_like, eps_to_score, score_to_eps
from ..lazy import (
    LazyDistribution,
    LazyField,
    UnconditionalDistribution,
    UnconditionalField,
)


class Diffusion(LazyDistribution):
    def __init__(
        self,
        model,
        schedule,
        solver,
        *,
        parameterization: str,
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
            device, dtype = _model_device_dtype(model)
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

        event_shape = tuple(base.event_shape)
        event_ndim = getattr(model, "event_ndim", None)
        if self.validate_args:
            if self.parameterization not in {"eps", "score", "x0"}:
                msg = "parameterization must be 'eps', 'score', or 'x0'"
                raise ValueError(msg)
            if event_ndim is not None and len(event_shape) != event_ndim:
                msg = "model.event_ndim does not match base.event_shape"
                raise ValueError(msg)

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


class DiffusionProcess:
    def __init__(
        self,
        model,
        schedule,
        solver,
        *,
        parameterization: str,
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

    def _cast_time(self, t: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, device=like.device, dtype=like.dtype)

    def _expand_context(
        self, c: torch.Tensor | None, target: torch.Tensor
    ) -> torch.Tensor | None:
        """Expand context to match target's sample dimensions."""
        if c is None:
            return None
        # c has shape: batch_shape + (context_dim,)
        # target has shape: sample_shape + batch_shape + event_shape
        # We need c to have shape: sample_shape + batch_shape + (context_dim,)
        event_ndim = len(self.event_shape)
        # Number of leading sample dims to prepend:
        #   target.ndim = len(sample) + len(batch) + event_ndim
        #   c.ndim      = len(batch) + 1  (the +1 is context_dim)
        #   n_expand    = len(sample) = target.ndim - event_ndim - c.ndim + 1
        n_expand = target.ndim - event_ndim - c.ndim + 1
        if n_expand > 0:
            for _ in range(n_expand):
                c = c.unsqueeze(0)
            # target.shape[:target.ndim - event_ndim] == sample_shape + batch_shape
            c = c.expand(*target.shape[: target.ndim - event_ndim], c.shape[-1])
        return c

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

        out = self._model(x, tt, context)
        if self._parameterization == "eps":
            eps = out
        elif self._parameterization == "score":
            eps = score_to_eps(out, sigma)
        elif self._parameterization == "x0":
            # Expand alpha/sigma for broadcasting with x
            eps = (x - _expand_like(alpha, x) * out) / _expand_like(sigma, x)
        else:  # pragma: no cover — factory validates
            msg = "unknown parameterization"
            raise ValueError(msg)

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
                tt = self._cast_time(t, x)
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
            tt = self._cast_time(t, x)
            eps, _, sigma = self._predict_eps(x, tt, context)
            if guidance_fn is not None:
                eps = guidance_fn(x, tt, eps)
            score = eps_to_score(eps, sigma)
            g = _expand_like(self._schedule.diffusion(tt), x)
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
            tt = self._cast_time(t, x)
            eps, _, sigma = self._predict_eps(x, tt, context)
            if guidance_fn is not None:
                eps = guidance_fn(x, tt, eps)
            score = eps_to_score(eps, sigma)
            g = _expand_like(self._schedule.diffusion(tt), x)
            f = self._schedule.drift(x, tt)
            return f - (g**2) * score

        def diffusion(t):
            return _expand_like(self._schedule.diffusion(self._cast_time(t, x0)), x0)

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


def _model_device_dtype(model) -> tuple[torch.device | None, torch.dtype | None]:
    if not hasattr(model, "parameters"):
        return None, None
    for p in model.parameters():
        return p.device, p.dtype
    return None, None
