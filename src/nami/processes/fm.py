from __future__ import annotations

import torch

from ..distributions.base import expand_distribution, has_rsample
from ..lazy import (
    LazyDistribution,
    LazyField,
    UnconditionalDistribution,
    UnconditionalField,
)


class FlowMatching(LazyDistribution):
    def __init__(
        self,
        field,
        base,
        solver,
        *,
        t0: float = 1.0,
        t1: float = 0.0,
        event_ndim: int | None = None,
        validate_args: bool = True,
    ):
        super().__init__()
        self.field = (
            field if isinstance(field, LazyField) else UnconditionalField(field)
        )
        self.base = (
            base
            if isinstance(base, LazyDistribution)
            else UnconditionalDistribution(base)
        )
        self.solver = solver
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.event_ndim = event_ndim
        self.validate_args = bool(validate_args)

    def forward(self, c: torch.Tensor | None = None) -> FlowMatchingProcess:
        field = self.field(c)
        base = self.base(c)

        if c is not None:
            base = expand_distribution(base, tuple(c.shape[:-1]))

        # Use explicit None check to support event_ndim=0 (scalar events)
        event_ndim = getattr(field, "event_ndim", None)
        if event_ndim is None:
            event_ndim = self.event_ndim
        if event_ndim is None:
            msg = "event_ndim must be provided or exposed by field"
            raise ValueError(msg)

        if self.validate_args and len(base.event_shape) != event_ndim:
            msg = "base.event_shape does not match field.event_ndim"
            raise ValueError(msg)

        return FlowMatchingProcess(
            field=field,
            base=base,
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            context=c,
            validate_args=self.validate_args,
        )


class FlowMatchingProcess:
    def __init__(
        self,
        field,
        base: torch.distributions.Distribution,
        solver,
        *,
        t0: float = 1.0,
        t1: float = 0.0,
        context: torch.Tensor | None = None,
        validate_args: bool = True,
    ):
        self._field = field
        self._base = base
        self._solver = solver
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._context = context
        self._validate_args = bool(validate_args)

    @property
    def field(self):
        return self._field

    @property
    def base(self):
        return self._base

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

    def _integrate(self, f, x0: torch.Tensor, *, t0: float, t1: float) -> torch.Tensor:
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = getattr(self._solver, "steps", None)
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps
        return self._solver.integrate(f, x0, t0=t0, t1=t1, **kwargs)

    def _integrate_augmented(
        self, f_aug, x0: torch.Tensor, logp0: torch.Tensor, *, t0: float, t1: float
    ):
        if not hasattr(self._solver, "integrate_augmented"):
            msg = "solver does not support augmented integration"
            raise NotImplementedError(msg)
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = getattr(self._solver, "steps", None)
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps
        return self._solver.integrate_augmented(
            f_aug, x0, logp0, t0=t0, t1=t1, **kwargs
        )

    def sample(self, sample_shape=()) -> torch.Tensor:
        z = self._base.sample(sample_shape)
        context = self._expand_context(self._context, z)

        def f(x, t):
            tt = self._cast_time(t, x)
            return self._field(x, tt, context)

        return self._integrate(f, z, t0=self._t0, t1=self._t1)

    def rsample(self, sample_shape=()) -> torch.Tensor:
        if not has_rsample(self._base):
            msg = "base distribution does not support rsample"
            raise NotImplementedError(msg)
        if not getattr(self._solver, "supports_rsample", False):
            msg = "solver does not support rsample"
            raise NotImplementedError(msg)

        z = self._base.rsample(sample_shape)
        context = self._expand_context(self._context, z)

        def f(x, t):
            tt = self._cast_time(t, x)
            return self._field(x, tt, context)

        return self._integrate(f, z, t0=self._t0, t1=self._t1)

    def log_prob(self, x: torch.Tensor, *, estimator=None) -> torch.Tensor:
        """Evaluate log-density via change of variables.

        Callers should usually pass ``estimator=...`` unless the field
        implements ``call_and_divergence`` itself. We do not auto-select a
        divergence estimator here because the tradeoff is model-dependent:
        exact traces are deterministic but can be expensive, while
        Hutchinson-style estimators scale better but add stochasticity.
        """
        event_ndim = getattr(self._field, "event_ndim", None)
        if event_ndim is None:
            msg = "field.event_ndim is required"
            raise ValueError(msg)
        lead = x.shape[:-event_ndim] if event_ndim else x.shape
        logp0 = torch.zeros(lead, device=x.device, dtype=x.dtype)
        context = self._expand_context(self._context, x)

        def f_aug(xi, t):
            tt = self._cast_time(t, xi)
            if estimator is not None:
                v = self._field(xi, tt, context)
                div = estimator(self._field, xi, tt, context)
            else:
                call_and_divergence = getattr(self._field, "call_and_divergence", None)
                if call_and_divergence is None:
                    msg = (
                        "log_prob requires either `estimator=...` or a field "
                        "implementing `call_and_divergence(x, t, c)`"
                    )
                    raise TypeError(msg)
                try:
                    v, div = call_and_divergence(xi, tt, context)
                except NotImplementedError:
                    msg = (
                        "log_prob requires either `estimator=...` or a field "
                        "implementing `call_and_divergence(x, t, c)`"
                    )
                    raise TypeError(msg) from None
            return v, -div

        z, delta = self._integrate_augmented(f_aug, x, logp0, t0=self._t1, t1=self._t0)
        return self._base.log_prob(z) + delta
