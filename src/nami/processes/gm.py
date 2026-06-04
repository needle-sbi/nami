"""Generator-Matching process: operator-parameterised ODE / SDE sampling.

The field emits packed operator parameters (drift, optional
diffusion); the operator interprets them at runtime to advance the
state along an ODE or SDE.

References
----------
- Holderrieth et al., *Generator Matching*, 2024.
"""

from __future__ import annotations

import torch

from nami.core.specs import TensorSpec
from nami.distributions.base import expand_distribution, has_rsample
from nami.distributions.normal import StandardNormal
from nami.fields._common import require_event_ndim
from nami.lazy import (
    LazyDistribution,
    LazyField,
    LazyProcess,
    UnconditionalDistribution,
    UnconditionalField,
)
from nami.parameterizations import GeneratorParams, Parameterization
from nami.processes._common import (
    ProcessRuntimeMixin,
    cast_time,
    eager_validate_base_event_shape,
    model_device_dtype,
    resolve_event_shape_override,
    validate_base_event_ndim,
)


class GeneratorMatching(LazyProcess):
    """Generator-matching process driven by a :class:`Parameterization`.

    Args:
        field: Field emitting packed generator parameters.
        solver: ODE or SDE solver.
        parameterization (Parameterization): Must contain a
            :class:`~nami.parameterizations.GeneratorParams` target.
        t0 (float): Initial integration time.
        t1 (float): Final integration time.
        base (LazyDistribution | torch.distributions.Distribution | None):
            Optional base distribution. If ``None``, a standard normal base is
            constructed from ``event_shape`` or the operator event shape.
        event_shape (tuple[int, ...] | None): Event shape used when creating a
            default base distribution. Mutually exclusive with ``spec``.
        spec (TensorSpec | None): Event specification supplying the event
            shape. Mutually exclusive with ``event_shape``.
        validate_args (bool): Whether to validate target and event-shape
            compatibility.

    The process extracts the operator from ``parameterization.target`` and
    applies ``parameterization.output_transform`` to map raw network output
    into valid operator-parameter space.
    """

    def __init__(
        self,
        field,
        solver,
        *,
        parameterization: Parameterization,
        t0: float = 0.0,
        t1: float = 1.0,
        base: LazyDistribution | torch.distributions.Distribution | None = None,
        event_shape: tuple[int, ...] | None = None,
        spec: TensorSpec | None = None,
        validate_args: bool = True,
    ):
        super().__init__()
        self.field = (
            field if isinstance(field, LazyField) else UnconditionalField(field)
        )
        self.parameterization = parameterization
        self.solver = solver
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.base = (
            base
            if base is None or isinstance(base, LazyDistribution)
            else UnconditionalDistribution(base)
        )
        self.event_shape = resolve_event_shape_override(spec, event_shape)
        self.validate_args = bool(validate_args)

        if self.validate_args:
            eager_validate_base_event_shape(self.field, self.base)

    @property
    def operator(self):
        """GeneratorOperator: Operator carried by the parameterization."""
        target = self.parameterization.target
        if not isinstance(target, GeneratorParams):
            msg = (
                "operator is only available when parameterization.target is "
                "GeneratorParams; got "
                f"{type(target).__name__}"
            )
            raise TypeError(msg)
        return target.operator

    def forward(self, c: torch.Tensor | None = None) -> GeneratorMatchingProcess:
        field = self.field(c)
        base = self.base(c) if self.base is not None else None

        if self.validate_args:
            if not isinstance(self.parameterization, Parameterization):
                msg = (
                    "parameterization must be a "
                    "nami.parameterizations.Parameterization instance; use "
                    "generator_prediction(op) to construct one"
                )
                raise TypeError(msg)
            if not isinstance(self.parameterization.target, GeneratorParams):
                msg = (
                    "GeneratorMatching supports only the GeneratorParams "
                    "target; got "
                    f"{type(self.parameterization.target).__name__}.  "
                    "Use FlowMatching for Velocity targets and Diffusion "
                    "for Score / Epsilon / X0."
                )
                raise TypeError(msg)

        target = self.parameterization.target
        if not isinstance(target, GeneratorParams):
            msg = (
                "GeneratorMatching supports only the GeneratorParams "
                "target; got "
                f"{type(target).__name__}."
            )
            raise TypeError(msg)
        operator = target.operator

        if base is None:
            event_shape = self.event_shape or tuple(operator.event_shape)
            if not event_shape:
                msg = "event_shape is required when base is None"
                raise ValueError(msg)
            device, dtype = model_device_dtype(field)
            batch_shape = tuple(c.shape[:-1]) if c is not None else ()
            base = StandardNormal(
                event_shape,
                batch_shape=batch_shape,
                device=device,
                dtype=dtype,
            )
        elif c is not None:
            base = expand_distribution(base, tuple(c.shape[:-1]))

        event_shape = tuple(base.event_shape)
        event_ndim = require_event_ndim(field)

        if self.validate_args:
            validate_base_event_ndim(
                base,
                event_ndim,
                field_event_shape=getattr(field, "event_shape", None),
                message="field.event_ndim does not match base.event_shape",
            )
            if tuple(operator.event_shape) != event_shape:
                msg = "operator.event_shape does not match base.event_shape"
                raise ValueError(msg)

        return GeneratorMatchingProcess(
            field=field,
            parameterization=self.parameterization,
            base=base,
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            context=c,
            validate_args=self.validate_args,
        )


class GeneratorMatchingProcess(ProcessRuntimeMixin):
    def __init__(
        self,
        field,
        base: torch.distributions.Distribution,
        solver,
        *,
        parameterization: Parameterization,
        t0: float = 0.0,
        t1: float = 1.0,
        context: torch.Tensor | None = None,
        validate_args: bool = True,
    ):
        self._field = field
        self._parameterization = parameterization
        # Pre-extract for hot-path: avoids isinstance + attribute walk in
        # every drift/diffusion call.
        if not isinstance(parameterization.target, GeneratorParams):
            msg = (
                "GeneratorMatchingProcess requires a GeneratorParams target; "
                f"got {type(parameterization.target).__name__}"
            )
            raise TypeError(msg)
        self._operator = parameterization.target.operator
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
    def parameterization(self) -> Parameterization:
        return self._parameterization

    @property
    def operator(self):
        return self._operator

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._base.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._base.batch_shape)

    def _projected_params(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        # generator_prediction(op) uses op.project as output_transform.  Keeping
        # the transform in the parameterization lets callers supply a custom
        # projection without changing the process.
        raw = self._field(x, t, context)
        params = self._parameterization.output_transform(raw)
        lead = x.shape[: -len(self.event_shape)] if self.event_shape else x.shape
        self._operator.validate_params(params, leading_shape=lead)
        return params

    def _ode_drift(self, x: torch.Tensor, t: torch.Tensor, context) -> torch.Tensor:
        params = self._projected_params(x, t, context)
        return self._operator.drift(x, t, params)

    def _sde_terms(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self._projected_params(x, t, context)
        drift = self._operator.drift(x, t, params)
        diffusion = self._operator.diffusion(x, t, params)
        return drift, diffusion

    def _integrate_ode(self, x0: torch.Tensor, *, context) -> torch.Tensor:
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = getattr(self._solver, "steps", None)
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps

        def drift(x, t):
            tt = cast_time(t, x)
            return self._ode_drift(x, tt, context)

        return self._solver.integrate(drift, x0, t0=self._t0, t1=self._t1, **kwargs)

    def _integrate_sde(self, x0: torch.Tensor, *, context) -> torch.Tensor:
        steps = getattr(self._solver, "steps", None)
        if steps is None:
            msg = "sde solver requires steps"
            raise ValueError(msg)

        def drift(x, t):
            tt = cast_time(t, x)
            drift_value, _ = self._sde_terms(x, tt, context)
            return drift_value

        def diffusion(x, t):
            tt = cast_time(t, x)
            _, diffusion_value = self._sde_terms(x, tt, context)
            return diffusion_value

        return self._solver.integrate(
            drift,
            diffusion,
            x0,
            t0=self._t0,
            t1=self._t1,
            steps=int(steps),
        )

    def _integrate_jump(self, x0: torch.Tensor, *, context) -> torch.Tensor:
        jump_step = getattr(self._operator, "jump_step", None)
        if jump_step is None:
            msg = "jump runtime requires an operator exposing jump_step"
            raise NotImplementedError(msg)
        steps = getattr(self._solver, "steps", None)
        if steps is None:
            msg = "jump solver requires steps"
            raise ValueError(msg)

        def transition(x, t, dt):
            # State is integer tokens; time stays a Python float and is cast to
            # the field's float dtype inside its time embedding.
            params = self._projected_params(x, t, context)
            return jump_step(x, t, dt, params)

        return self._solver.integrate(
            transition, x0, t0=self._t0, t1=self._t1, steps=int(steps)
        )

    def sample(self, sample_shape=()) -> torch.Tensor:
        x0 = self._base.sample(sample_shape)
        context = self._expand_context(self._context, x0)
        if self._operator.runtime_kind == "ode":
            return self._integrate_ode(x0, context=context)
        if self._operator.runtime_kind == "sde":
            return self._integrate_sde(x0, context=context)
        if self._operator.runtime_kind == "jump":
            return self._integrate_jump(x0, context=context)
        # GeneratorOperator.__init__ validates runtime_kind, so this is
        # unreachable unless a subclass subverts the property.
        msg = f"unsupported runtime kind {self._operator.runtime_kind!r}"  # pragma: no cover
        raise NotImplementedError(msg)  # pragma: no cover

    def rsample(self, sample_shape=()) -> torch.Tensor:
        if self._operator.runtime_kind != "ode":
            msg = "rsample is supported only for ODE generators"
            raise NotImplementedError(msg)
        if not has_rsample(self._base):
            msg = "base distribution does not support rsample"
            raise NotImplementedError(msg)
        if not getattr(self._solver, "supports_rsample", False):
            msg = "solver does not support rsample"
            raise NotImplementedError(msg)

        x0 = self._base.rsample(sample_shape)
        context = self._expand_context(self._context, x0)
        return self._integrate_ode(x0, context=context)
