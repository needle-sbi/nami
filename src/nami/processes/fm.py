"""Flow-Matching process: velocity-field ODE sampling and log-density.

Wraps a velocity field plus base distribution plus ODE solver, with
optional augmented-state integration for change-of-variables
log-density. Only the Velocity target is supported here; converting
score / epsilon to velocity needs a schedule the FM process does not
carry.

References
----------
- Lipman et al., *Flow Matching for Generative Modeling*, 2022
  (arXiv:2210.02747).
- Liu et al., *Rectified Flow*, 2022 (arXiv:2209.03003).
- Chen et al., *Neural Ordinary Differential Equations*, 2018
  (arXiv:1806.07366) — augmented-state log-density.
- Grathwohl et al., *FFJORD*, 2018 (arXiv:1810.01367) — divergence
  estimator integration into the augmented state.
"""

from __future__ import annotations

import torch

from nami.distributions.base import expand_distribution, has_rsample
from nami.fields._common import require_event_ndim
from nami.interpolants.linear import velocity_prediction
from nami.lazy import (
    LazyDistribution,
    LazyField,
    LazyProcess,
    UnconditionalDistribution,
    UnconditionalField,
)
from nami.parameterizations import Parameterization, Velocity
from nami.processes._common import (
    ProcessRuntimeMixin,
    TransformedField,
    cast_time,
    resolve_event_ndim,
    validate_base_event_ndim,
)


class FlowMatching(LazyProcess):
    r"""Flow-matching process driven by a :class:`Parameterization`.

    Unlike :class:`~nami.processes.diffusion.Diffusion`, FM has no
    algebraic dispatch over multiple targets — it only supports
    :class:`~nami.parameterizations.Velocity`, because converting score
    or ``\epsilon`` to velocity requires a schedule the FM Process does not carry.
    The ``parameterization`` kwarg therefore exists primarily for API
    consistency with ``Diffusion`` and to provide the
    ``output_transform`` slot, so a trained field whose raw output
    needs projection (e.g. through a constraint) can be sampled
    without a wrapper class.

    The default, ``velocity_prediction()``, has identity
    ``output_transform`` and reproduces the legacy behaviour exactly.

    .. note::

       ``output_transform`` is applied inside the integration path
       (``sample`` / ``rsample`` / ``log_prob``).  Estimator-based
       ``log_prob`` automatically differentiates the *transformed*
       velocity (via the internal ``TransformedField`` adapter) so
       the change-of-variables identity holds for any
       ``output_transform``.  The bundled ``call_and_divergence`` field
       method path rejects non-identity transforms with an explicit
       error — its divergence is for the raw output and would mix two
       different velocities in the density bookkeeping.
    """

    def __init__(
        self,
        field,
        base,
        solver,
        *,
        parameterization: Parameterization | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
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
        self.parameterization = (
            parameterization if parameterization is not None else velocity_prediction()
        )
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.event_ndim = event_ndim
        self.validate_args = bool(validate_args)

    def forward(self, c: torch.Tensor | None = None) -> FlowMatchingProcess:
        field = self.field(c)
        base = self.base(c)

        if c is not None:
            base = expand_distribution(base, tuple(c.shape[:-1]))

        event_ndim = resolve_event_ndim(field, self.event_ndim)

        if self.validate_args:
            if not isinstance(self.parameterization, Parameterization):
                msg = (
                    "parameterization must be a "
                    "nami.parameterizations.Parameterization instance"
                )
                raise TypeError(msg)
            if not isinstance(self.parameterization.target, Velocity):
                msg = (
                    "FlowMatching supports only the Velocity target; got "
                    f"{type(self.parameterization.target).__name__}.  "
                    "Use Diffusion for Score / Epsilon / X0 targets — "
                    "those conversions require a NoiseSchedule that the "
                    "FM Process does not carry."
                )
                raise TypeError(msg)
            validate_base_event_ndim(
                base,
                event_ndim,
                message="base.event_shape does not match field.event_ndim",
            )

        return FlowMatchingProcess(
            field=field,
            base=base,
            solver=self.solver,
            parameterization=self.parameterization,
            t0=self.t0,
            t1=self.t1,
            context=c,
            validate_args=self.validate_args,
        )


class FlowMatchingProcess(ProcessRuntimeMixin):
    def __init__(
        self,
        field,
        base: torch.distributions.Distribution,
        solver,
        *,
        parameterization: Parameterization | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
        context: torch.Tensor | None = None,
        validate_args: bool = True,
    ):
        self._field = field
        self._base = base
        self._solver = solver
        self._parameterization = (
            parameterization if parameterization is not None else velocity_prediction()
        )
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

    def _velocity(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None
    ) -> torch.Tensor:
        """Field call composed with ``output_transform``.

        Centralises the transform so every integration path picks up
        non-identity transforms automatically.
        """
        return self._parameterization.output_transform(self._field(x, t, context))

    def _transformed_field(self):
        r"""Adapter exposing the transformed velocity to divergence estimators.

        Estimators read ``field.event_ndim`` and call ``field(x, t, c)``;
        without this adapter they would differentiate the *raw* field
        output while the integrator uses ``output_transform(field(...))``.
        For a non-identity transform the integrated drift and the
        divergence used by ``log_prob`` would describe different
        dynamics — the change-of-variables identity ``\partial_t \log p_t = -\nabla \cdot v``
        only holds when the divergence is taken of the *same* velocity
        the trajectory follows.  Wrapping cheaply removes that
        inconsistency in the identity case (the wrapper is just one
        extra Python call) and makes non-identity transforms correct.
        """
        return TransformedField(self._field, self._parameterization.output_transform)

    def _call_and_divergence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        *,
        estimator=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if estimator is not None:
            transformed = self._transformed_field()
            v = transformed(x, t, context)
            div = estimator(transformed, x, t, context)
            return v, div

        call_and_divergence = getattr(self._field, "call_and_divergence", None)
        if call_and_divergence is None:
            msg = (
                "density evaluation requires either `estimator=...` or a field "
                "implementing `call_and_divergence(x, t, c)`"
            )
            raise TypeError(msg)

        # The custom call_and_divergence path is incompatible with a
        # non-identity output_transform: the field's bundled divergence
        # is for the raw output, not the transformed one.  Force an
        # explicit estimator in that case.
        if not self._parameterization.is_identity_transform:
            msg = (
                "density evaluation with a non-identity output_transform "
                "requires `estimator=...`; the field's bundled "
                "`call_and_divergence` returns the divergence of the raw "
                "output, not of the transformed velocity, and using it "
                "would yield density dynamics inconsistent with the "
                "integrated drift."
            )
            raise TypeError(msg)

        try:
            return call_and_divergence(x, t, context)
        except NotImplementedError:
            msg = (
                "density evaluation requires either `estimator=...` or a field "
                "implementing `call_and_divergence(x, t, c)`"
            )
            raise TypeError(msg) from None

    def _sample_base(self, sample_shape=(), *, reparameterized: bool) -> torch.Tensor:
        if reparameterized:
            if not has_rsample(self._base):
                msg = "base distribution does not support rsample"
                raise NotImplementedError(msg)
            if not getattr(self._solver, "supports_rsample", False):
                msg = "solver does not support rsample"
                raise NotImplementedError(msg)
            return self._base.rsample(sample_shape)
        return self._base.sample(sample_shape)

    def _sample_path(
        self,
        z: torch.Tensor,
        *,
        context: torch.Tensor | None,
        return_logp: bool,
        estimator=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_logp:

            def f(x, t):
                tt = cast_time(t, x)
                return self._velocity(x, tt, context)

            return self._integrate(f, z, t0=self._t0, t1=self._t1)

        logp0 = self._base.log_prob(z)

        def f_aug(xi, t):
            tt = cast_time(t, xi)
            v, div = self._call_and_divergence(xi, tt, context, estimator=estimator)
            return v, -div

        return self._integrate_augmented(f_aug, z, logp0, t0=self._t0, t1=self._t1)

    def sample(
        self,
        sample_shape=(),
        *,
        return_logp: bool = False,
        estimator=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self._sample_base(sample_shape, reparameterized=False)
        context = self._expand_context(self._context, z)
        return self._sample_path(
            z,
            context=context,
            return_logp=return_logp,
            estimator=estimator,
        )

    def rsample(
        self,
        sample_shape=(),
        *,
        return_logp: bool = False,
        estimator=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self._sample_base(sample_shape, reparameterized=True)
        context = self._expand_context(self._context, z)
        return self._sample_path(
            z,
            context=context,
            return_logp=return_logp,
            estimator=estimator,
        )

    def log_prob(self, x: torch.Tensor, *, estimator=None) -> torch.Tensor:
        """Evaluate log-density via change of variables.

        Callers should usually pass ``estimator=...`` unless the field
        implements ``call_and_divergence`` itself. We do not auto-select a
        divergence estimator here because the tradeoff is model-dependent:
        exact traces are deterministic but can be expensive, while
        Hutchinson-style estimators scale better but add stochasticity.
        """
        event_ndim = require_event_ndim(self._field)
        lead = x.shape[:-event_ndim] if event_ndim else x.shape
        logp0 = torch.zeros(lead, device=x.device, dtype=x.dtype)
        context = self._expand_context(self._context, x)

        def f_aug(xi, t):
            tt = cast_time(t, xi)
            v, div = self._call_and_divergence(xi, tt, context, estimator=estimator)
            return v, -div

        z, delta = self._integrate_augmented(f_aug, x, logp0, t0=self._t1, t1=self._t0)
        return self._base.log_prob(z) - delta
