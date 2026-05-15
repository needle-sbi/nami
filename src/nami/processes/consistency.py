"""Consistency Flow Matching process: single-step sampling along a velocity field.

Sampling is one Euler step from ``t0`` to ``t1`` through the
consistency-trained velocity. Log-density is either single-pass via
an optional ``h_head`` predicting ``\\log p_t`` or full augmented-ODE
integration as a fallback.

References
----------
- Song et al., *Consistency Models*, 2023 (arXiv:2303.01469).
- Yang et al., *Consistency Flow Matching*, 2024.
"""

from __future__ import annotations

import torch

from nami.distributions.base import expand_distribution, has_rsample
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
    cast_time,
    resolve_event_ndim,
    validate_base_event_ndim,
)
from nami.processes.fm import FlowMatchingProcess


class ConsistencyFlowMatching(LazyProcess):
    r"""Lazy process for consistency flow matching.

    Sampling is single-step via the consistency function.  Log-probability
    evaluation is either one-step (via an optional ``h_head``) or full ODE
    integration (requires a solver).

    The ``parameterization`` kwarg mirrors the one on
    :class:`~nami.processes.fm.FlowMatching`: ``output_transform`` is
    applied to every raw field output before it enters the consistency
    function, so a model trained with
    :func:`~nami.losses.consistency.consistency_loss` using a non-trivial
    transform is sampled / inverted with the same transform applied at
    runtime.  Only the :class:`~nami.parameterizations.Velocity` target
    is supported; converting Score / Epsilon / X0 to a velocity needs a
    schedule that this Process does not carry.

    Args:
        field: Velocity field or lazy field.
        base: Base distribution or lazy base distribution.
        solver: Optional ODE solver for ODE-based :meth:`log_prob`.
        parameterization (Parameterization | None): Velocity target and output
            projection. Defaults to :func:`velocity_prediction`.
        h_head: Optional scalar head predicting ``\log p_t(x_t)``.
        t0 (float): Source time, usually the noise endpoint.
        t1 (float): Target time, usually the data endpoint.
        event_ndim (int | None): Event rank fallback when the field does not
            expose ``event_ndim``.
        validate_args (bool): Whether to validate target and event-shape
            compatibility.
    """

    def __init__(
        self,
        field,
        base,
        solver=None,
        *,
        parameterization: Parameterization | None = None,
        h_head=None,
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
        self.h_head = h_head
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.event_ndim = event_ndim
        self.validate_args = bool(validate_args)

    def forward(self, c: torch.Tensor | None = None) -> ConsistencyFlowMatchingProcess:
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
                    "ConsistencyFlowMatching supports only the Velocity "
                    "target; got "
                    f"{type(self.parameterization.target).__name__}.  "
                    "Converting Score / Epsilon / X0 needs a NoiseSchedule "
                    "this Process does not carry."
                )
                raise TypeError(msg)
            validate_base_event_ndim(
                base,
                event_ndim,
                message="base.event_shape does not match field.event_ndim",
            )

        return ConsistencyFlowMatchingProcess(
            field=field,
            base=base,
            solver=self.solver,
            parameterization=self.parameterization,
            h_head=self.h_head,
            t0=self.t0,
            t1=self.t1,
            context=c,
            validate_args=self.validate_args,
        )


class ConsistencyFlowMatchingProcess(ProcessRuntimeMixin):
    """Concrete consistency flow matching process.

    Sampling evaluates the forward consistency function in a single pass.
    Inversion evaluates the reverse consistency function in a single pass.

    Log-probability evaluation has two modes:

    - **One-step** (default when ``h_head`` is provided): evaluates
      :math:`h_\\theta(x, t_1)` at the data endpoint.
    - **ODE fallback**: full change-of-variables integration via
      :class:`FlowMatchingProcess` (requires a solver and estimator).
    """

    def __init__(
        self,
        field,
        base: torch.distributions.Distribution,
        solver,
        *,
        parameterization: Parameterization | None = None,
        h_head=None,
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
        self._h_head = h_head
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

    def _velocity(self, x, t, context):
        """Field call composed with ``output_transform`` — mirrors the
        FlowMatching Process pattern so a custom transform applied at
        training time is also applied at runtime.
        """
        return self._parameterization.output_transform(self._field(x, t, context))

    def _single_step(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the consistency function: one Euler step from t0 to t1."""
        context = self._expand_context(self._context, z)
        t0 = cast_time(self._t0, z)
        v = self._velocity(z, t0, context)
        return z + (self._t1 - self._t0) * v

    def sample(self, sample_shape=()) -> torch.Tensor:
        z = self._base.sample(sample_shape)
        return self._single_step(z)

    def rsample(self, sample_shape=()) -> torch.Tensor:
        if not has_rsample(self._base):
            msg = "base distribution does not support rsample"
            raise NotImplementedError(msg)
        z = self._base.rsample(sample_shape)
        return self._single_step(z)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """One-step inversion: data to noise.

        Applies the reverse consistency function:

        .. math::

            z = x + (t_0 - t_1) \\cdot v_\\theta(x, t_1)

        With defaults ``t0=0.0, t1=1.0`` this gives
        ``z = x - v_\\theta(x, 1)``.
        """
        context = self._expand_context(self._context, x)
        t1 = cast_time(self._t1, x)
        v = self._velocity(x, t1, context)
        return x + (self._t0 - self._t1) * v

    def log_prob(
        self,
        x: torch.Tensor,
        *,
        estimator=None,
        ode: bool = False,
    ) -> torch.Tensor:
        """Evaluate log-density.

        When an ``h_head`` was provided and ``ode=False`` (the default),
        returns the one-step prediction :math:`h_\\theta(x, t_1)` at the
        data endpoint.

        When ``ode=True`` (or no ``h_head`` is available), falls back to
        full ODE integration via :class:`FlowMatchingProcess`.  This
        requires a solver and a divergence estimator.

        Parameters
        ----------
        x : Tensor
            Data points to evaluate.
        estimator : nami.divergence.base.DivergenceEstimator or None
            Required for ODE mode.
        ode : bool
            Force full ODE integration even when ``h_head`` is available.
        """
        if not ode and self._h_head is not None:
            context = self._expand_context(self._context, x)
            t1 = cast_time(self._t1, x)
            return self._h_head(x, t1, context)

        if self._solver is None:
            msg = "log_prob requires a solver; pass one to ConsistencyFlowMatching"
            raise ValueError(msg)

        fm_proc = FlowMatchingProcess(
            field=self._field,
            base=self._base,
            solver=self._solver,
            parameterization=self._parameterization,
            t0=self._t0,
            t1=self._t1,
            context=self._context,
            validate_args=self._validate_args,
        )
        return fm_proc.log_prob(x, estimator=estimator)
