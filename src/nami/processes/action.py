"""Action-matching transport process.

The field is a scalar-output head (e.g.
:class:`~nami.fields.action.ActionHead`) emitting the action potential
``s(x, t)``; the sampling drift is ``\\nabla_x s(x, t)`` recovered by
autograd at each integrator step.

Mirrors :class:`~nami.processes.fm.FlowMatching` in shape — same
``(field, base, solver)`` constructor and ``sample`` / ``rsample`` API
— but no ``output_transform`` (the scalar emission has no canonical
projection at runtime) and no ``log_prob`` (the change-of-variables
identity would need the Laplacian of ``s``, a separate piece of
machinery; can land in a follow-up when an action-density consumer
appears).

References
----------
- Neklyudov et al., *Action Matching: Learning Stochastic Dynamics
  from Samples*, 2023.
"""

from __future__ import annotations

import torch

from nami.distributions.base import expand_distribution, has_rsample
from nami.lazy import (
    LazyDistribution,
    LazyField,
    LazyProcess,
    UnconditionalDistribution,
    UnconditionalField,
)
from nami.parameterizations import Action, Parameterization
from nami.processes._common import (
    ProcessRuntimeMixin,
    cast_time,
    resolve_event_ndim,
    validate_base_event_ndim,
)


def _action_prediction_default() -> Parameterization:
    # Local import to avoid a circular import at module load:
    # ``parameterizations.py`` is below the field/loss/process layer in
    # the dependency graph, so the factory lives next to the loss / head
    # rather than alongside ``velocity_prediction`` in interpolants/.
    return Parameterization(target=Action())


class ActionMatching(LazyProcess):
    """Action-matching process.

    Constructed exactly like :class:`~nami.FlowMatching`: a scalar field,
    a base distribution, and a solver.  The integrator drift at each
    step is :math:`\\nabla_x s(x, t)` computed via ``torch.autograd.grad``.

    Parameters
    ----------
    field
        Scalar-output head :class:`~nami.fields.action.ActionHead` is
        the canonical choice.  Must expose ``event_ndim``.
    base
        Base distribution sampled at ``t0``.
    solver
        ODE solver (any that ``FlowMatching`` accepts).
    parameterization
        Optional :class:`Parameterization` with ``Action`` target.
        Defaults to ``Parameterization(target=Action())``.  The
        ``weighting`` slot is **ignored at runtime** — only the loss
        consumes it; the field's emission *is* the action and there is
        no analogous output projection.
    t0, t1
        Time endpoints for integration.  ``t0=0.0, t1=1.0`` matches the
        flow-matching convention (noise to data).
    event_ndim
        Optional override for the field's event rank.
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
            parameterization
            if parameterization is not None
            else _action_prediction_default()
        )
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.event_ndim = event_ndim
        self.validate_args = bool(validate_args)

    def forward(self, c: torch.Tensor | None = None) -> ActionMatchingProcess:
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
            if not isinstance(self.parameterization.target, Action):
                msg = (
                    "ActionMatching supports only the Action target; got "
                    f"{type(self.parameterization.target).__name__}.  Use "
                    "FlowMatching for Velocity, Diffusion for "
                    "Score / Epsilon / X0."
                )
                raise TypeError(msg)
            validate_base_event_ndim(
                base,
                event_ndim,
                message="base.event_shape does not match field.event_ndim",
            )

        return ActionMatchingProcess(
            field=field,
            base=base,
            solver=self.solver,
            parameterization=self.parameterization,
            t0=self.t0,
            t1=self.t1,
            context=c,
            validate_args=self.validate_args,
        )


class ActionMatchingProcess(ProcessRuntimeMixin):
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
            parameterization
            if parameterization is not None
            else _action_prediction_default()
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

    def _velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        """Recover ``\nabla_x s(x, t, c)`` via autograd.

        ``s`` is a scalar per sample; summing across the batch lets a
        single ``autograd.grad`` call return the per-sample gradient
        because ``s_i`` does not depend on ``x_j`` for ``i != j``.

        We always run under ``torch.enable_grad()`` because ``sample()``
        is commonly called inside ``torch.no_grad()`` and the gradient
        of the field with respect to its *input* is still required even
        when the field's parameters are frozen.
        """
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            s = self._field(xx, t, context)
            (grad_s,) = torch.autograd.grad(
                outputs=s.sum(),
                inputs=xx,
                create_graph=create_graph,
            )
        return grad_s

    def _integrate(self, f, x0: torch.Tensor, *, t0: float, t1: float) -> torch.Tensor:
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = getattr(self._solver, "steps", None)
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps
        return self._solver.integrate(f, x0, t0=t0, t1=t1, **kwargs)

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
        reparameterized: bool,
    ) -> torch.Tensor:
        # ``reparameterized`` paths need the gradient graph through the
        # solver; non-reparam ``sample()`` can detach.  Inner autograd
        # still runs because ``\nabla_x s`` is what *is* the drift, not an
        # incidental quantity.
        create_graph = reparameterized

        def f(x, t):
            tt = cast_time(t, x)
            return self._velocity(x, tt, context, create_graph=create_graph)

        return self._integrate(f, z, t0=self._t0, t1=self._t1)

    def sample(self, sample_shape=()) -> torch.Tensor:
        z = self._sample_base(sample_shape, reparameterized=False)
        context = self._expand_context(self._context, z)
        return self._sample_path(z, context=context, reparameterized=False)

    def rsample(self, sample_shape=()) -> torch.Tensor:
        z = self._sample_base(sample_shape, reparameterized=True)
        context = self._expand_context(self._context, z)
        return self._sample_path(z, context=context, reparameterized=True)

    def log_prob(self, x: torch.Tensor, **_) -> torch.Tensor:
        del x
        # log_prob via change-of-variables would need the Laplacian of
        # the scalar potential ``s`` (since the drift is ``\nabla_x s``). That's a
        # second-order autograd pass we deliberately do not bundle —
        # it can land alongside the first action-density consumer.
        msg = (
            "ActionMatchingProcess.log_prob is not implemented: the "
            "drift is \\nabla_x s, so change-of-variables requires the "
            "Laplacian of s (second-order autograd).  Wire it in when "
            "an action-density consumer drives the requirement."
        )
        raise NotImplementedError(msg)


# Pyright needs the forward reference; the docstring already names it.
ActionMatching.__annotations__["forward"] = "ActionMatchingProcess"
