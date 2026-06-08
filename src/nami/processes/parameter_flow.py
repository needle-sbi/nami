r"""Continuity-equation flow on x-space with parameter-space path-time.

Transports samples from :math:`p_{\theta_0}` to :math:`p_{\theta_1}`
by integrating :math:`\dot x = \dot\theta(s)\,\nabla_x\phi(x, \theta(s))`
from ``s = 0`` to ``s = 1`` along a chosen
:class:`~nami.paths.parameter.ParameterPath`.

Sibling of :class:`~nami.processes.fm.FlowMatching`, not subclass.  The
time semantics differ (``s`` parameterises :math:`\theta_0 \to
\theta_1` in parameter space, not base :math:`\to` data in distribution
space); the field is a scalar potential whose gradient is the velocity
(the curl-free Otto-horizontal gauge. See
:class:`~nami.fields.scalar_potential.ScalarPotentialField`); and
samples entering at ``s = 0`` come from a *user-provided*
:math:`p_{\theta_0}` (a flow model trained at :math:`\theta_0`, or an
analytic toy), not a learned base, so no ``base`` argument and no
``sample()``.

Runtime only; the training surface lives in
:func:`~nami.losses.parameter_flow.parameter_flow_loss`.

The :math:`\dot\theta` multiplication happens *here*, not in the field:
:math:`\phi` is a potential, and for ``dim(theta) == 1`` (the supported
case) the transport velocity along the path is the scalar
:math:`\dot\theta(s)` times :math:`\nabla_x\phi`.  A vector
:math:`\theta` would need one component potential per direction plus a
Frobenius-compatibility treatment. Currently out of scope.
"""

from __future__ import annotations

import torch

from nami.divergence import ExactDivergence
from nami.lazy import LazyProcess
from nami.processes._common import ProcessRuntimeMixin, cast_time


class ParameterFlow(LazyProcess):
    r"""Lazy parameter-flow process: binds a path at call time.

    A single trained potential serves multi-endpoint workflows — bind a
    new :class:`~nami.paths.parameter.ParameterPath` per
    :math:`\theta_0 \to \theta_1` pair without re-instantiation.

    Unlike other ``LazyProcess`` subclasses, ``forward`` binds a
    *parameter path* rather than a context tensor ``c`` — the path is
    the parameter-flow analogue of bind-time conditioning.

    Parameters
    ----------
    field: ScalarPotentialField
        Scalar potential with the
        :class:`~nami.fields.scalar_potential.ScalarPotentialField`
        surface (``velocity`` / ``velocity_field`` helpers,
        ``event_shape``).
    solver: ODESolver
        ODE solver (any that :class:`~nami.FlowMatching` accepts).
    s0: float
        Path-parameter start.
    s1: float
        Path-parameter end. ``s0 -> s1`` transports
        :math:`p_{\theta_0} \to p_{\theta_1}`.
    """

    def __init__(self, field, solver, *, s0: float = 0.0, s1: float = 1.0):
        super().__init__()
        self.field = field
        self.solver = solver
        self.s0 = float(s0)
        self.s1 = float(s1)

    def forward(self, c=None) -> ParameterFlowProcess:
        """Bind a :class:`~nami.paths.parameter.ParameterPath`.

        The parameter is named ``c`` to honour the ``LazyProcess``
        contract, but the bound context here is a *path*, not a
        tensor: ``process = ParameterFlow(field, solver)(path)``.
        """
        path = c
        if path is None:
            msg = "ParameterFlow requires a ParameterPath to bind"
            raise ValueError(msg)
        return ParameterFlowProcess(
            field=self.field,
            solver=self.solver,
            path=path,
            s0=self.s0,
            s1=self.s1,
        )


class ParameterFlowProcess(ProcessRuntimeMixin):
    """One bound parameter path: transport, log-prob delta, score supply."""

    def __init__(self, field, solver, *, path, s0: float, s1: float):
        self._field = field
        self._solver = solver
        self._path = path
        self._s0 = float(s0)
        self._s1 = float(s1)

    @property
    def field(self):
        return self._field

    @property
    def path(self):
        return self._path

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._field.event_shape)

    def _integrate(self, f, x0: torch.Tensor, *, t0: float, t1: float):
        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            steps = getattr(self._solver, "steps", None)
            if steps is None:
                msg = "solver requires steps"
                raise ValueError(msg)
            kwargs["steps"] = steps
        return self._solver.integrate(f, x0, t0=t0, t1=t1, **kwargs)

    @property
    def pinned(self) -> bool:
        r"""Whether transport runs in *path-pinned* (multi-:math:`\theta`) mode.

        ``True`` when the bound path has :math:`d_\theta > 1`.  In pinned
        mode the field must have been trained with
        :func:`~nami.losses.parameter_flow.path_pinned_parameter_flow_loss`
        **on this exact path** (the trained :math:`\phi` is path-locked):
        :math:`\phi` is the per-unit-``s`` potential, so transport
        integrates :math:`\dot x = \nabla_x\phi(x, \theta(s))` with **no**
        :math:`\dot\theta` scaling.  In the ``d_theta == 1`` case
        (``pinned == False``) :math:`\phi` is the per-unit-:math:`\theta`
        potential and the velocity is scaled by :math:`\dot\theta`.
        """
        probe = torch.zeros(1, dtype=torch.float32)
        return self._path.dtheta_ds(probe).shape[-1] != 1

    def _path_at(self, s, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lead = like.shape[: like.ndim - len(self.event_shape)]
        s_ = cast_time(s, like).expand(lead)
        theta = self._path.theta(s_)
        dtheta = self._path.dtheta_ds(s_)
        return theta, dtheta

    def transport(self, x_at_theta0: torch.Tensor) -> torch.Tensor:
        r"""Transport samples from :math:`p_{\theta_0}` to :math:`p_{\theta_1}`.

        For ``dim(theta) == 1`` integrates :math:`\dot x =
        \dot\theta(s)\,\nabla_x\phi(x, \theta(s))` from ``s0`` to ``s1``.
        For a path with ``dim(theta) > 1`` (path-pinned mode, see
        :attr:`pinned`) the field is the per-unit-``s`` potential and
        transport integrates :math:`\dot x = \nabla_x\phi(x, \theta(s))`
        directly without :math:`\dot\theta` scaling.  The field must then
        have been trained with
        :func:`~nami.losses.parameter_flow.path_pinned_parameter_flow_loss`
        on this exact path (path-locked).
        """
        pinned = self.pinned

        def f(xi, s):
            theta, dtheta = self._path_at(s, xi)
            v = self._field.velocity(xi, theta, create_graph=False)
            return v if pinned else v * dtheta

        return self._integrate(f, x_at_theta0, t0=self._s0, t1=self._s1)

    def transport_with_logp(
        self,
        x_at_theta0: torch.Tensor,
        log_p_at_theta0: torch.Tensor,
        *,
        divergence_estimator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Transport with the instantaneous change-of-variables delta.

        The transported log-density is computable but not controlled in
        KL. The PDE residual loss bounds a Fisher divergence along the path, 
        not the KL of the endpoint, so the returned log-density carries no 
        accuracy guarantee even at zero training loss. This is only meant as
        a diagnostic.

        Parameters
        ----------
        x_at_theta0: torch.Tensor
            Samples from :math:`p_{\theta_0}`, shape ``(*lead, *event_shape)``.
        log_p_at_theta0: torch.Tensor
            Log-density of ``x_at_theta0`` under :math:`p_{\theta_0}`, shape ``(*lead,)``.
            Samples and their log-density under :math:`p_{\theta_0}`.
        divergence_estimator: DivergenceEstimator
            Estimator applied to the potential's gradient field. In the
            ``dim(theta) == 1`` case the divergence of the transport
            velocity is :math:`\dot\theta\,\Delta_x\phi`; in path-pinned
            mode (see :attr:`pinned`) it is :math:`\Delta_x\phi` with no
            :math:`\dot\theta` scaling, the same operator the matching
            training loss uses.
        """
        grad_field = self._field.velocity_field()
        pinned = self.pinned

        def f_aug(xi, s):
            theta, dtheta = self._path_at(s, xi)
            v = self._field.velocity(xi, theta, create_graph=False)
            # The potential ignores t; a dummy scalar keeps the
            # (x, t, c) estimator convention intact.
            t_dummy = torch.zeros((), device=xi.device, dtype=xi.dtype)
            lap = divergence_estimator(grad_field, xi, t_dummy, theta)
            if pinned:
                return v, -lap
            div = lap * dtheta.squeeze(-1)
            return v * dtheta, -div

        kwargs = {}
        if getattr(self._solver, "requires_steps", False):
            kwargs["steps"] = self._solver.steps
        return self._solver.integrate_augmented(
            f_aug,
            x_at_theta0,
            log_p_at_theta0,
            t0=self._s0,
            t1=self._s1,
            **kwargs,
        )

    def score_supply(
        self,
        x: torch.Tensor,
        theta: torch.Tensor | None = None,
        *,
        s: torch.Tensor | None = None,
        spatial_score,
        divergence_estimator=None,
    ) -> torch.Tensor:
        r"""Joint score :math:`\partial_\theta \log \hat p_\theta(x)` from the trained potential.

        Inverts the parameter-flow PDE at evaluation points:

        .. math::

            \partial_\theta \log \hat p_\theta(x)
            = -\Delta_x\phi(x, \theta)
              - \nabla_x\phi(x, \theta) \cdot \nabla_x \log p_\theta(x)

        Recovering the joint score from a trained :math:`\phi` is **not**
        autograd through :math:`\phi` alone — the PDE couples it to the
        spatial score :math:`\nabla_x \log p_\theta` *at evaluation
        time*, hence the required ``spatial_score`` estimator.

        Path-pinned mode (:attr:`pinned`, ``dim(theta) > 1``)
        -----------------------------------------------------
        A path-pinned :math:`\phi` knows the density only along its
        training path, so the **full** :math:`d_\theta`-vector joint score
        is *not* recoverable from it.  What the pinned PDE yields is the
        **directional** (along-path) score
        :math:`\dot\theta(s)\cdot\partial_\theta\log\hat p_{\theta(s)}(x)
        = \tfrac{d}{ds}\log\hat p_{\theta(s)}(x)`, a scalar per sample,
        returned with shape ``(*lead, 1)``. It is exactly the information the pinned
        residual constrains.  ``theta`` is ignored in pinned mode
        such that the path supplies :math:`\theta(s)` from the evaluation ``s``;
        instead pass ``s`` of shape ``(*lead,)`` via the ``s`` argument.

        Parameters
        ----------
        x: torch.Tensor
            Evaluation points, shape ``(*lead, *event_shape)``.
        theta: torch.Tensor
            Parameters, shape ``(*lead, 1)``.  Required in ``dim(theta) ==
            1`` mode; ignored (and may be ``None``) in pinned mode, where
            ``s`` supplies :math:`\theta(s)` instead.
        s: torch.Tensor
            Path parameters, shape ``(*lead,)``.  Required in pinned mode;
            ignored in ``dim(theta) == 1`` mode.
        spatial_score: ScoreEstimator
            :class:`~nami.scores.base.ScoreEstimator` for
            :math:`\nabla_x \log p_\theta(x)`.
        divergence_estimator: DivergenceEstimator
            Estimator for :math:`\Delta_x\phi`; defaults to
            :class:`~nami.divergence.ExactDivergence`.

        Returns
        -------
        torch.Tensor, shape ``(*lead, 1)``
            In ``dim(theta) == 1`` mode the estimated joint score; in
            pinned mode the estimated **directional** (along-path) score
            :math:`\tfrac{d}{ds}\log\hat p_{\theta(s)}(x)`.
        """
        if divergence_estimator is None:
            divergence_estimator = ExactDivergence()

        if self.pinned:
            if s is None:
                msg = "pinned-mode score_supply requires the path parameter s"
                raise ValueError(msg)
            theta_eval, _ = self._path_at(s, x)
        else:
            if theta is None:
                msg = "score_supply requires theta in dim(theta) == 1 mode"
                raise ValueError(msg)
            theta_eval = theta

        grad_phi = self._field.velocity(x, theta_eval, create_graph=False)
        t_dummy = torch.zeros((), device=x.device, dtype=x.dtype)
        lap = divergence_estimator(self._field.velocity_field(), x, t_dummy, theta_eval)
        with torch.no_grad():
            grad_logp = spatial_score(x, theta_eval)

        return (-lap).unsqueeze(-1) - (grad_phi * grad_logp).sum(dim=-1, keepdim=True)


# Pyright needs the forward reference; the docstring already names it.
ParameterFlow.__annotations__["forward"] = "ParameterFlowProcess"
