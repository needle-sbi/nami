r"""Elliptic-PDE residual loss for parameter-flow training.

Trains a scalar potential :math:`\phi(x, \theta)` so that its gradient
flow transports :math:`p_{\theta_0} \to p_{\theta_1}` along a path in
parameter space.  The objective is the squared residual of the
continuity equation in elliptic form:

.. math::

    \mathcal{L}(\phi) = \mathbb{E}_{p_\theta(x),\, p(\theta)} \Bigl[
        \bigl( \partial_\theta \log p_\theta(x)
               + \Delta_x \phi(x, \theta)
               + \nabla_x \phi(x, \theta) \cdot \nabla_x \log p_\theta(x)
        \bigr)^2
    \Bigr]

Both scores are fixed regression targets from
:class:`~nami.scores.base.ScoreEstimator` objects (oracles, mined simulator
scores, or separately-trained networks); this loss does not train them.

The Laplacian :math:`\Delta_x\phi` is the divergence of the potential's
gradient field
(:meth:`~nami.fields.scalar_potential.ScalarPotentialField.velocity_field`),
so loss and runtime log-density share one :mod:`nami.divergence` estimator.
"""

from __future__ import annotations

import torch

from nami.divergence import ExactDivergence
from nami.losses._common import leading_shape, reduce_loss, require_event_ndim


def parameter_flow_loss(
    field,
    *,
    x: torch.Tensor,
    theta: torch.Tensor,
    joint_score,
    spatial_score,
    divergence_estimator=None,
    reduction: str = "mean",
    create_graph: bool = True,
) -> torch.Tensor:
    r"""Squared residual of the parameter-flow continuity PDE.

    Parameters
    ----------
    field : (ScalarPotentialField)
        Scalar potential with the
        :class:`~nami.fields.scalar_potential.ScalarPotentialField` surface.
    x : (torch.Tensor)
        Samples from :math:`p_\theta`, shape ``(*lead, *event_shape)``.
    theta : (torch.Tensor)
        Matching parameters, shape ``(*lead, d_theta)``.
    joint_score : ScoreEstimator
        Frozen estimator for :math:`\partial_\theta \log p_\theta(x)`,
        returning ``(*lead, d_theta)``.
    spatial_score : (ScoreEstimator)
        Frozen estimator for :math:`\nabla_x \log p_\theta(x)`, returning
        ``(*lead, d_x)``.
    divergence_estimator : (DivergenceEstimator)
        Estimator for :math:`\Delta_x\phi` over the gradient field. Defaults
        to ``ExactDivergence(create_graph=create_graph)``. A custom one used
        for training must set ``create_graph=True`` or ``loss.backward()``
        will not reach the Laplacian term.
    reduction : (str)
        ``"mean"`` | ``"sum"`` | ``"none"``.
    create_graph : (bool)
        Build the second-order graph so ``loss.backward()`` reaches
        :math:`\phi`. Set ``False`` only for eval-time residual monitoring.
        
    Returns
    -------
    torch.Tensor, shape ``(*lead,)``
        The loss.
    

    Notes
    -----
    This is the ``dim(theta) == 1`` loss: :math:`\phi` is the
    per-unit-:math:`\theta` potential, and the matching transport scales
    the velocity by :math:`\dot\theta` (see
    :meth:`~nami.processes.parameter_flow.ParameterFlowProcess.transport`).
    For multi-:math:`\theta` use the path-pinned loss,
    :func:`path_pinned_parameter_flow_loss`, where :math:`\phi` is the
    per-unit-``s`` potential and transport carries no :math:`\dot\theta` scaling.
    """
    event_ndim = require_event_ndim(field)
    if event_ndim != 1:
        msg = (
            "parameter_flow_loss currently supports event_ndim == 1 "
            f"(flat event vectors); got {event_ndim}"
        )
        raise ValueError(msg)
    lead = leading_shape(x, event_ndim)

    if theta.shape[-1] != 1:
        # A single scalar potential cannot represent transport along a
        # vector theta. The velocity must be linear in theta-dot, which
        # requires one component potential per theta direction and
        # reintroduces Frobenius compatibility between them.
        msg = (
            "parameter_flow_loss supports dim(theta) == 1; got "
            f"theta_dim={theta.shape[-1]}.  Multi-theta parameter-flow "
            "requires per-component potentials with Frobenius compatibility."
        )
        raise ValueError(msg)

    if divergence_estimator is None:
        divergence_estimator = ExactDivergence(create_graph=create_graph)

    x = x.detach()

    grad_phi = field.velocity(x, theta, create_graph=create_graph)
    # The potential ignores t; a dummy scalar keeps the (x, t, c) estimator convention intact.
    t_dummy = torch.zeros((), device=x.device, dtype=x.dtype)
    lap_phi = divergence_estimator(field.velocity_field(), x, t_dummy, theta)

    with torch.no_grad():
        dlogp_dtheta = joint_score(x, theta)
        grad_logp = spatial_score(x, theta)

    if dlogp_dtheta.shape != theta.shape:
        msg = (
            f"joint_score returned shape {tuple(dlogp_dtheta.shape)}; "
            f"expected theta's shape {tuple(theta.shape)}"
        )
        raise ValueError(msg)
    if grad_logp.shape != grad_phi.shape:
        msg = (
            f"spatial_score returned shape {tuple(grad_logp.shape)}; "
            f"expected x's shape {tuple(grad_phi.shape)}"
        )
        raise ValueError(msg)

    residual = (
        dlogp_dtheta
        + lap_phi.unsqueeze(-1)
        + (grad_phi * grad_logp).sum(dim=-1, keepdim=True)
    )

    loss_per_sample = residual.pow(2).sum(dim=-1)
    assert loss_per_sample.shape == lead
    return reduce_loss(loss_per_sample, reduction)


def path_pinned_parameter_flow_loss(
    field,
    *,
    x: torch.Tensor,
    s: torch.Tensor,
    path,
    joint_score,
    spatial_score,
    directional_score: bool = False,
    divergence_estimator=None,
    reduction: str = "mean",
    create_graph: bool = True,
) -> torch.Tensor:
    r"""Squared residual of the parameter-flow PDE pinned to one path.

    Multi-:math:`\theta` transport as a single scalar potential trained for
    one fixed :class:`~nami.paths.parameter.ParameterPath`. Projecting the
    continuity equation onto the path tangent :math:`\dot\theta(s)` collapses
    the :math:`d_\theta`-component vector PDE to one scalar equation in ``s``:

    .. math::

        \mathcal{L}(\phi) = \mathbb{E}_{s,\, p_{\theta(s)}(x)} \Bigl[
            \bigl( \dot\theta(s)\cdot\partial_\theta\log p_{\theta(s)}(x)
                   + \Delta_x \phi(x, \theta(s))
                   + \nabla_x \phi(x, \theta(s)) \cdot
                     \nabla_x \log p_{\theta(s)}(x)
            \bigr)^2
        \Bigr]

    where :math:`\dot\theta(s)\cdot\partial_\theta\log p` is the directional
    (along-path) derivative :math:`\tfrac{d}{ds}\log p_{\theta(s)}(x)`, formed
    by contracting the joint score against the path tangent.

    .. note::

       This loss defines :math:`\phi` as the per-unit-``s`` potential, with
       the path tangent already baked into the residual, so transport
       integrates :math:`\dot x = \nabla_x\phi(x, \theta(s))` directly.
       Contrast :func:`parameter_flow_loss`, whose :math:`\phi` is
       per-unit-:math:`\theta` and whose transport scales by
       :math:`\dot\theta`. The trained :math:`\phi` is path-locked — valid
       only along the path it was trained on. Sample with
       :class:`~nami.processes.parameter_flow.ParameterFlowProcess` bound to
       the same path; a path with ``d_theta > 1`` selects pinned mode
       automatically.

    Parameters
    ----------
    field : (ScalarPotentialField)
        Scalar potential
        (:class:`~nami.fields.scalar_potential.ScalarPotentialField`)
        conditioned on :math:`\theta(s)`.
    x : (torch.Tensor)
        Samples from :math:`p_{\theta(s)}`, shape ``(*lead, *event_shape)``.
    s : (torch.Tensor)
        Path parameters, shape ``(*lead,)`` — one scalar ``s`` per sample.
    path : (ParameterPath)
        Fixed :class:`~nami.paths.parameter.ParameterPath` supplying
        :math:`\theta(s)` and :math:`\dot\theta(s)`.
    joint_score : (ScoreEstimator)
        Frozen estimator for :math:`\partial_\theta\log p_\theta(x)`,
        returning ``(*lead, d_theta)``, contracted with :math:`\dot\theta(s)`.
        With ``directional_score=True`` it instead returns the contracted
        along-path derivative :math:`\tfrac{d}{ds}\log p`.
    spatial_score : (ScoreEstimator)
        Frozen estimator for :math:`\nabla_x\log p_\theta(x)`, returning
        ``(*lead, d_x)``.
    directional_score : (bool)
        Treat ``joint_score`` as the scalar :math:`\tfrac{d}{ds}\log p`
        (shape ``(*lead,)`` or ``(*lead, 1)``) and skip the
        :math:`\dot\theta(s)` contraction — the shape produced by
        :class:`~nami.scores.ctsm.CTSMJointScore` with ``directional=True``.
    divergence_estimator : (DivergenceEstimator)
        Estimator for :math:`\Delta_x\phi`. Defaults to
        ``ExactDivergence(create_graph=create_graph)``.
    reduction : (str)
        ``"mean"`` | ``"sum"`` | ``"none"``.
    create_graph : (bool)
        Build the second-order graph so ``loss.backward()`` reaches
        :math:`\phi`.

    Returns
    -------
    torch.Tensor, shape ``(*lead,)``
        The loss.
    """
    event_ndim = require_event_ndim(field)
    if event_ndim != 1:
        msg = (
            "path_pinned_parameter_flow_loss currently supports event_ndim "
            f"== 1 (flat event vectors); got {event_ndim}"
        )
        raise ValueError(msg)
    lead = leading_shape(x, event_ndim)

    if tuple(s.shape) != lead:
        msg = f"s must have the leading shape of x {tuple(lead)}; got {tuple(s.shape)}"
        raise ValueError(msg)

    theta = path.theta(s)
    dtheta = path.dtheta_ds(s)
    if theta.shape != dtheta.shape:
        msg = (
            "path.theta(s) and path.dtheta_ds(s) must share a shape; got "
            f"{tuple(theta.shape)} and {tuple(dtheta.shape)}"
        )
        raise ValueError(msg)

    if divergence_estimator is None:
        divergence_estimator = ExactDivergence(create_graph=create_graph)

    x = x.detach()

    grad_phi = field.velocity(x, theta, create_graph=create_graph)
    t_dummy = torch.zeros((), device=x.device, dtype=x.dtype)
    lap_phi = divergence_estimator(field.velocity_field(), x, t_dummy, theta)

    with torch.no_grad():
        dlogp_dtheta = joint_score(x, theta)
        grad_logp = spatial_score(x, theta)

    if grad_logp.shape != grad_phi.shape:
        msg = (
            f"spatial_score returned shape {tuple(grad_logp.shape)}; "
            f"expected x's shape {tuple(grad_phi.shape)}"
        )
        raise ValueError(msg)

    if directional_score:
        # joint_score already returns the along-path derivative d/ds log p;
        # use it directly with no tangent contraction.
        if dlogp_dtheta.shape == lead:
            directional = dlogp_dtheta
        elif dlogp_dtheta.shape == (*lead, 1):
            directional = dlogp_dtheta.squeeze(-1)
        else:
            msg = (
                "directional joint_score must return shape "
                f"{tuple(lead)} or {(*lead, 1)}; got "
                f"{tuple(dlogp_dtheta.shape)}"
            )
            raise ValueError(msg)
    else:
        if dlogp_dtheta.shape != theta.shape:
            msg = (
                f"joint_score returned shape {tuple(dlogp_dtheta.shape)}; "
                f"expected theta's shape {tuple(theta.shape)}"
            )
            raise ValueError(msg)
        # project the joint score onto the path tangent to form the
        # directional (along-path) derivative d/ds log p, a scalar per sample.
        directional = (dtheta * dlogp_dtheta).sum(dim=-1)

    residual = directional + lap_phi + (grad_phi * grad_logp).sum(dim=-1)

    loss_per_sample = residual.pow(2)
    assert loss_per_sample.shape == lead
    return reduce_loss(loss_per_sample, reduction)
