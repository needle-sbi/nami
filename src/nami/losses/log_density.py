"""Log-density consistency loss.

This module formerly housed ``cfm_loss`` (forward consistency) and
``cfm_reverse_loss`` (reverse consistency); both were deleted in
stage 4 and replaced by
:func:`~nami.losses.consistency.consistency_loss`, which dispatches
between forward and reverse via the ``target_time`` kwarg on the
unified ``Interpolant + Parameterization`` vocabulary.

What survives here is :func:`log_density_consistency_loss` — a
structurally different loss that trains a scalar log-density head
``h_θ(x_t, t)`` via the instantaneous change-of-variables identity
along an ODE trajectory (Chen et al., *Neural Ordinary Differential
Equations*, 2018, arXiv:1806.07366) using a stochastic-trace
divergence estimator (Hutchinson, 1989; Grathwohl et al., *FFJORD:
Free-form Continuous Dynamics for Scalable Reversible Generative
Models*, 2018, arXiv:1810.01367).  It consumes
:class:`~nami.interpolants.protocol.Interpolant` rather than the
deleted ``ProbabilityPath``.
"""

from __future__ import annotations

import math

import torch

from nami.divergence.hutchinson import HutchinsonDivergence
from nami.interpolants.linear import LinearInterpolant
from nami.interpolants.protocol import Interpolant
from nami.losses._common import (
    leading_shape,
    reduce_loss,
    require_event_ndim,
    sample_t,
)


def _log_prob_base_normal(x: torch.Tensor, event_ndim: int) -> torch.Tensor:
    """Log-density of a standard normal, reduced over event dims."""
    d = 1
    for s in x.shape[-event_ndim:]:
        d *= s
    flat = x.reshape(*x.shape[:-event_ndim], -1)
    return -0.5 * (d * math.log(2.0 * math.pi) + flat.pow(2).sum(dim=-1))


def log_density_consistency_loss(
    field,
    h_head,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: Interpolant | None = None,
    target_h_head=None,
    delta: float = 0.01,
    lambda_boundary: float = 1.0,
    divergence_estimator=None,
    euler_step: bool = False,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Log-prob consistency loss for the scalar head :math:`h_\theta`.

    Mathematical object: a trajectory-pair consistency objective for a
    log-density head, derived from the instantaneous change-of-variables
    identity along the ODE flow (Chen et al., *Neural Ordinary
    Differential Equations*, 2018, arXiv:1806.07366; Grathwohl et al.,
    *FFJORD*, 2018, arXiv:1810.01367 — divergence estimator).  Anchors
    to the analytically-known base density at ``t = 1`` via an
    auxiliary boundary term.

    Trains :math:`h_\\theta(x_t, t)` to predict :math:`\\log p_t(x_t)` via
    the consistency condition

    .. math::

        h(x_t, t)
        \\approx h(x_{t+\\delta}, t+\\delta)
        + \\delta \\cdot \\operatorname{div} v_\\theta(x_t, t)

    which follows from the instantaneous change-of-variables formula along the
    ODE trajectory.  The target (right-hand side) is stop-gradiented, anchoring
    from the noise endpoint where :math:`h` is known.

    An auxiliary boundary loss anchors :math:`h` at the noise endpoint
    :math:`t = 1` where the base density is known analytically:

    .. math::

        \\mathcal{L}_{\\mathrm{bc}}
        = \\lVert h(x_1, 1) - \\log p_{\\mathrm{base}}(x_1) \\rVert^2

    Parameters
    ----------
    field : nn.Module
        Velocity field (used for divergence estimation).
    h_head : nn.Module
        Scalar head predicting :math:`\\log p_t(x_t)`.
    x_target : Tensor
        Target (data) minibatch.
    x_source : Tensor
        Source (noise) minibatch.
    t : Tensor or None
        Time samples; drawn from U[0, 1] when ``None``.
    c : Tensor or None
        Optional conditioning context.
    interpolant : nami.interpolants.protocol.Interpolant or None
        Interpolant for ``x_t`` sampling; defaults to
        :class:`~nami.interpolants.linear.LinearInterpolant`.
    target_h_head : nn.Module or None
        EMA copy of ``h_head`` for the target prediction at ``t + delta``.
        When ``None`` the target is computed with ``h_head`` and detached.
    delta : float
        Time offset along the trajectory.
    lambda_boundary : float
        Weight of the boundary loss at :math:`t = 1`.
    divergence_estimator : nami.divergence.base.DivergenceEstimator or None
        Estimator for :math:`\\operatorname{div} v_\\theta`.  Defaults to
        :class:`HutchinsonDivergence`.
    euler_step : bool
        When ``True``, generate ``x_{t+\delta}`` via a detached Euler step of the
        learned velocity instead of the conditional path.  Reduces gradient
        variance from trajectory mismatch.
    z : Tensor or None
        Optional latent noise for stochastic interpolants.  Forwarded
        as ``noise=z`` to *both* trajectory-point samples (``x_t`` and
        ``x_{t+\delta}``) so they sit on the same bridge realisation. The
        boundary point ``x_1`` is *not* given the same noise — it is
        meant to be a fresh sample from the noise endpoint distribution
        for the boundary anchor.  Deterministic interpolants
        (``LinearInterpolant``) reject ``z`` explicitly; stochastic
        interpolants without an explicit ``z`` would otherwise draw
        independent noise per sample call and break the consistency
        claim across the trajectory pair.
    reduction : str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    event_ndim = require_event_ndim(field)

    if delta <= 0.0:
        msg = (
            f"delta must be positive; got {delta}.  Non-positive delta "
            "would make the trajectory pair degenerate or push tt below "
            "zero (negative deltas are *not* clamped — only the upper "
            "bound is)."
        )
        raise ValueError(msg)

    if interpolant is None:
        interpolant = LinearInterpolant()
    if divergence_estimator is None:
        divergence_estimator = HutchinsonDivergence()

    lead = leading_shape(x_target, event_ndim)
    t = sample_t(x_target, lead, t, eps_t=0.0)
    tt = (t + delta).clamp(max=1.0)

    # Trajectory-pair samples share ``z`` so a stochastic interpolant
    # places ``x_t`` and ``x_{t+\delta}`` on the same realisation.
    xt = interpolant.sample(x_target, x_source, t, noise=z).xt

    # Divergence of the velocity field at (xt, t).
    div_v = divergence_estimator(field, xt, t, c)

    if euler_step:
        vt = field(xt, t, c)
        def _reshape(s: torch.Tensor) -> torch.Tensor:
            return s.reshape(s.shape + (1,) * event_ndim)

        delta_broad = _reshape(tt - t)
        xtt = (xt + delta_broad * vt).detach()
    else:
        xtt = interpolant.sample(x_target, x_source, tt, noise=z).xt

    # h predictions at both trajectory points.
    # h has its known boundary at t=1 (noise), so h_tt (closer to t=1) is
    # the reliable anchor.  The online prediction h_t receives gradient.
    h_t = h_head(xt, t, c)

    if target_h_head is not None:
        with torch.no_grad():
            h_tt = target_h_head(xtt, tt, c)
    else:
        h_tt = h_head(xtt, tt, c)

    # Target: h(xtt, tt) + delta * div v(xt, t), all stop-gradiented.
    # From the ODE: ``h(x_t, t) \approx h(x_{t+\delta}, t+\delta) + \delta \, \mathrm{div}\, v(x_t, t)``.
    actual_delta = tt - t  # may differ from `delta` near boundary due to clamp
    target = (h_tt + actual_delta * div_v).detach()

    consistency_mse = (h_t - target).pow(2)

    # Boundary loss: h(x, 1) should equal log p_base(x) at the noise endpoint.
    x_at_one = interpolant.sample(x_target, x_source, torch.ones_like(t)).xt
    h_at_one = h_head(x_at_one, torch.ones_like(t), c)
    log_p_base = _log_prob_base_normal(x_at_one, event_ndim)
    boundary_mse = (h_at_one - log_p_base).pow(2)

    total = consistency_mse + lambda_boundary * boundary_mse
    return reduce_loss(total, reduction)
