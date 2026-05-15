"""Log-density consistency loss.

This module trains a scalar log-density head ``h_\theta(x_t,t)`` with the
instantaneous change-of-variables identity along an ODE trajectory.  It uses
divergence estimates in the style of Hutchinson traces and FFJORD.
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
    """Evaluate standard-normal log density over event dimensions.

    Args:
        x (torch.Tensor): Tensor with shape ``lead + event_shape``.
        event_ndim (int): Number of trailing event dimensions.

    Returns:
        torch.Tensor: Log density with shape ``lead``.
    """
    d = 1
    for s in x.shape[-event_ndim:]:
        d *= s
    flat = x.reshape(*x.shape[:-event_ndim], -1)
    return -0.5 * (d * math.log(2.0 * math.pi) + flat.pow(2).sum(dim=-1))


def log_density_consistency_loss(
    field,
    h_head,
    *,
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    interpolant: Interpolant | None = None,
    target_h_head=None,
    delta: float = 0.01,
    lambda_boundary: float = 1.0,
    divergence_estimator=None,
    euler_step: bool = False,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute log-probability consistency loss for ``h_\theta``.

    Mathematical object: a trajectory-pair consistency objective for a
    log-density head, derived from the instantaneous change-of-variables
    identity along the ODE flow (Chen et al., *Neural Ordinary
    Differential Equations*, 2018, arXiv:1806.07366; Grathwohl et al.,
    *FFJORD*, 2018, arXiv:1810.01367 — divergence estimator).  Anchors
    to the analytically-known base density at ``t = 0`` (the noise
    endpoint in the FM convention) via an auxiliary boundary term.

    Time-direction note: sampling integrates ``t : 0 → 1`` (noise →
    data), but the log-density identity is integrated ``t : 1 → 0``
    (data → noise) so that the trajectory terminates at the base
    distribution whose density is known.  This loss enforces the
    discrete analogue along the consistency pair ``(t, t - δ)``, stepping
    toward the boundary anchor at ``t = 0``.

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
    :math:`t = 0` where the base density is known analytically:

    .. math::

        \\mathcal{L}_{\\mathrm{bc}}
        = \\lVert h(x_0, 0) - \\log p_{\\mathrm{base}}(x_0) \\rVert^2

    Notes:
    Unlike :func:`~nami.losses.consistency.consistency_loss` and
    :func:`~nami.losses.regression.regression_loss`, this loss does
    **not** take a ``parameterization=`` kwarg.  The divergence term
    ``div v_θ(x_t, t)`` and the optional Euler-step branch both consume
    the raw field output as a velocity, so the loss is intrinsically
    tied to the :class:`~nami.parameterizations.Velocity` parameterization.
    For models trained under other parameterizations (Epsilon / Score /
    X0 / VPrediction), convert to a velocity at the :class:`Process`
    layer before training a log-density head against it, or use
    exact-likelihood ODE integration via
    :meth:`ConsistencyFlowMatching.log_prob` with ``ode=True``.

    Args:
        field: Velocity field used for divergence estimation.
        h_head: Scalar head predicting ``\log p_t(x_t)``.
        x_noise (torch.Tensor): Noise endpoint minibatch.
        x_data (torch.Tensor): Data endpoint minibatch.
        t (torch.Tensor | None): Optional time samples. Drawn from
            ``U[0, 1]`` when ``None``.
        c (torch.Tensor | None): Optional conditioning context.
        interpolant (Interpolant | None): Interpolant for ``x_t`` sampling.
            Defaults to :class:`~nami.interpolants.linear.LinearInterpolant`.
        target_h_head: Optional target network, such as an EMA copy, for the
            detached prediction.
        delta (float): Positive time offset along the trajectory.
        lambda_boundary (float): Weight of the boundary loss at ``t=0``.
        divergence_estimator: Estimator for
            ``\operatorname{div} v_\theta``. Defaults to
            :class:`HutchinsonDivergence`.
        euler_step (bool): If ``True``, generate the paired trajectory point
            via a detached Euler step.
        z (torch.Tensor | None): Optional latent noise shared by stochastic
            interpolant samples.
        reduction (str): ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        torch.Tensor: Reduced log-density consistency loss.
    """
    event_ndim = require_event_ndim(field)

    if delta <= 0.0:
        msg = (
            f"delta must be positive; got {delta}.  Non-positive delta "
            "would make the trajectory pair degenerate or push tt above "
            "one (negative deltas are *not* clamped — only the lower "
            "bound at zero is)."
        )
        raise ValueError(msg)

    if interpolant is None:
        interpolant = LinearInterpolant()
    if divergence_estimator is None:
        divergence_estimator = HutchinsonDivergence()

    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t=0.0)
    # Step backward toward the noise endpoint at t=0 (the boundary anchor
    # in the FM convention).  tt < t and (tt - t) < 0 inside the
    # consistency formula — the sign falls out naturally.
    tt = (t - delta).clamp(min=0.0)

    # Trajectory-pair samples share ``z`` so a stochastic interpolant
    # places ``x_t`` and ``x_{t-\delta}`` on the same realisation.
    xt = interpolant.sample(x_noise, x_data, t, noise=z).xt

    # Divergence of the velocity field at (xt, t).
    div_v = divergence_estimator(field, xt, t, c)

    if euler_step:
        vt = field(xt, t, c)

        def _reshape(s: torch.Tensor) -> torch.Tensor:
            return s.reshape(s.shape + (1,) * event_ndim)

        delta_broad = _reshape(tt - t)
        xtt = (xt + delta_broad * vt).detach()
    else:
        xtt = interpolant.sample(x_noise, x_data, tt, noise=z).xt

    # h predictions at both trajectory points.
    # h has its known boundary at t=0 (noise), so h_tt (closer to t=0) is
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

    # Boundary loss: h(x, 0) should equal log p_base(x) at the noise endpoint.
    x_at_zero = interpolant.sample(x_noise, x_data, torch.zeros_like(t)).xt
    h_at_zero = h_head(x_at_zero, torch.zeros_like(t), c)
    log_p_base = _log_prob_base_normal(x_at_zero, event_ndim)
    boundary_mse = (h_at_zero - log_p_base).pow(2)

    total = consistency_mse + lambda_boundary * boundary_mse
    return reduce_loss(total, reduction)
