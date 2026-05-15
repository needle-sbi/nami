r"""Consistency loss for trajectory-pair self-consistency training.

In the flow-matching convention used here, ``t=0`` is the noise endpoint and
``t=1`` is the data endpoint.  Forward consistency anchors at
``target_time=1.0``; reverse consistency anchors at ``target_time=0.0``.

Implements consistency-style distillation / training in the spirit of
Song et al., *Consistency Models*, 2023 (arXiv:2303.01469), with the
flow-matching variant of Yang et al., *Consistency Flow Matching*,
2024.  The consistency function ``f(x, t, v) = x + (T - t) v`` is the
linear-interpolant boundary map; ``target_time = 1`` reduces to the
forward-consistency objective and ``target_time = 0`` to the reverse-
consistency objective.

Consistency losses sample two trajectory points ``(x_t, x_{t+\delta})`` and
minimize MSE between consistency-function evaluations, rather than regressing
a single prediction onto a single target.

The anchor side (which point is detached vs receives gradient) is
chosen automatically based on whether ``target_time`` is closer to
``t=1`` (data endpoint, "forward consistency") or ``t=0`` (noise
endpoint, "reverse consistency").  Only ``Velocity`` targets are
supported — the consistency function ``f(x, t, v) = x + (T - t) v``
needs a velocity, and this loss carries no schedule with which to
convert score / ``\epsilon`` / ``x_0``.
"""

from __future__ import annotations

import torch

from nami.interpolants.protocol import Interpolant
from nami.losses._common import (
    leading_shape,
    per_sample_mse,
    reduce_loss,
    require_event_ndim,
    sample_t,
)
from nami.parameterizations import Parameterization, Velocity


def consistency_loss(
    field,
    *,
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    interpolant: Interpolant,
    parameterization: Parameterization,
    target_time: float = 1.0,
    delta: float = 0.01,
    target_field=None,
    euler_step: bool = False,
    z: torch.Tensor | None = None,
    eps_t: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Compute self-consistency MSE between two trajectory points.

    The consistency map is

    .. math::

       f(x,t,v) = x + (T-t)v,

    and the objective enforces
    ``f(x_t,t,v_t) \approx f(x_{t+\delta},t+\delta,v_{t+\delta})``.

    Args:
        field: Network emitting velocity values before ``output_transform``.
        x_noise (torch.Tensor): Noise endpoint of the conditional path.
        x_data (torch.Tensor): Data endpoint of the conditional path.
        t (torch.Tensor | None): Optional time tensor with leading shape
            matching ``x_data``. If ``None``, times are sampled from
            ``U[eps_t, 1 - eps_t]``.
        c (torch.Tensor | None): Optional conditioning tensor.
        interpolant (Interpolant): Interpolant used to sample trajectory
            points.
        parameterization (Parameterization): Must contain a
            :class:`~nami.parameterizations.Velocity` target.
        target_time (float): Endpoint ``T`` in the consistency map. Must be
            ``0.0`` or ``1.0``.
        delta (float): Positive time offset between the two trajectory points.
        target_field: Optional target network, such as an EMA copy, used for
            the detached anchor.
        euler_step (bool): If ``True``, constructs ``x_{t+\delta}`` with a
            detached Euler step instead of a second interpolant sample.
        z (torch.Tensor | None): Optional shared latent noise for stochastic
            interpolants.
        eps_t (float): Endpoint margin used when sampling ``t``.
        reduction (str): ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        torch.Tensor: Reduced consistency loss, or per-sample losses when
        ``reduction="none"``.
    """
    if not isinstance(parameterization.target, Velocity):
        msg = (
            "consistency_loss requires a Velocity target — the consistency "
            "function f(x, t, v) needs a velocity v, and this loss carries "
            "no schedule with which to convert from Score / Epsilon / X0.  "
            "If you want consistency over a Score-trained model, train a "
            "velocity head on top of it and pass that here."
        )
        raise TypeError(msg)

    if target_time not in (0.0, 1.0):
        msg = (
            f"target_time must be exactly 1.0 (forward consistency, data "
            f"endpoint) or 0.0 (reverse consistency, noise endpoint); "
            f"got {target_time}.  The anchor-side selection is endpoint-"
            "specific; intermediate values would need per-sample anchor selection."
        )
        raise ValueError(msg)

    if delta <= 0.0:
        msg = (
            f"delta must be positive; got {delta}.  A non-positive delta "
            "would push tt = clamp(t + delta, max=1.0) below or equal to "
            "t, breaking the trajectory-pair semantics — and a negative "
            "delta could send tt outside [0, 1] entirely, producing "
            "sqrt(t (1-t)) of a negative argument inside "
            "BrownianBridgeInterpolant.sample."
        )
        raise ValueError(msg)

    event_ndim = require_event_ndim(field)
    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t)
    tt = (t + delta).clamp(max=1.0)

    # Sample ``(x_t, x_{t+\delta})``. Both calls forward the same ``noise=z``
    # so a stochastic interpolant places both trajectory points on the
    # same realisation — without this, BrownianBridgeInterpolant draws
    # independent z at each time and the consistency claim breaks.
    state_t = interpolant.sample(x_noise, x_data, t, noise=z)
    xt = state_t.xt
    vt = parameterization.output_transform(field(xt, t, c))

    if euler_step:
        delta_broad = (tt - t).reshape(t.shape + (1,) * event_ndim)
        xtt = (xt + delta_broad * vt).detach()
    else:
        state_tt = interpolant.sample(x_noise, x_data, tt, noise=z)
        xtt = state_tt.xt

    vtt = parameterization.output_transform(field(xtt, tt, c))

    t_broad = t.reshape(t.shape + (1,) * event_ndim)
    tt_broad = tt.reshape(tt.shape + (1,) * event_ndim)

    def f(x, t_broad_, v):
        return x + (target_time - t_broad_) * v

    # Anchor selection: the side closer to ``target_time`` is the
    # reliable boundary value (because f's identity is at T) and is
    # detached.  For target_time=0 (reverse consistency, noise endpoint)
    # that's the t-side (smaller t, closer to 0); for target_time=1
    # (forward consistency, data endpoint) that's the tt-side (larger t,
    # closer to 1).  ``target_time`` is validated to be exactly one of
    # those two values above, so this branch is the only honest split.
    if target_time == 0.0:
        if target_field is not None:
            with torch.no_grad():
                vt_anchor = parameterization.output_transform(target_field(xt, t, c))
            anchor = f(xt, t_broad, vt_anchor)
        else:
            anchor = f(xt, t_broad, vt).detach()
        prediction = f(xtt, tt_broad, vtt)
    else:
        if target_field is not None:
            with torch.no_grad():
                vtt_anchor = parameterization.output_transform(target_field(xtt, tt, c))
            anchor = f(xtt, tt_broad, vtt_anchor)
        else:
            anchor = f(xtt, tt_broad, vtt).detach()
        prediction = f(xt, t_broad, vt)

    mse = per_sample_mse(prediction, anchor, lead)
    return reduce_loss(mse, reduction)
