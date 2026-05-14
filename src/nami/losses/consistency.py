from __future__ import annotations

"""Unified consistency loss for trajectory-pair self-consistency training.

Replaces the legacy ``cfm_loss`` (forward consistency, ``target_time=0``)
and ``cfm_reverse_loss`` (reverse consistency, ``target_time=1``) with
one function on the unified Interpolant + Parameterization vocabulary.

Implements consistency-style distillation / training in the spirit of
Song et al., *Consistency Models*, 2023 (arXiv:2303.01469), with the
flow-matching variant of Yang et al., *Consistency Flow Matching*,
2024.  The consistency function ``f(x, t, v) = x + (T - t) v`` is the
linear-interpolant boundary map; ``target_time = 0`` reduces to the
forward-consistency objective and ``target_time = 1`` to the reverse-
consistency objective.

Consistency-style losses do **not** fit the
:func:`~nami.losses.regression.regression_loss` shape — they sample two
trajectory points ``(x_t, x_{t+\delta})`` and MSE between consistency-function
evaluations rather than between a single prediction and a single
target.  Honest second loss family: same vocabulary, different
dispatch.

The anchor side (which point is detached vs receives gradient) is
chosen automatically based on whether ``target_time`` is closer to
``t=0`` (data endpoint, "forward consistency") or ``t=1`` (source
endpoint, "reverse consistency").  Only ``Velocity`` targets are
supported — the consistency function ``f(x, t, v) = x + (T - t) v``
needs a velocity, and this loss carries no schedule with which to
convert score / ``\epsilon`` / ``x_0``.
"""



import torch

from nami.interpolants.protocol import Interpolant
from nami.parameterizations import Parameterization, Velocity
from nami.losses._common import (
    leading_shape,
    per_sample_mse,
    reduce_loss,
    require_event_ndim,
    sample_t,
)


def consistency_loss(
    field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: Interpolant,
    parameterization: Parameterization,
    target_time: float = 0.0,
    delta: float = 0.01,
    target_field=None,
    euler_step: bool = False,
    z: torch.Tensor | None = None,
    eps_t: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Self-consistency MSE between two trajectory points under
    ``f(x, t, v) = x + (T - t) v``.

    Mathematical object: a consistency-distillation / consistency-
    training objective (cf. Song et al., *Consistency Models*, 2023,
    arXiv:2303.01469; Yang et al., *Consistency Flow Matching*, 2024)
    that enforces ``f(x_t, t) \approx f(x_{t+\delta}, t+\delta)`` along the conditional
    path of an interpolant.  The anchor side (stop-gradient) is chosen
    by proximity to ``target_time``: ``t=0`` to forward consistency,
    ``t=1`` to reverse consistency.

    Parameters
    ----------
    field
        Network emitting velocity values (after ``output_transform``).
    x_target, x_source
        Endpoints of the conditional path.
    t
        Optional pre-sampled times of shape matching the leading dims of
        ``x_target``.  When ``None``, drawn from ``U[eps_t, 1 - eps_t]``.
    c
        Optional context.
    interpolant
        Interpolant used to sample ``x_t`` and (when ``euler_step=False``)
        ``x_{t+\delta}`` from the conditional path.
    parameterization
        Must carry a :class:`~nami.parameterizations.Velocity` target.
        ``output_transform`` is applied to the raw network output before
        it enters the consistency function.
    target_time
        Endpoint ``T`` of the consistency function.  Must be exactly
        ``0.0`` (data; forward consistency, legacy ``cfm_loss``) or
        ``1.0`` (source; reverse consistency, legacy
        ``cfm_reverse_loss``).  Intermediate values are rejected
        because the anchor-side selection is endpoint-specific:
        per-sample anchor selection (the only honest extension to
        intermediate ``T``) is a future stage when there is a real
        consumer driving the requirement.
    delta
        Time offset between the two trajectory points.
    target_field
        Optional separate network (e.g. an EMA copy) used for the
        anchor evaluation.  When ``None`` the anchor is computed with
        ``field`` and stop-gradient detached.
    euler_step
        When ``True``, generate ``x_{t+\delta}`` via a detached Euler step of
        the learned velocity instead of resampling from the
        interpolant.  Reduces gradient variance from trajectory
        mismatch at no extra forward-pass cost.
    z
        Optional latent noise for stochastic interpolants (e.g.
        :class:`~nami.interpolants.bridge.BrownianBridgeInterpolant`).
        Forwarded as ``noise=z`` to *both* ``interpolant.sample`` calls
        — without this, a stochastic interpolant draws independent
        noise at ``t`` and ``t+\delta`` and the two trajectory points
        cease to share the same realisation.  Deterministic
        interpolants (``LinearInterpolant``) ignore this and reject
        it explicitly; the legacy ``cfm_loss`` was always paired with
        ``LinearPath`` so the question never arose.
    eps_t
        Auto-sampled-``t`` clamping floor.  FM-style callers (linear
        interpolant) pass ``0.0`` for bit-exact equivalence with the
        legacy ``cfm_loss``; diffusion-style interpolants should pass
        a non-zero value to avoid endpoint singularities.
    reduction
        ``"mean"`` | ``"sum"`` | ``"none"``.
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
            f"target_time must be exactly 0.0 (forward consistency) or "
            f"1.0 (reverse consistency); got {target_time}.  The anchor-"
            "side selection is endpoint-specific; intermediate values "
            "would need per-sample anchor selection, which is a future "
            "stage."
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
    lead = leading_shape(x_target, event_ndim)
    t = sample_t(x_target, lead, t, eps_t)
    tt = (t + delta).clamp(max=1.0)

    # Sample ``(x_t, x_{t+\delta})``. Both calls forward the same ``noise=z``
    # so a stochastic interpolant places both trajectory points on the
    # same realisation — without this, BrownianBridgeInterpolant draws
    # independent z at each time and the consistency claim breaks.
    state_t = interpolant.sample(x_target, x_source, t, noise=z)
    xt = state_t.xt
    vt = parameterization.output_transform(field(xt, t, c))

    if euler_step:
        delta_broad = (tt - t).reshape(t.shape + (1,) * event_ndim)
        xtt = (xt + delta_broad * vt).detach()
    else:
        state_tt = interpolant.sample(x_target, x_source, tt, noise=z)
        xtt = state_tt.xt

    vtt = parameterization.output_transform(field(xtt, tt, c))

    t_broad = t.reshape(t.shape + (1,) * event_ndim)
    tt_broad = tt.reshape(tt.shape + (1,) * event_ndim)

    def f(x, t_broad_, v):
        return x + (target_time - t_broad_) * v

    # Anchor selection: the side closer to ``target_time`` is the
    # reliable boundary value (because f's identity is at T) and is
    # detached.  For target_time=0 (forward consistency) that's the
    # t-side; for target_time=1 (reverse consistency) that's the
    # tt-side.  ``target_time`` is validated to be exactly one of
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
