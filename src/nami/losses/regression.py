r"""Unified regression loss for the Interpolant + Parameterization vocabulary.

``regression_loss`` is the training-time half of the unified design.  It
consumes any :class:`~nami.interpolants.protocol.Interpolant` and any
:class:`~nami.parameterizations.Parameterization`, computes the
weighted per-sample MSE between the network's emission (after
``output_transform``) and the interpolant's target, and reduces.

This single objective specialises (via choice of interpolant and
parameterization target / weighting) to:

* Flow Matching — Lipman et al., *Flow Matching for Generative
  Modeling*, 2022 (arXiv:2210.02747); Liu et al., *Rectified Flow*,
  2022 (arXiv:2209.03003).
* Score / denoising / ``\epsilon``-prediction — Song et al., *Score-Based
  Generative Modeling through SDEs*, 2020 (arXiv:2011.13456);
  Karras et al., *EDM*, 2022 (arXiv:2206.00364).
* Stochastic-interpolant velocity / score — Albergo, Boffi &
  Vanden-Eijnden, *Stochastic Interpolants: A Unifying Framework*,
  2023 (arXiv:2303.08797).

It owns *t-sampling discipline*: by default ``t`` is drawn from
``U[eps_t, 1 - eps_t]`` so callers cannot accidentally hit the endpoint
singularities of score / SNR targets.  This is the contract that lets
:class:`~nami.interpolants.gaussian.GaussianInterpolant` and the
parameterization factories stay mathematically pure (no silent clamps).
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
from nami.parameterizations import Parameterization


def regression_loss(
    field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: Interpolant,
    parameterization: Parameterization,
    eps_t: float = 1e-3,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Weighted regression of ``output_transform(field(x_t, t, c))`` against the
    interpolant's target.

    Mathematical object: the unified conditional-target MSE that
    specialises to Flow Matching (Lipman et al., 2022, arXiv:2210.02747;
    Liu et al., *Rectified Flow*, 2022, arXiv:2209.03003), score /
    ``\epsilon``-prediction (Song et al., 2020, arXiv:2011.13456; Karras et al.,
    *EDM*, 2022, arXiv:2206.00364), and stochastic-interpolant targets
    (Albergo, Boffi & Vanden-Eijnden, 2023, arXiv:2303.08797),
    depending on the interpolant / parameterization passed in.

    Parameters
    ----------
    field
        Network emitting raw parameters.  Must expose ``event_ndim``.
    x_target, x_source
        Endpoints of the conditional path (data at ``t=0``, source / noise
        at ``t=1`` per nami's convention).
    t
        Optional pre-sampled times of shape matching the leading dims of
        ``x_target``.  When ``None``, drawn from ``U[eps_t, 1 - eps_t]``.
    c
        Optional context.
    interpolant
        Path object implementing the
        :class:`~nami.interpolants.protocol.Interpolant` protocol.
    parameterization
        Bundle of (target, weighting, output_transform).
    eps_t
        Minimum distance from ``{0, 1}`` for auto-sampled ``t``.  Pass
        ``0.0`` to disable clamping (only sensible for interpolants
        whose targets are non-singular at the endpoints).  Ignored when
        ``t`` is supplied — endpoint discipline for explicit ``t`` is
        the caller's responsibility.
    z
        Optional latent noise for stochastic interpolants.  Forwarded to
        ``interpolant.sample(..., noise=z)``.
    reduction
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    event_ndim = require_event_ndim(field)
    lead = leading_shape(x_target, event_ndim)
    t = sample_t(x_target, lead, t, eps_t)

    state = interpolant.sample(x_target, x_source, t, noise=z)
    target_value = interpolant.target(parameterization.target, state)

    raw = field(state.xt, t, c)
    prediction = parameterization.output_transform(raw)

    mse = per_sample_mse(prediction, target_value, lead)

    weight = parameterization.weighting(t)
    if weight.shape != mse.shape:
        weight = weight.expand_as(mse)
    weighted = weight * mse

    return reduce_loss(weighted, reduction)
