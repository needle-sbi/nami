from __future__ import annotations

import torch

from ..interpolants.gamma import BrownianGamma, GammaSchedule
from ..paths.linear import LinearPath
from ._common import (
    expand_like_time,
    leading_shape,
    per_sample_mse,
    prepare_time,
    reduce_loss,
    require_event_ndim,
)


def stochastic_fm_loss(
    field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    path=None,
    gamma: GammaSchedule | None = None,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Flow matching loss for stochastic interpolants with additive Gaussian noise."""
    event_ndim = require_event_ndim(field)

    if path is None:
        path = LinearPath()
    if gamma is None:
        gamma = BrownianGamma()

    lead = leading_shape(x_target, event_ndim)
    t = prepare_time(x_target, lead, t)

    xt_det = path.sample_xt(x_target, x_source, t)
    ut_det = path.target_ut(x_target, x_source, t)

    if z is None:
        z = torch.randn_like(xt_det)
    if tuple(z.shape) != tuple(xt_det.shape):
        msg = "z must match the shape of path samples"
        raise ValueError(msg)

    gamma_t = expand_like_time(gamma.gamma(t), xt_det, event_ndim=event_ndim)
    gamma_dot_t = expand_like_time(gamma.gamma_dot(t), xt_det, event_ndim=event_ndim)

    xt = xt_det + gamma_t * z
    ut = ut_det + gamma_dot_t * z
    vt = field(xt, t, c)

    mse = per_sample_mse(vt, ut, lead)
    return reduce_loss(mse, reduction)
