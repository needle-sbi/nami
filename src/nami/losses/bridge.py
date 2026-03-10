from __future__ import annotations

import torch

from ..paths.bridge import BrownianBridgePath
from ._common import (
    leading_shape,
    per_sample_mse,
    prepare_time,
    reduce_loss,
    require_event_ndim,
)


def bridge_matching_loss(
    flow_field,
    score_field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    path: BrownianBridgePath | None = None,
    z: torch.Tensor | None = None,
    flow_weight: float = 1.0,
    score_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Schrodinger bridge matching loss (flow + score regression).

    ``x_target`` should be posterior-side samples (paired with context ``c`` if used),
    and ``x_source`` should be independent prior-side samples.

    If ``t`` is not provided, times are sampled uniformly on
    ``[path.eps, 1 - path.eps]`` to avoid endpoint singularities in bridge targets.

    Reference:
    https://proceedings.mlr.press/v238/tong24a/tong24a.pdf
    """
    event_ndim = require_event_ndim(flow_field)
    score_event_ndim = require_event_ndim(score_field)
    if event_ndim != score_event_ndim:
        msg = "flow_field.event_ndim and score_field.event_ndim must match"
        raise ValueError(msg)

    if path is None:
        path = BrownianBridgePath()

    lead = leading_shape(x_target, event_ndim)
    if t is None:
        dtype = x_target.dtype if x_target.dtype.is_floating_point else torch.float32
        t = torch.rand(lead, device=x_target.device, dtype=dtype)
        t = path.eps + (1.0 - 2.0 * path.eps) * t
    else:
        t = prepare_time(x_target, lead, t)

    if z is not None and tuple(z.shape) != tuple(x_target.shape):
        msg = "z must match the shape of x_target"
        raise ValueError(msg)

    xt = path.sample_xt(x_target, x_source, t, z=z)
    flow_target = path.target_ut(x_target, x_source, t, xt=xt)
    score_tgt = path.score_target(x_target, x_source, t, xt=xt)

    vt = flow_field(xt, t, c)
    st = score_field(xt, t, c)

    flow_mse = per_sample_mse(vt, flow_target, lead)
    score_mse = per_sample_mse(st, score_tgt, lead)
    loss = flow_weight * flow_mse + score_weight * score_mse
    return reduce_loss(loss, reduction)
