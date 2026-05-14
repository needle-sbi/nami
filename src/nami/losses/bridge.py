from __future__ import annotations

"""Schrödinger-bridge matching loss on the unified vocabulary.

Trains a flow head (Velocity target) and a score head (Score target)
jointly on a Brownian bridge.  Each head is regressed independently
via :func:`~nami.losses.regression.regression_loss`; this function
just bundles them with shared ``t``, ``z``, and weighting kwargs so
callers don't have to thread the same RNG state through two calls.

Implements the joint flow + score regression of Tong et al.,
*Simulation-free Schrödinger Bridges via Score and Flow Matching*,
AISTATS 2024 (https://proceedings.mlr.press/v238/tong24a/tong24a.pdf).
Related bridge-matching constructions: Shi et al., *Diffusion
Schrödinger Bridge Matching*, 2023; Peluchetti, *Diffusion Bridge
Mixture Transports*, 2023.
"""



import torch

from nami.interpolants.bridge import BrownianBridgeInterpolant
from nami.parameterizations import Parameterization, Score, Velocity
from nami.losses._common import (
    leading_shape,
    reduce_loss,
    require_event_ndim,
    sample_t,
)
from nami.losses.regression import regression_loss


def bridge_matching_loss(
    flow_field,
    score_field,
    x_target: torch.Tensor,
    x_source: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: BrownianBridgeInterpolant | None = None,
    z: torch.Tensor | None = None,
    flow_weight: float = 1.0,
    score_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Schrödinger-bridge matching loss (joint flow + score regression).

    Mathematical object: simultaneous regression of the conditional
    velocity (Velocity target) and Stein score (Score target) of a
    Brownian-bridge interpolant between paired endpoints — the
    simulation-free SB objective of Tong et al., *Simulation-free
    Schrödinger Bridges via Score and Flow Matching*, AISTATS 2024.

    ``x_target`` should be posterior-side samples (paired with context
    ``c`` if used), and ``x_source`` should be independent prior-side
    samples.

    Times are auto-sampled from ``[interpolant.eps, 1 - interpolant.eps]``
    when ``t`` is omitted, avoiding the Brownian bridge's endpoint
    singularities.  Both heads see the *same* ``(t, z)`` so they sit
    on the same bridge realisation — when ``z`` is omitted we draw
    one here (rather than letting each ``regression_loss`` call draw
    independent noise inside ``interpolant.sample``).
    """
    event_ndim = require_event_ndim(flow_field)
    score_event_ndim = require_event_ndim(score_field)
    if event_ndim != score_event_ndim:
        msg = "flow_field.event_ndim and score_field.event_ndim must match"
        raise ValueError(msg)

    if interpolant is None:
        interpolant = BrownianBridgeInterpolant()

    lead = leading_shape(x_target, event_ndim)
    # Sample t once; both heads share it (matches legacy behaviour and
    # ensures the same bridge point is used for the velocity and
    # score regressions).
    t = sample_t(x_target, lead, t, eps_t=interpolant.eps)

    if z is not None and tuple(z.shape) != tuple(x_target.shape):
        msg = "z must match the shape of x_target"
        raise ValueError(msg)

    # Draw shared noise once when not supplied — without this, the two
    # ``regression_loss`` calls below would each forward ``noise=None``
    # to ``interpolant.sample``, which draws independent z, and the
    # flow / score regressions would land on different bridge points.
    if z is None:
        z = torch.randn_like(x_target)

    flow_loss = regression_loss(
        flow_field,
        x_target,
        x_source,
        t=t,
        c=c,
        interpolant=interpolant,
        parameterization=Parameterization(target=Velocity()),
        z=z,
        eps_t=interpolant.eps,
        reduction="none",
    )
    score_loss = regression_loss(
        score_field,
        x_target,
        x_source,
        t=t,
        c=c,
        interpolant=interpolant,
        parameterization=Parameterization(target=Score()),
        z=z,
        eps_t=interpolant.eps,
        reduction="none",
    )

    total = flow_weight * flow_loss + score_weight * score_loss
    return reduce_loss(total, reduction)
