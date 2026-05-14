r"""Stochastic flow-matching loss on the unified vocabulary.

Trains a velocity field on a stochastic linear interpolant (Albergo &
Vanden-Eijnden, *Building Normalizing Flows with Stochastic
Interpolants*, 2023; Albergo, Boffi & Vanden-Eijnden, *Stochastic
Interpolants: A Unifying Framework*, 2023, arXiv:2303.08797):
``x_t = (1-t) x_data + t x_noise + gamma(t) z`` with conditional
velocity ``u_t = (x_noise - x_data) + \dot{\gamma}(t) z``.

Implementation is now a thin shim around
:func:`~nami.losses.regression.regression_loss` with a
:class:`~nami.interpolants.linear.StochasticLinearInterpolant` -
preserved as a separate function name for callers used to the
legacy import path (``nami.losses.stochastic_fm.stochastic_fm_loss``)
and because the stochastic-linear case is the most common
gamma-scheduled use that benefits from a one-liner factory.
"""

from __future__ import annotations

import torch

from nami.interpolants.gamma import GammaSchedule
from nami.interpolants.linear import (
    StochasticLinearInterpolant,
    velocity_prediction,
)
from nami.losses.regression import regression_loss


def stochastic_fm_loss(
    field,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: StochasticLinearInterpolant | None = None,
    gamma: GammaSchedule | None = None,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Flow-matching loss for the stochastic linear interpolant.

    Mathematical object: velocity regression against the conditional
    velocity of the gamma-scheduled stochastic linear interpolant of
    Albergo, Boffi & Vanden-Eijnden, *Stochastic Interpolants: A
    Unifying Framework*, 2023 (arXiv:2303.08797).  Thin shim around
    :func:`~nami.losses.regression.regression_loss` that wires up the
    interpolant and the velocity parameterization.

    Either pass an ``interpolant`` (a fully-configured
    ``StochasticLinearInterpolant``) or a bare ``gamma`` schedule to
    construct one with default settings.  Passing both raises
    ``ValueError`` - the API is unambiguous.

    ``z`` is the latent noise; when ``None``, fresh noise is sampled
    internally and the same draw is used inside ``regression_loss``
    (since ``regression_loss`` forwards ``noise=z`` to
    ``interpolant.sample``).
    """
    if interpolant is not None and gamma is not None:
        msg = "pass either `interpolant` or `gamma`, not both"
        raise ValueError(msg)
    if interpolant is None:
        if gamma is None:
            interpolant = StochasticLinearInterpolant()
        else:
            interpolant = StochasticLinearInterpolant(gamma=gamma)

    return regression_loss(
        field,
        x_data,
        x_noise,
        t=t,
        c=c,
        interpolant=interpolant,
        parameterization=velocity_prediction(),
        z=z,
        eps_t=0.0,
        reduction=reduction,
    )
