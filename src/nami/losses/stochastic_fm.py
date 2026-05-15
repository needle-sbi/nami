r"""Stochastic flow-matching loss.

Trains a velocity field on a stochastic linear interpolant (Albergo &
Vanden-Eijnden, *Building Normalizing Flows with Stochastic
Interpolants*, 2023; Albergo, Boffi & Vanden-Eijnden, *Stochastic
Interpolants: A Unifying Framework*, 2023, arXiv:2303.08797):

.. math::

   x_t = (1-t)x_{\mathrm{noise}} + t x_{\mathrm{data}} + \gamma(t)z,
   \qquad
   u_t = x_{\mathrm{data}} - x_{\mathrm{noise}} + \dot{\gamma}(t)z.
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
    *,
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    interpolant: StochasticLinearInterpolant | None = None,
    gamma: GammaSchedule | None = None,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute stochastic flow-matching loss.

    Args:
        field: Velocity field.
        x_noise (torch.Tensor): Noise endpoint.
        x_data (torch.Tensor): Data endpoint.
        t (torch.Tensor | None): Optional time tensor.
        c (torch.Tensor | None): Optional conditioning tensor.
        interpolant (StochasticLinearInterpolant | None): Fully configured
            stochastic interpolant.
        gamma (GammaSchedule | None): Gamma schedule used to construct an
            interpolant when ``interpolant`` is ``None``.
        z (torch.Tensor | None): Optional latent noise shared by sampling and
            target construction.
        reduction (str): ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        torch.Tensor: Reduced stochastic flow-matching loss.
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
        x_noise=x_noise,
        x_data=x_data,
        t=t,
        c=c,
        interpolant=interpolant,
        parameterization=velocity_prediction(),
        z=z,
        eps_t=0.0,
        reduction=reduction,
    )
