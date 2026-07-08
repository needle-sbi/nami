r"""Spatial score via denoising score matching, amortised in theta."""

from __future__ import annotations

import torch


class DSMSpatialScore:
    r"""Spatial score :math:`\nabla_x\log p_\theta(x)` from a trained DSM net.

    Thin adapter: wraps a network ``net(x, theta) -> grad_x log p_theta``
    trained by :func:`~nami.losses.score_matching.denoising_score_matching_loss`
    (denoising score matching, Vincent 2011, amortised over
    :math:`\theta`).  Carries the
    :class:`~nami.scores.base.ScoreEstimator` shape contract so the
    trained net can be a frozen target for the parameter-flow loss.

    This is the **default** route for spatial-score input.

    Alternative ``CTSMSpatialScore`` (not implemented yet)
    ----------------------------------------------------
    The spatial score can instead be obtained by autodiff through a
    trained CTSM time-score network (Eq. 32 of Yu 2025), avoiding a second
    trained network.  That route carries the Liu et al. 2024
    reliability caveat: the autodiff-through-time-score spatial estimate
    can be inaccurate where the time-score net is itself poorly fit, so it
    is not the default and is flagged here rather than shipped. Only DSM
    is implemented.

    Parameters
    ----------
    trained_dsm_net: nn.Module
        Callable ``(x, theta) -> Tensor`` of shape ``(*lead, d_x)``.
    """

    def __init__(self, trained_dsm_net):
        self.net = trained_dsm_net

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.net(x, theta)
