r"""Closed-form score supplier."""

from __future__ import annotations

import torch


class OracleScore:
    r"""Closed-form :math:`\nabla \log p_\theta` — wraps a user callable.

    For analytic toy families (e.g. the 1-d linear-tilt and 2-d
    Gaussian-mean) both joint and spatial scores are
    closed-form. Also the natural adapter around a differentiable simulator's
    analytic per-event likelihood (e.g. a ``LatentLikelihood.score_latent``)
    which returns the score in the same shape convention (e.g. umi).

    Parameters
    ----------
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Callable ``(x, theta) -> Tensor`` returning the score with the
        protocol's shape convention: ``(*lead, d_theta)`` for joint
        scores, ``(*lead, d_x)`` for spatial scores.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.fn(x, theta)
