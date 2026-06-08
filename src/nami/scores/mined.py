r"""Joint score via the mining-gold identity."""

from __future__ import annotations

import torch


class MinedJointScore:
    r"""Joint score :math:`\partial_\theta \log p_\theta(x)` via mining-gold.

    Wraps a simulator's per-event score interface (closed-form latent
    likelihoods, or autodiff through a matrix element).  Not a trained
    network — delegates entirely to simulator instrumentation; the
    class exists so simulator-supplied scores carry the
    :class:`~nami.scores.base.ScoreEstimator` shape contract explicitly.

    References
    ----------
    - Brehmer, Louppe, Pavez, Cranmer, *Mining gold from implicit
      models to improve likelihood-free inference*, PNAS 2020.

    Parameters
    ----------
    simulator_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Callable ``(x, theta) -> Tensor`` of shape ``(*lead, d_theta)``.
    """

    def __init__(self, simulator_score_fn):
        self.simulator_score_fn = simulator_score_fn

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.simulator_score_fn(x, theta)
