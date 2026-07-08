r"""Score-estimator protocol for parameter-flow training and recovery.

A score estimator supplies one of the two frozen regression targets consumed
by :func:`~nami.losses.parameter_flow.parameter_flow_loss`
(and, at evaluation time, by :meth:`~nami.processes.parameter_flow.ParameterFlowProcess.score_supply`):

- joint score :math:`\partial_\theta \log p_\theta(x)`, returned with shape ``(lead, d_theta)``;
- spatial score :math:`\nabla_x \log p_\theta(x)`, returned with shape ``(lead, d_x)``.

Both roles share one calling convention, ``(x, theta) -> Tensor``; the trailing dimension distinguishes them.
Estimators are upstream objects; the parameter-flow loss never trains them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ScoreEstimator(Protocol):
    """Generic score interface for both joint (theta) and spatial (x)."""

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor: ...
