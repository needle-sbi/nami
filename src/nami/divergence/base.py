"""Abstract divergence-estimator contract.

A divergence estimator returns ``\\nabla \\cdot v(x, t, c)`` (or an
unbiased estimate of it) given a field and an evaluation point. Used
by :class:`FlowMatchingProcess.log_prob` for the change-of-variables
identity.
"""

from __future__ import annotations


import torch


class DivergenceEstimator:
    """Abstract base for estimators of ``\\nabla \\cdot v``."""

    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the (possibly estimated) divergence of ``field`` at ``(x, t, c)``."""
        raise NotImplementedError
