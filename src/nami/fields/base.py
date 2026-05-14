"""Abstract base for time- and context-conditioned vector fields.

Defines the ``(x, t, c) -> v`` calling convention used throughout the
library and an optional bundled ``call_and_divergence`` hook for
fields that can emit ``\\nabla \\cdot v`` analytically (closed-form or
fused autograd) more cheaply than a generic estimator.
"""

from __future__ import annotations

import torch
from torch import nn


class VectorField(nn.Module):
    """Abstract vector field with the ``(x, t, c)`` calling convention."""

    @property
    def event_ndim(self) -> int:
        """Rank of the event dimensions (trailing dims of ``x``)."""
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the field at ``(x, t)`` with optional context ``c``."""
        raise NotImplementedError

    def call_and_divergence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        *,
        estimator=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(v, \\nabla \\cdot v)`` simultaneously.

        Default raises; subclasses or wrappers (e.g. divergence
        estimators) provide concrete implementations.
        """
        raise NotImplementedError
