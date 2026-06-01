"""Base distributions for transport processes.

Thin wrappers around ``torch.distributions`` that expose the
``event_shape`` / ``batch_shape`` conventions nami relies on.
"""

from __future__ import annotations

from nami.distributions.mask import AllMask
from nami.distributions.normal import DiagonalNormal, StandardNormal

__all__ = ["AllMask", "DiagonalNormal", "StandardNormal"]
