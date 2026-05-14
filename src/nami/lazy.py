"""Lazy adapters: ``forward(c)`` returns a concrete distribution / field / process.

Conditional workflows pass context ``c`` once at process-instantiation
time and recover a concrete object that does not have to thread ``c``
through every call. The ``Unconditional*`` wrappers are trivial
adapters that ignore ``c`` and return the underlying object — they
let user-facing APIs accept either plain ``torch.distributions``
objects or lazy ones uniformly.
"""

from __future__ import annotations



import torch
from torch import nn


class LazyDistribution(nn.Module):
    """Module whose ``forward(c)`` returns a concrete ``Distribution``."""

    def forward(
        self, c: torch.Tensor | None = None
    ) -> torch.distributions.Distribution:
        raise NotImplementedError


class LazyProcess(nn.Module):
    """Module whose ``forward(c)`` returns a concrete process object."""

    def forward(self, c: torch.Tensor | None = None):
        raise NotImplementedError


class LazyField(nn.Module):
    """Module whose ``forward(c)`` returns a concrete callable field."""

    @property
    def event_ndim(self) -> int | None:
        return None

    def forward(self, c: torch.Tensor | None = None):
        raise NotImplementedError


class UnconditionalDistribution(LazyDistribution):
    """Lazy adapter wrapping a fixed ``torch.distributions.Distribution``."""

    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self._dist = dist

    def forward(
        self, c: torch.Tensor | None = None
    ) -> torch.distributions.Distribution:
        _ = c
        return self._dist


class UnconditionalField(LazyField):
    """Lazy adapter wrapping a fixed field module."""

    def __init__(self, field: nn.Module):
        super().__init__()
        self._field = field

    @property
    def event_ndim(self) -> int | None:
        return getattr(self._field, "event_ndim", None)

    def forward(self, c: torch.Tensor | None = None):
        _ = c
        return self._field
