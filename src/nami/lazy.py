from __future__ import annotations

import torch
from torch import nn


class LazyDistribution(nn.Module):
    def forward(
        self, c: torch.Tensor | None = None
    ) -> torch.distributions.Distribution:
        raise NotImplementedError


class LazyProcess(nn.Module):
    def forward(self, c: torch.Tensor | None = None):
        raise NotImplementedError


class LazyField(nn.Module):
    @property
    def event_ndim(self) -> int | None:
        return None

    def forward(self, c: torch.Tensor | None = None):
        raise NotImplementedError


class UnconditionalDistribution(LazyDistribution):
    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self._dist = dist

    def forward(
        self, c: torch.Tensor | None = None
    ) -> torch.distributions.Distribution:
        _ = c
        return self._dist


class UnconditionalField(LazyField):
    def __init__(self, field: nn.Module):
        super().__init__()
        self._field = field

    @property
    def event_ndim(self) -> int | None:
        return getattr(self._field, "event_ndim", None)

    def forward(self, c: torch.Tensor | None = None):
        _ = c
        return self._field
