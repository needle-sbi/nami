"""Standard and diagonal Normal base distributions.

``StandardNormal`` is the default source distribution for transport
processes; ``DiagonalNormal`` lets callers parameterise per-coordinate
mean and scale (e.g. learned base distributions, posterior priors).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch.distributions import Distribution, Independent, Normal
from torch.types import _size

from nami.core.specs import as_tuple

_EMPTY_SIZE = torch.Size()


class StandardNormal(Distribution):
    """Isotropic ``N(0, I)`` over ``event_shape`` with explicit batch shape."""

    has_rsample = True
    arg_constraints: ClassVar[
        dict[str, object]
    ] = {}  # no tensor args to validate; shape/device/dtype are handled internally

    def __init__(
        self,
        event_shape: tuple[int, ...] | int,
        *,
        batch_shape: tuple[int, ...] | int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        validate_args: bool = True,
    ):
        self._event_shape = as_tuple(event_shape)
        self._batch_shape = as_tuple(batch_shape)
        shape = self._batch_shape + self._event_shape
        loc = torch.zeros(shape, device=device, dtype=dtype)
        scale = torch.ones(shape, device=device, dtype=dtype)
        base = Normal(loc, scale)
        if self._event_shape:
            base = Independent(base, len(self._event_shape))
        self._base = base
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape: _size = _EMPTY_SIZE) -> torch.Tensor:
        return self._base.sample(sample_shape)

    def rsample(self, sample_shape: _size = _EMPTY_SIZE) -> torch.Tensor:
        return self._base.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._base.log_prob(value)

    @property
    def mean(self) -> torch.Tensor:
        return self._base.mean

    @property
    def variance(self) -> torch.Tensor:
        return self._base.variance

    def expand(
        self, batch_shape: _size, _instance: Distribution | None = None
    ) -> StandardNormal:
        base = self._base.expand(batch_shape)
        new = self.__class__.__new__(self.__class__)
        Distribution.__init__(
            new, base.batch_shape, base.event_shape, validate_args=False
        )
        new._event_shape = self._event_shape
        new._batch_shape = tuple(batch_shape)
        new._base = base
        return new


class DiagonalNormal(Distribution):
    """Diagonal Gaussian with per-coordinate ``loc`` and ``scale``."""

    has_rsample = True
    arg_constraints: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        *,
        event_ndim: int = 1,
        validate_args: bool = True,
    ):
        base = Normal(loc, scale)
        if event_ndim:
            base = Independent(base, event_ndim)
        self._base = base
        self.event_ndim = event_ndim
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape: _size = _EMPTY_SIZE) -> torch.Tensor:
        return self._base.sample(sample_shape)

    def rsample(self, sample_shape: _size = _EMPTY_SIZE) -> torch.Tensor:
        return self._base.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._base.log_prob(value)

    @property
    def mean(self) -> torch.Tensor:
        return self._base.mean

    @property
    def variance(self) -> torch.Tensor:
        return self._base.variance

    def expand(
        self, batch_shape: _size, _instance: Distribution | None = None
    ) -> DiagonalNormal:
        base = self._base.expand(batch_shape)
        new = self.__class__.__new__(self.__class__)
        Distribution.__init__(
            new, base.batch_shape, base.event_shape, validate_args=False
        )
        new._base = base
        new.event_ndim = self.event_ndim
        return new
