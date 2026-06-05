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

from nami.core.specs import TensorSpec, as_tuple

_EMPTY_SIZE = torch.Size()


class StandardNormal(Distribution):
    """Isotropic ``N(0, I)`` over ``event_shape`` with explicit batch shape.

    Args:
        event_shape (tuple[int, ...] | int | None): Shape of a single event.
            Mutually exclusive with ``spec``.
        spec (TensorSpec | None): Event specification supplying both
            ``event_shape`` and ``dtype``. Mutually exclusive with the
            explicit ``event_shape`` / ``dtype`` arguments.
        batch_shape (tuple[int, ...] | int | None): Optional batch shape.
        device (torch.device | None): Device for the distribution tensors.
        dtype (torch.dtype | None): Tensor dtype. Mutually exclusive with
            ``spec``.
        validate_args (bool): Whether torch validates arguments.
    """

    has_rsample = True
    arg_constraints: ClassVar[
        dict[str, object]
    ] = {}  # no tensor args to validate; shape/device/dtype are handled internally

    def __init__(
        self,
        event_shape: tuple[int, ...] | int | None = None,
        *,
        spec: TensorSpec | None = None,
        batch_shape: tuple[int, ...] | int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        validate_args: bool = True,
    ):
        if spec is not None:
            if event_shape is not None or dtype is not None:
                msg = "pass either spec or explicit event_shape/dtype, not both"
                raise ValueError(msg)
        elif event_shape is None:
            msg = "either event_shape or spec is required"
            raise ValueError(msg)
        else:
            spec = TensorSpec(as_tuple(event_shape), dtype=dtype)

        self._spec = spec
        self._event_shape = spec.event_shape
        self._batch_shape = as_tuple(batch_shape)
        dtype = spec.dtype
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

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    def expand(
        self, batch_shape: _size, _instance: Distribution | None = None
    ) -> StandardNormal:
        base = self._base.expand(batch_shape)
        new = self.__class__.__new__(self.__class__)
        Distribution.__init__(
            new, base.batch_shape, base.event_shape, validate_args=False
        )
        new._spec = self._spec
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
        self._spec = TensorSpec(
            tuple(loc.shape[-event_ndim:]) if event_ndim else (),
            dtype=loc.dtype,
        )
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

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @property
    def event_ndim(self) -> int:
        return self._spec.event_ndim

    def expand(
        self, batch_shape: _size, _instance: Distribution | None = None
    ) -> DiagonalNormal:
        base = self._base.expand(batch_shape)
        new = self.__class__.__new__(self.__class__)
        Distribution.__init__(
            new, base.batch_shape, base.event_shape, validate_args=False
        )
        new._base = base
        new._spec = self._spec
        return new
