"""All-mask base distribution for masking-CTMC sampling.

The masking CTMC starts every sampling trajectory from the fully-masked state
(the ``t = 0`` noise end). :class:`AllMask` is the degenerate base that draws
that state: an integer token tensor filled with the operator's mask index.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch.distributions import Distribution
from torch.types import _size

from nami.core.specs import as_tuple

_EMPTY_SIZE = torch.Size()


class AllMask(Distribution):
    """Degenerate base that returns the all-mask token state.

    Args:
        event_shape (tuple[int, ...] | int): Shape of a single event (token
            coordinates).
        mask_index (int): Vocabulary index of the absorbing mask token.
        batch_shape (tuple[int, ...] | int | None): Optional batch shape.
        device (torch.device | None): Device for sampled tensors.
    """

    has_rsample = False
    arg_constraints: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        event_shape: tuple[int, ...] | int,
        *,
        mask_index: int,
        batch_shape: tuple[int, ...] | int | None = None,
        device: torch.device | None = None,
        validate_args: bool = False,
    ):
        self._event_shape_ = as_tuple(event_shape)
        self._batch_shape_ = as_tuple(batch_shape)
        self.mask_index = int(mask_index)
        self.device = device
        super().__init__(
            batch_shape=torch.Size(self._batch_shape_),
            event_shape=torch.Size(self._event_shape_),
            validate_args=validate_args,
        )

    def sample(self, sample_shape: _size = _EMPTY_SIZE) -> torch.Tensor:
        shape = tuple(sample_shape) + self._batch_shape_ + self._event_shape_
        return torch.full(shape, self.mask_index, dtype=torch.long, device=self.device)

    def expand(
        self, batch_shape: _size, _instance: Distribution | None = None
    ) -> AllMask:
        return AllMask(
            self._event_shape_,
            mask_index=self.mask_index,
            batch_shape=tuple(batch_shape),
            device=self.device,
        )
