"""Event-shape primitives: tuple normalisation, flatten/unflatten, validation.

Establishes nami's ``leading_shape + event_shape`` convention. ``event_shape``
is the shape of a single sample (e.g. ``(d,)`` for vectors, ``(C, H, W)``
for images); everything to the left is batch / sample / time replication.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


def as_tuple(x: Iterable[int] | int | None) -> tuple[int, ...]:
    """Normalize an event-shape argument.

    Args:
        x (Iterable[int] | int | None): Flexible shape input.

    Returns:
        tuple[int, ...]: ``()`` for ``None``, ``(x,)`` for an integer, or a
        tuple copy of an iterable.
    """
    if x is None:
        return ()
    if isinstance(x, int):
        return (int(x),)
    return tuple(int(v) for v in x)


def event_numel(event_shape: Iterable[int] | None) -> int:
    """Return the number of scalar elements in an event.

    Args:
        event_shape (Iterable[int] | None): Event shape.

    Returns:
        int: Product of event-shape dimensions, or ``1`` for scalar events.
    """
    shape = as_tuple(event_shape)
    if not shape:
        return 1
    return int(math.prod(shape))


def split_event(
    x: torch.Tensor, event_ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a tensor shape into leading and event parts.

    Args:
        x (torch.Tensor): Tensor with shape ``lead + event_shape``.
        event_ndim (int): Number of trailing event dimensions.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: ``(lead, event_shape)``.
    """
    if event_ndim < 0:
        msg = "event_ndim must be >= 0"
        raise ValueError(msg)
    if event_ndim > x.ndim:
        msg = "event_ndim exceeds x.ndim"
        raise ValueError(msg)
    if event_ndim == 0:
        return tuple(x.shape), ()
    return tuple(x.shape[:-event_ndim]), tuple(x.shape[-event_ndim:])


def flatten_event(x: torch.Tensor, event_ndim: int) -> torch.Tensor:
    """Flatten event dimensions into one trailing dimension.

    Args:
        x (torch.Tensor): Tensor with shape ``lead + event_shape``.
        event_ndim (int): Number of trailing event dimensions.

    Returns:
        torch.Tensor: Tensor with shape ``lead + (prod(event_shape),)``.
    """
    if event_ndim < 0:
        msg = "event_ndim must be >= 0"
        raise ValueError(msg)
    if event_ndim > x.ndim:
        msg = "event_ndim exceeds x.ndim"
        raise ValueError(msg)
    if event_ndim == 0:
        return x
    return x.reshape(*x.shape[:-event_ndim], -1)


def unflatten_event(x: torch.Tensor, event_shape: tuple[int, ...]) -> torch.Tensor:
    """Restore flattened event dimensions.

    Args:
        x (torch.Tensor): Tensor with one flattened trailing event dimension.
        event_shape (tuple[int, ...]): Target event shape.

    Returns:
        torch.Tensor: Tensor with shape ``lead + event_shape``.
    """
    if not event_shape:
        return x
    return x.reshape(*x.shape[:-1], *event_shape)


def validate_shapes(
    tensor: torch.Tensor,
    event_ndim: int,
    expected_event_shape: tuple[int, ...] | None = None,
    batch_shape: tuple[int, ...] | None = None,
) -> None:
    """Validate event and batch shape contracts.

    Args:
        tensor (torch.Tensor): Tensor to validate.
        event_ndim (int): Number of trailing event dimensions.
        expected_event_shape (tuple[int, ...] | None): Expected event shape.
        batch_shape (tuple[int, ...] | None): Expected leading shape.

    Raises:
        ValueError: If ``event_ndim`` is invalid or a shape check fails.
    """
    if event_ndim < 0:
        msg = "event_ndim must be >= 0"
        raise ValueError(msg)
    if event_ndim > tensor.ndim:
        msg = "event_ndim exceeds tensor.ndim"
        raise ValueError(msg)

    if expected_event_shape is not None:
        actual_event_shape = tuple(tensor.shape[-event_ndim:] if event_ndim > 0 else ())
        if actual_event_shape != expected_event_shape:
            msg = f"event_shape mismatch: expected {expected_event_shape}, got {actual_event_shape}"
            raise ValueError(msg)

    if batch_shape is not None:
        actual_batch_shape = tuple(
            tensor.shape[:-event_ndim] if event_ndim > 0 else tensor.shape
        )
        if actual_batch_shape != batch_shape:
            msg = f"batch_shape mismatch: expected {batch_shape}, got {actual_batch_shape}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TensorSpec:
    """Minimal tensor specification for models, samplers, and distributions.

    Args:
        event_shape (tuple[int, ...]): Shape of one event.
        dtype (torch.dtype | None): Expected tensor dtype, when constrained.
    """

    event_shape: tuple[int, ...]
    dtype: torch.dtype | None = None

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    @property
    def numel(self) -> int:
        return event_numel(self.event_shape)
