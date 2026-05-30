from __future__ import annotations

import torch

from nami.core.specs import as_tuple


def require_event_ndim(field) -> int:
    """Read the required event rank from a field-like object.

    Args:
        field: Field-like object exposing ``event_ndim``.

    Returns:
        int: Number of trailing event dimensions.

    Raises:
        ValueError: If the field does not expose ``event_ndim``.
    """
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)
    return int(event_ndim)


def normalise_event_shape(dim: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a flexible event-shape argument.

    Args:
        dim (int | tuple[int, ...]): Event size or event shape.

    Returns:
        tuple[int, ...]: Event shape tuple.
    """
    event_shape = as_tuple(dim)
    if not event_shape:
        msg = "dim must define at least one event dimension"
        raise ValueError(msg)
    if any(size <= 0 for size in event_shape):
        msg = f"dim must be positive, got {dim}"
        raise ValueError(msg)
    return event_shape


def validate_context(
    c: torch.Tensor | None,
    condition_dim: int,
    lead_shape: tuple[int, ...],
) -> None:
    """Validate optional conditioning inputs.

    Args:
        c (torch.Tensor | None): Optional conditioning tensor.
        condition_dim (int): Expected final context dimension.
        lead_shape (tuple[int, ...]): Expected leading shape.

    Raises:
        ValueError: If ``c`` has the wrong rank, leading shape, or final
        dimension.
    """
    if condition_dim == 0:
        if c is not None:
            msg = "context was provided but condition_dim is 0"
            raise ValueError(msg)
        return
    if c is None:
        msg = f"conditioning input with last dimension {condition_dim} is required"
        raise ValueError(msg)
    if c.shape[:-1] != lead_shape:
        msg = (
            f"context shape mismatch: expected leading shape {lead_shape}, "
            f"got {tuple(c.shape[:-1])}"
        )
        raise ValueError(msg)
    if c.shape[-1] != condition_dim:
        msg = f"context dimension mismatch: expected {condition_dim}, got {c.shape[-1]}"
        raise ValueError(msg)
