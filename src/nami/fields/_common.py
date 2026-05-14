from __future__ import annotations


import torch

from nami.core.specs import as_tuple


def require_event_ndim(field) -> int:
    """Read the required ``event_ndim`` from a field-like object.

    A field's ``event_ndim`` is the contract every loss and Process leans
    on to broadcast time tensors and reduce per-sample MSE.  Centralising
    the lookup here keeps the failure message consistent and avoids the
    parallel definitions that drifted between ``losses._common`` and
    ``processes._common``.
    """
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)
    return int(event_ndim)


def normalise_event_shape(dim: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalise a flexible event shape argument."""
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
    """Validate optional conditioning inputs against a target leading shape."""
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
