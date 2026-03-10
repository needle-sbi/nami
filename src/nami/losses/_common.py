from __future__ import annotations

import torch


def expand_like_time(
    scale: torch.Tensor, target: torch.Tensor, event_ndim: int = 1
) -> torch.Tensor:
    lead_ndim = target.ndim - event_ndim
    n_prepend = lead_ndim - scale.ndim
    shape = (1,) * n_prepend + tuple(scale.shape) + (1,) * event_ndim
    return scale.reshape(shape)


def require_event_ndim(field) -> int:
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)
    return int(event_ndim)


def leading_shape(x: torch.Tensor, event_ndim: int) -> tuple[int, ...]:
    if event_ndim:
        return tuple(x.shape[:-event_ndim])
    return tuple(x.shape)


def prepare_time(
    x_target: torch.Tensor,
    lead: tuple[int, ...],
    t: torch.Tensor | None,
) -> torch.Tensor:
    if t is None:
        dtype = x_target.dtype if x_target.dtype.is_floating_point else torch.float32
        return torch.rand(lead, device=x_target.device, dtype=dtype)
    if tuple(t.shape) != lead:
        return t.expand(lead)
    return t


def per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lead: tuple[int, ...],
) -> torch.Tensor:
    return (prediction - target).pow(2).reshape(*lead, -1).mean(dim=-1)


def reduce_loss(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    msg = "reduction must be 'mean', 'sum', or 'none'"
    raise ValueError(msg)
