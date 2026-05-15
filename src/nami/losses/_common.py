from __future__ import annotations

import torch

from nami.fields._common import require_event_ndim

__all__ = [
    "leading_shape",
    "per_sample_mse",
    "reduce_loss",
    "require_event_ndim",
    "sample_t",
]


def leading_shape(x: torch.Tensor, event_ndim: int) -> tuple[int, ...]:
    if event_ndim:
        return tuple(x.shape[:-event_ndim])
    return tuple(x.shape)


def sample_t(
    x_data: torch.Tensor,
    lead: tuple[int, ...],
    t: torch.Tensor | None,
    eps_t: float,
) -> torch.Tensor:
    """Sample or broadcast time values.

    Args:
        x_data (torch.Tensor): Reference tensor used for device and dtype.
        lead (tuple[int, ...]): Desired leading shape.
        t (torch.Tensor | None): Optional explicit time tensor.
        eps_t (float): Endpoint margin for automatic sampling.

    Returns:
        torch.Tensor: A tensor with shape ``lead``. If ``t`` is ``None``, values
        are sampled from ``U[eps_t, 1 - eps_t]``. Explicit ``t`` values are
        expanded but not clamped.
    """
    if t is not None:
        if tuple(t.shape) != lead:
            return t.expand(lead)
        return t

    dtype = x_data.dtype if x_data.dtype.is_floating_point else torch.float32
    u = torch.rand(lead, device=x_data.device, dtype=dtype)
    if eps_t == 0.0:
        return u
    if not 0.0 < eps_t < 0.5:
        msg = "eps_t must be in (0, 0.5) or exactly 0 to disable clamping"
        raise ValueError(msg)
    return eps_t + (1.0 - 2.0 * eps_t) * u


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
