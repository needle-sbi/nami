"""
Masked flow matching for variable-cardinality inputs.

Mask convention: ``1 = real object``, ``0 = padding``.  Masks have shape
``(..., N)`` where *N* is the object (first event) dimension.

TODO: better incoporate into main api, move to losses
"""

from __future__ import annotations

import torch

from nami.interpolants.linear import LinearInterpolant
from nami.interpolants.protocol import Interpolant
from nami.losses._common import (
    leading_shape,
    sample_t,
)
from nami.parameterizations import Parameterization, Velocity


def _expand_mask(mask: torch.Tensor, x: torch.Tensor, event_ndim: int) -> torch.Tensor:
    """Expand *mask* so it broadcasts with *x*.

    Parameters
    ----------
    mask : Tensor
        Binary mask, shape ``lead + (N,)``.
    x : Tensor
        Data tensor, shape ``(..., lead, N, D1, D2, ...)``.
    event_ndim : int
        Number of trailing event dimensions in *x* (must be >= 1).

    Returns
    -------
    Tensor
        Mask broadcastable with *x*, shape ``(..., lead, N, 1, 1, ...)``.
    """
    n_prepend = x.ndim - mask.ndim - (event_ndim - 1)
    m = mask
    for _ in range(n_prepend):
        m = m.unsqueeze(0)
    m = m.expand(x.shape[: x.ndim - event_ndim + 1])
    for _ in range(event_ndim - 1):
        m = m.unsqueeze(-1)
    return m


def masked_fm_loss(
    field,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    *,
    interpolant: Interpolant | None = None,
    parameterization: Parameterization | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Flow-matching loss computed only over real (unmasked) objects.

    Padded positions (``mask == 0``) are excluded from the per-sample
    MSE; the average is taken over real objects only.

    Parameters
    ----------
    field : nn.Module
        Velocity field.  Must expose an ``event_ndim`` attribute >= 2.
    x_data, x_noise : Tensor
        Target and source tensors, each ``lead + event_shape``.
    mask : Tensor
        Binary mask, ``lead + (N,)`` where *N* is the first event dim.
        ``1 = real``, ``0 = padding``.
    t : Tensor, optional
        Per-sample time values (``lead``).  Uniform random if *None*.
    c : Tensor, optional
        Conditioning context forwarded to the field.
    interpolant : Interpolant, optional
        Defaults to :class:`~nami.interpolants.linear.LinearInterpolant`.
        Any interpolant supporting ``Velocity`` is accepted.
    parameterization : Parameterization, optional
        Defaults to ``Parameterization(target=Velocity())``.  Must be
        a Velocity parameterization — masking only makes sense for
        per-object velocity regression.
    reduction : ``'mean'`` | ``'sum'`` | ``'none'``

    Returns
    -------
    Tensor
        Scalar loss, or per-sample losses when ``reduction='none'``.
    """
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)
    if event_ndim < 2:
        msg = "masked_fm_loss requires event_ndim >= 2 (objects x features)"
        raise ValueError(msg)

    if interpolant is None:
        interpolant = LinearInterpolant()
    if parameterization is None:
        parameterization = Parameterization(target=Velocity())
    if not isinstance(parameterization.target, Velocity):
        msg = (
            "masked_fm_loss requires a Velocity target — masking is "
            "applied to the per-object squared error of velocity "
            "regression."
        )
        raise TypeError(msg)

    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t=0.0)

    state = interpolant.sample(x_data, x_noise, t)
    target = interpolant.target(parameterization.target, state)
    prediction = parameterization.output_transform(field(state.xt, t, c))

    sq_err = (prediction - target).pow(2)  # lead + event_shape

    mask_f = mask.float()
    mask_exp = _expand_mask(mask_f, sq_err, event_ndim)  # lead + (N, 1, ...)
    sq_err = sq_err * mask_exp

    # Per-object MSE: collapse feature dims to ``lead + (N,)``.
    n_objects = sq_err.shape[len(lead)]
    sq_err_flat = sq_err.reshape(*lead, n_objects, -1)
    per_object = sq_err_flat.mean(dim=-1)

    # Average over real objects only
    n_real = mask_f.sum(dim=-1).clamp(min=1)
    mse = per_object.sum(dim=-1) / n_real

    if reduction == "none":
        return mse
    if reduction == "sum":
        return mse.sum()
    if reduction == "mean":
        return mse.mean()
    msg = "reduction must be 'mean', 'sum', or 'none'"
    raise ValueError(msg)


def masked_sample(
    field,
    base,
    solver,
    mask: torch.Tensor,
    *,
    sample_shape: tuple[int, ...] = (),
    c: torch.Tensor | None = None,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Sample from a flow matching model with variable-cardinality masking.

    1. Draws noise from *base* and zeros padded positions.
    2. At every solver step, masks the velocity output so padded positions
       receive zero velocity and remain at zero throughout integration.

    Parameters
    ----------
    field : nn.Module
        Velocity field with ``forward(x, t, c)`` and ``event_ndim``.
    base : Distribution
        Base (source) distribution.
    solver
        ODE solver with ``integrate(f, x0, *, t0, t1, ...)``.
    mask : Tensor
        Binary mask ``(batch..., N)``.  ``1 = real``, ``0 = padding``.
    sample_shape : tuple of int
        Independent sample dimensions prepended to the output.
    c : Tensor, optional
        Conditioning context forwarded to the field.
    t0 : float
        Integration start (default ``0.0``, noise endpoint in nami's
        FM convention).
    t1 : float
        Integration end (default ``1.0``, data endpoint).

    Returns
    -------
    Tensor
        Samples with shape ``sample_shape + batch + event_shape``.
        Padded positions are exactly zero.
    """
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)

    z = base.sample(sample_shape)
    mask_f = mask.float()
    mask_full = _expand_mask(mask_f, z, event_ndim)

    # Zero noise at padded positions
    z = z * mask_full

    # Expand context to match sample dimensions
    c_exp = c
    if c_exp is not None:
        lead_ndim = z.ndim - event_ndim
        while c_exp.ndim < lead_ndim + 1:
            c_exp = c_exp.unsqueeze(0)
        lead_shape = z.shape[:lead_ndim]
        c_exp = c_exp.expand(*lead_shape, c_exp.shape[-1])

    def f(x, t):
        tt = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        v = field(x, tt, c_exp)
        return v * mask_full

    kwargs = {}
    if getattr(solver, "requires_steps", False):
        steps = getattr(solver, "steps", None)
        if steps is None:
            msg = "solver requires steps"
            raise ValueError(msg)
        kwargs["steps"] = steps

    return solver.integrate(f, z, t0=t0, t1=t1, **kwargs)
