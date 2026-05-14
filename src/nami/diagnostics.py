"""Inspection helpers: tensor descriptions, field statistics, reversibility.

Diagnostics for validating trained fields and
solvers.
"""

from __future__ import annotations

import torch

from nami.core.broadcast import broadcast
from nami.core.specs import split_event
from nami.fields._common import require_event_ndim
from nami.processes._common import cast_time


def describe(
    x: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    event_ndim: int | None = None,
) -> str:
    """Format the shape / dtype / device of input tensors for inspection."""
    lines = []
    if x is not None:
        lines.append(f"x: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        if event_ndim is not None:
            lead, event_shape = split_event(x, event_ndim)
            lines.append(f"  lead={lead} event_shape={event_shape}")
    if c is not None:
        lines.append(f"c: shape={tuple(c.shape)} dtype={c.dtype} device={c.device}")
    if t is not None:
        lines.append(f"t: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    return "\n".join(lines)


def field_stats(
    field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Return mean / std / min / max of per-sample velocity norms at ``(x, t)``."""
    event_ndim = require_event_ndim(field)

    bx, bt, bc = broadcast(x, t, c, event_ndim=event_ndim, validate_args=True)
    if bt is None:  # pragma: no cover - t is required by the function signature
        msg = "t is required"
        raise TypeError(msg)
    v = field(bx, bt, bc)

    if event_ndim:
        flat = v.reshape(*v.shape[:-event_ndim], -1)
    else:
        flat = v.reshape(*v.shape, 1)

    norms = flat.norm(dim=-1)
    return {
        "mean": norms.mean(),
        "std": norms.std(unbiased=False),
        "min": norms.min(),
        "max": norms.max(),
    }


def divergence_stats(
    field,
    x: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    estimator=None,
) -> dict[str, torch.Tensor]:
    """Return mean / std / min / max of the field's divergence at ``(x, t)``."""
    event_ndim = require_event_ndim(field)

    bx, bt, bc = broadcast(x, t, c, event_ndim=event_ndim, validate_args=True)
    if bt is None:  # pragma: no cover - t is required by the function signature
        msg = "t is required"
        raise TypeError(msg)

    if estimator is not None:
        div = estimator(field, bx, bt, bc)
    else:
        call_and_divergence = getattr(field, "call_and_divergence", None)
        if call_and_divergence is None:
            msg = (
                "divergence_stats requires either `estimator=...` or a field "
                "implementing `call_and_divergence(x, t, c)`"
            )
            raise TypeError(msg)
        try:
            _, div = call_and_divergence(bx, bt, bc)
        except NotImplementedError:
            msg = (
                "divergence_stats requires either `estimator=...` or a field "
                "implementing `call_and_divergence(x, t, c)`"
            )
            raise TypeError(msg) from None

    return {
        "mean": div.mean(),
        "std": div.std(unbiased=False),
        "min": div.min(),
        "max": div.max(),
    }


def reversibility_error(
    field,
    solver,
    x: torch.Tensor,
    *,
    t0: float = 1.0,
    t1: float = 0.0,
    steps: int | None = None,
    c: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Round-trip error of integrating ``field`` from ``t1 -> t0 -> t1``.

    Useful as a smoke test for trained fields and solver step counts.
    """
    event_ndim = require_event_ndim(field)

    def f(xi: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        tt = cast_time(t, xi)
        return field(xi, tt, c)

    kwargs = {}
    if getattr(solver, "requires_steps", False):
        steps = int(steps or getattr(solver, "steps", 0))
        if steps <= 0:
            msg = "solver requires steps"
            raise ValueError(msg)
        kwargs["steps"] = steps

    z = solver.integrate(f, x, t0=t1, t1=t0, **kwargs)
    x_hat = solver.integrate(f, z, t0=t0, t1=t1, **kwargs)

    diff = x_hat - x
    if event_ndim:
        flat = diff.reshape(*diff.shape[:-event_ndim], -1)
    else:
        flat = diff.reshape(*diff.shape, 1)

    err = flat.norm(dim=-1)
    return {
        "mean": err.mean(),
        "std": err.std(unbiased=False),
        "max": err.max(),
    }
