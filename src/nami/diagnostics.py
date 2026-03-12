from __future__ import annotations

import torch

from .core.broadcast import broadcast


def field_stats(
    field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)

    x, t, c = broadcast(x, t, c, event_ndim=event_ndim, validate_args=True)
    v = field(x, t, c)

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
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)

    x, t, c = broadcast(x, t, c, event_ndim=event_ndim, validate_args=True)

    if estimator is not None:
        div = estimator(field, x, t, c)
    else:
        call_and_divergence = getattr(field, "call_and_divergence", None)
        if call_and_divergence is None:
            msg = (
                "divergence_stats requires either `estimator=...` or a field "
                "implementing `call_and_divergence(x, t, c)`"
            )
            raise TypeError(msg)
        try:
            _, div = call_and_divergence(x, t, c)
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
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        msg = "field.event_ndim is required"
        raise ValueError(msg)

    def cast_time(t: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, device=like.device, dtype=like.dtype)

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
