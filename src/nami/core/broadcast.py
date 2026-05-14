"""Broadcasting helper for ``(x, t, c)`` triples with explicit event rank.

Used throughout the library to align state ``x``, time ``t``, and
context ``c`` to a common leading (batch) shape while preserving the
event dimensions of ``x`` and the feature dimension of ``c``.
"""

from __future__ import annotations

import torch

from nami.core.specs import split_event


def broadcast(
    x: torch.Tensor,
    t: torch.Tensor | None,
    c: torch.Tensor | None,
    *,
    event_ndim: int,
    validate_args: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Broadcast ``x``, ``t``, ``c`` to a common leading shape.

    ``x`` keeps its trailing ``event_ndim`` dimensions; ``t`` is a
    scalar field over leading dims; ``c`` carries a feature dim
    ``(..., ctx)`` whose ``ctx`` is preserved.

    Returns
    -------
    tuple
        ``(x, t, c)`` each expanded to the joint leading shape (``t``
        and ``c`` may be ``None`` on input and pass through).
    """
    lead, event_shape = split_event(x, event_ndim)
    shapes: list[tuple[int, ...]] = [lead]

    if t is not None:
        shapes.append(tuple(t.shape))
    if c is not None:
        if c.ndim < 1:
            if validate_args:
                msg = "context tensor must have at least 1 dim"
                raise ValueError(msg)
        else:
            shapes.append(tuple(c.shape[:-1]))

    try:
        target = torch.broadcast_shapes(*shapes)
    except RuntimeError as exc:
        if validate_args:
            msg = "failed to broadcast x, t, c"
            raise ValueError(msg) from exc
        raise

    if tuple(x.shape[:-event_ndim] if event_ndim else x.shape) != target:
        x = x.expand(target + event_shape)

    if t is not None and tuple(t.shape) != target:
        t = t.expand(target)

    if c is not None and c.ndim >= 1:
        ctx = c.shape[-1]
        if tuple(c.shape[:-1]) != target:
            c = c.expand((*target, ctx))

    return x, t, c
