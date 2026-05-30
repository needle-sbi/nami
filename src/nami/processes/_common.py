"""Shared runtime helpers for concrete process classes.

The helpers in this module implement common tensor mechanics used by
:class:`FlowMatchingProcess`, :class:`DiffusionProcess`,
:class:`GeneratorMatchingProcess`, and
:class:`ConsistencyFlowMatchingProcess`.
"""

from __future__ import annotations

import torch


def cast_time(t: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Cast time values to match a reference tensor.

    Args:
        t (float | torch.Tensor): Scalar or tensor time value.
        like (torch.Tensor): Reference tensor for device and dtype.

    Returns:
        torch.Tensor: ``t`` on ``like.device`` with ``like.dtype``.
    """
    return torch.as_tensor(t, device=like.device, dtype=like.dtype)


def expand_context(
    c: torch.Tensor | None,
    target: torch.Tensor,
    event_ndim: int,
) -> torch.Tensor | None:
    """Broadcast a context tensor over a target's leading sample dims.

    Args:
        c (torch.Tensor | None): Context with shape
            ``batch_shape + (context_dim,)``.
        target (torch.Tensor): Tensor with shape
            ``sample_shape + batch_shape + event_shape``.
        event_ndim (int): Number of trailing event dimensions in ``target``.

    Returns:
        torch.Tensor | None: Context expanded to
        ``sample_shape + batch_shape + (context_dim,)``, or ``None``.

    ``c`` has shape ``batch_shape + (context_dim,)``.  ``target`` has
    shape ``sample_shape + batch_shape + event_shape``.  This returns
    ``c`` reshaped to ``sample_shape + batch_shape + (context_dim,)``
    so it can be passed alongside ``target`` to a field that expects
    matching leading dims.
    """
    if c is None:
        return None
    n_expand = target.ndim - event_ndim - c.ndim + 1
    if n_expand > 0:
        for _ in range(n_expand):
            c = c.unsqueeze(0)
        c = c.expand(*target.shape[: target.ndim - event_ndim], c.shape[-1])
    return c


def model_device_dtype(
    model,
) -> tuple[torch.device | None, torch.dtype | None]:
    """Probe a model's first parameter for its (device, dtype).

    Args:
        model: Object that may expose ``parameters()``.

    Returns:
        tuple[torch.device | None, torch.dtype | None]: First parameter device
        and dtype, or ``(None, None)`` when unavailable.
    """
    if not hasattr(model, "parameters"):
        return None, None
    for p in model.parameters():
        return p.device, p.dtype
    return None, None


def resolve_event_ndim(field, fallback: int | None = None) -> int:
    """Resolve the event rank for a field.

    Args:
        field: Field-like object that may expose ``event_ndim``.
        fallback (int | None): Explicit event rank used when the field does not
            expose one.

    Returns:
        int: Number of trailing event dimensions.
    """
    event_ndim = getattr(field, "event_ndim", None)
    if event_ndim is None:
        event_ndim = fallback
    if event_ndim is None:
        msg = "event_ndim must be provided or exposed by field"
        raise ValueError(msg)
    return int(event_ndim)


def validate_base_event_ndim(
    base: torch.distributions.Distribution,
    event_ndim: int,
    *,
    message: str = "field.event_ndim does not match base.event_shape",
) -> None:
    """Validate a base distribution event rank.

    Args:
        base (torch.distributions.Distribution): Base distribution.
        event_ndim (int): Expected event rank.
        message (str): Error message used on mismatch.

    Raises:
        ValueError: If ``len(base.event_shape) != event_ndim``.
    """
    if len(base.event_shape) != event_ndim:
        raise ValueError(message)


class TransformedField:
    """Adapter that composes a field with ``output_transform``.

    Args:
        field: Field-like callable with ``field(x, t, c)``.
        transform: Callable applied to the raw field output.

    Exposed to divergence estimators so they differentiate the
    transformed velocity (the one the integrator follows), not the raw
    field output.  Estimators only require ``event_ndim`` and ``__call__``.
    """

    def __init__(self, field, transform):
        self._field = field
        self._transform = transform

    @property
    def event_ndim(self):
        return getattr(self._field, "event_ndim", None)

    def __call__(self, x, t, c=None):
        return self._transform(self._field(x, t, c))


class ProcessRuntimeMixin:
    """Shared concrete-Process runtime helpers.

    Subclasses must expose ``event_shape``.
    """

    @property
    def event_shape(self) -> tuple[int, ...]:  # pragma: no cover - abstract shape
        raise NotImplementedError

    def _expand_context(
        self,
        c: torch.Tensor | None,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        return expand_context(c, target, len(self.event_shape))
