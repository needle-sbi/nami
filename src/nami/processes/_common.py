"""Shared runtime helpers for concrete process classes.

The helpers in this module implement common tensor mechanics used by
:class:`FlowMatchingProcess`, :class:`DiffusionProcess`,
:class:`GeneratorMatchingProcess`, and
:class:`ConsistencyFlowMatchingProcess`.
"""

from __future__ import annotations

import torch

from nami.core.specs import TensorSpec


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


def resolve_event_ndim_override(
    spec: TensorSpec | None, event_ndim: int | None
) -> int | None:
    """Resolve an event-rank constructor override from a spec or explicit rank.

    Args:
        spec (TensorSpec | None): Event specification, when supplied.
        event_ndim (int | None): Explicit event rank, when supplied.

    Returns:
        int | None: The resolved rank, or ``None`` when neither is given.

    Raises:
        ValueError: If both ``spec`` and ``event_ndim`` are given.
    """
    if spec is not None and event_ndim is not None:
        msg = "pass either spec or event_ndim, not both"
        raise ValueError(msg)
    if spec is not None:
        return spec.event_ndim
    return event_ndim


def resolve_event_shape_override(
    spec: TensorSpec | None, event_shape: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    """Resolve an event-shape constructor override from a spec or explicit shape.

    Args:
        spec (TensorSpec | None): Event specification, when supplied.
        event_shape (tuple[int, ...] | None): Explicit event shape, when
            supplied.

    Returns:
        tuple[int, ...] | None: The resolved shape, or ``None`` when neither
        is given.

    Raises:
        ValueError: If both ``spec`` and ``event_shape`` are given.
    """
    if spec is not None and event_shape is not None:
        msg = "pass either spec or event_shape, not both"
        raise ValueError(msg)
    if spec is not None:
        return spec.event_shape
    return event_shape


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
    field_event_shape: tuple[int, ...] | None = None,
    message: str = "field.event_ndim does not match base.event_shape",
) -> None:
    """Validate a base distribution event rank, and shape when known.

    The rank check (``len(base.event_shape) == event_ndim``) always runs.
    When ``field_event_shape`` is supplied — i.e. the field exposes a
    concrete event shape rather than only a rank — the full tuples are
    compared as well, so a ``(2,)`` field paired with a ``(4,)`` base is
    rejected even though both have rank ``1``.

    Args:
        base (torch.distributions.Distribution): Base distribution.
        event_ndim (int): Expected event rank.
        field_event_shape (tuple[int, ...] | None): The field's concrete
            event shape, when it exposes one. ``None`` falls back to the
            rank-only check.
        message (str): Error message used on mismatch.

    Raises:
        ValueError: If the base event rank differs from ``event_ndim``, or
            if ``field_event_shape`` is given and differs from
            ``base.event_shape``.
    """
    base_shape = tuple(base.event_shape)
    if len(base_shape) != event_ndim:
        raise ValueError(message)
    if field_event_shape is not None and tuple(field_event_shape) != base_shape:
        msg = (
            f"{message}: field event_shape {tuple(field_event_shape)} "
            f"!= base event_shape {base_shape}"
        )
        raise ValueError(msg)


def _static_event_shape(obj: object) -> tuple[int, ...] | None:
    """Best-effort concrete ``event_shape`` behind a field or its lazy wrapper.

    Unwraps the ``Unconditional*`` adapters (which hold the concrete object
    in ``_field`` / ``_dist``) by duck typing, so no import of
    :mod:`nami.lazy` is needed. Returns ``None`` when no concrete shape is
    reachable — the genuinely conditional case, where the shape is unknown
    until a context is bound.
    """
    if obj is None:
        return None
    inner = getattr(obj, "_field", None)
    if inner is None:
        inner = getattr(obj, "_dist", None)
    target = obj if inner is None else inner
    shape = getattr(target, "event_shape", None)
    return None if shape is None else tuple(shape)


def eager_validate_base_event_shape(
    field: object,
    base: object,
    *,
    message: str = "field.event_shape does not match base.event_shape",
) -> None:
    """Fail at bind time when field and base shapes mismatch.

    Intended for process constructors.
    
    If both the field and the base expose a concrete ``event_shape``
    (the unconditional case), a mismatch raises immediately rather than surviving 
    until ``sample()``.
    
    It is a deliberate no-op when either side is conditional, its shape is not
    knowable without a context, so lazy/conditional workflows are unaffected.
    """
    field_shape = _static_event_shape(field)
    base_shape = _static_event_shape(base)
    
    # no need to check when either side is conditional
    if field_shape is None or base_shape is None:
        return
    
    # mismatch raises immediately
    if field_shape != base_shape:
        msg = f"{message}: field {field_shape} != base {base_shape}"
        raise ValueError(msg)


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
