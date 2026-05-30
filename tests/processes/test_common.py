"""Pin the ``processes/_common`` shared helpers.

These helpers (``cast_time``, ``expand_context``, ``model_device_dtype``,
``resolve_event_ndim``, ``validate_base_event_ndim``, the
``TransformedField`` adapter and the ``ProcessRuntimeMixin``) used to
be private duplicated methods on every Process class.  Centralising
them risked silent behaviour drift; this test file pins each helper's
contract independently and verifies all four Process classes share
the mixin.
"""

from __future__ import annotations

import pytest
import torch

from nami import StandardNormal
from nami.processes import (
    ConsistencyFlowMatchingProcess,
    DiffusionProcess,
    FlowMatchingProcess,
    GeneratorMatchingProcess,
)
from nami.processes._common import (
    ProcessRuntimeMixin,
    TransformedField,
    cast_time,
    expand_context,
    model_device_dtype,
    resolve_event_ndim,
    validate_base_event_ndim,
)

# ---------------------------------------------------------------------------
# cast_time / expand_context / model_device_dtype


def test_cast_time_matches_target_device_and_dtype() -> None:
    like = torch.zeros(3, dtype=torch.float64)
    out = cast_time(0.5, like)
    assert out.dtype == torch.float64
    assert out.device == like.device
    assert out.item() == 0.5


def test_expand_context_broadcasts_over_sample_dims() -> None:
    """``c`` (batch + ctx) is reshaped to (sample, batch + ctx)
    when ``target`` has leading sample dims.
    """
    c = torch.randn(4, 2)  # batch_shape=(4,), ctx_dim=2
    target = torch.zeros(7, 4, 3)  # sample=7, batch=4, event=(3,)
    out = expand_context(c, target, event_ndim=1)
    assert out.shape == (7, 4, 2)
    # Same context broadcast across all 7 sample dims.
    assert torch.equal(out[0], c)
    assert torch.equal(out[6], c)


def test_expand_context_returns_none_when_no_context() -> None:
    assert expand_context(None, torch.zeros(2, 3), event_ndim=1) is None


def test_model_device_dtype_probes_first_parameter() -> None:
    model = torch.nn.Linear(3, 3).to(dtype=torch.float64)
    device, dtype = model_device_dtype(model)
    assert dtype == torch.float64
    assert device.type == "cpu"


def test_model_device_dtype_handles_parameterless_model() -> None:
    """A bare callable returns ``(None, None)`` rather than crashing."""

    def field(x, t, c=None):  # noqa: ARG001
        return torch.zeros_like(x)

    assert model_device_dtype(field) == (None, None)


# ---------------------------------------------------------------------------
# Event-ndim helpers


def test_resolve_event_ndim_reads_field_attribute() -> None:
    class _F:
        event_ndim = 2

    assert resolve_event_ndim(_F()) == 2


def test_resolve_event_ndim_uses_fallback_when_field_lacks_attribute() -> None:
    def field(x, t, c=None):  # noqa: ARG001
        return torch.zeros_like(x)

    assert resolve_event_ndim(field, fallback=1) == 1


def test_resolve_event_ndim_raises_when_neither_available() -> None:
    def field(x, t, c=None):  # noqa: ARG001
        return torch.zeros_like(x)

    with pytest.raises(ValueError, match="event_ndim"):
        resolve_event_ndim(field)


def test_validate_base_event_ndim_rejects_mismatch() -> None:
    base = StandardNormal(event_shape=(3,))  # event_ndim=1
    validate_base_event_ndim(base, 1)
    with pytest.raises(ValueError, match="event_shape"):
        validate_base_event_ndim(base, 2)


# ---------------------------------------------------------------------------
# TransformedField adapter


def test_transformed_field_composes_field_with_transform() -> None:
    class _F:
        event_ndim = 1

        def __call__(self, x, t, c=None):  # noqa: ARG002
            return -x

    adapter = TransformedField(_F(), lambda y: 2.0 * y)
    x = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor(0.0)
    out = adapter(x, t)
    assert torch.equal(out, torch.tensor([-2.0, -4.0, -6.0]))
    assert adapter.event_ndim == 1


# ---------------------------------------------------------------------------
# ProcessRuntimeMixin


@pytest.mark.parametrize(
    "process_cls",
    [
        FlowMatchingProcess,
        DiffusionProcess,
        GeneratorMatchingProcess,
        ConsistencyFlowMatchingProcess,
    ],
    ids=["fm", "diffusion", "gm", "cfm"],
)
def test_all_process_classes_inherit_runtime_mixin(process_cls) -> None:
    """All four Process families share the runtime mixin so the
    ``_cast_time`` / ``_expand_context`` plumbing has one source of truth.
    """
    assert issubclass(process_cls, ProcessRuntimeMixin)


def test_runtime_mixin_expand_context_dispatches_to_pure_helper() -> None:
    """The mixin's ``_expand_context`` must produce the same output as
    the underlying free function — pinning that it is an honest
    one-line delegator that closes over ``self.event_shape`` rather
    than a divergent re-implementation.
    """

    class _StubProcess(ProcessRuntimeMixin):
        @property
        def event_shape(self) -> tuple[int, ...]:
            return (3,)

    proc = _StubProcess()
    c = torch.randn(2, 4)
    target = torch.zeros(5, 2, 3)
    expected = expand_context(c, target, event_ndim=1)
    actual = proc._expand_context(c, target)
    assert torch.equal(actual, expected)
