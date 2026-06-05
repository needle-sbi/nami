from __future__ import annotations

import pytest
import torch

from nami.core.specs import (
    TensorSpec,
    as_tuple,
    event_numel,
    flatten_event,
    split_event,
    unflatten_event,
    validate_shapes,
)

# as_tuple tests


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        (None, ()),
        (5, (5,)),
        ((2, 3), (2, 3)),
        ([4, 5, 6], (4, 5, 6)),
    ],
    ids=["none", "scalar", "tuple", "list"],
)
def test_as_tuple(input_val, expected):
    assert as_tuple(input_val) == expected


# event_numel tests


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (None, 1),
        ((), 1),
        ((3,), 3),
        ((2, 3), 6),
        ((2, 3, 4), 24),
    ],
    ids=["none", "empty", "1d", "2d", "3d"],
)
def test_event_numel(shape, expected):
    assert event_numel(shape) == expected


# split_event tests


@pytest.mark.parametrize(
    ("event_ndim", "expected_lead", "expected_event"),
    [
        (2, (2, 3), (4, 5)),
        (0, (2, 3, 4, 5), ()),
        (1, (2, 3, 4), (5,)),
        (4, (), (2, 3, 4, 5)),
    ],
    ids=["ndim2", "ndim0", "ndim1", "ndim4"],
)
def test_split_event(sample_tensor_4d, event_ndim, expected_lead, expected_event):
    lead, event = split_event(sample_tensor_4d, event_ndim)
    assert lead == expected_lead
    assert event == expected_event


@pytest.mark.parametrize(
    ("event_ndim", "match"),
    [
        (-1, "event_ndim must be >= 0"),
        (10, "event_ndim exceeds x.ndim"),
    ],
    ids=["negative", "exceeds_ndim"],
)
def test_split_event_errors(sample_tensor_4d, event_ndim, match):
    with pytest.raises(ValueError, match=match):
        split_event(sample_tensor_4d, event_ndim)


# flatten_event tests


@pytest.mark.parametrize(
    ("event_ndim", "expected_shape"),
    [
        (2, (2, 3, 20)),
        (0, (2, 3, 4, 5)),
        (1, (2, 3, 4, 5)),
        (3, (2, 60)),
    ],
    ids=["ndim2", "ndim0", "ndim1", "ndim3"],
)
def test_flatten_event(sample_tensor_4d, event_ndim, expected_shape):
    flat = flatten_event(sample_tensor_4d, event_ndim)
    assert flat.shape == expected_shape


@pytest.mark.parametrize(
    ("event_ndim", "match"),
    [
        (-1, "event_ndim must be >= 0"),
        (10, "event_ndim exceeds x.ndim"),
    ],
    ids=["negative", "exceeds_ndim"],
)
def test_flatten_event_errors(sample_tensor_4d, event_ndim, match):
    with pytest.raises(ValueError, match=match):
        flatten_event(sample_tensor_4d, event_ndim)


# unflatten_event tests


@pytest.mark.parametrize(
    ("event_shape", "expected_shape"),
    [
        ((4, 5), (2, 3, 4, 5)),
        ((), (2, 3, 20)),
        ((2, 10), (2, 3, 2, 10)),
    ],
    ids=["4x5", "empty", "2x10"],
)
def test_unflatten_event(event_shape, expected_shape):
    x = torch.randn(2, 3, 20)
    unflat = unflatten_event(x, event_shape)
    assert unflat.shape == expected_shape


def test_flatten_unflatten_roundtrip(sample_tensor_4d):
    flat = flatten_event(sample_tensor_4d, 2)
    restored = unflatten_event(flat, (4, 5))
    assert torch.allclose(sample_tensor_4d, restored)


# validate_shapes tests


@pytest.mark.parametrize(
    ("event_ndim", "expected_event_shape", "batch_shape"),
    [
        (2, (4, 5), (2, 3)),
        (0, (), (2, 3, 4, 5)),
        (2, (4, 5), None),
        (2, None, (2, 3)),
        (2, None, None),
    ],
    ids=["full", "ndim0", "no_batch", "no_event", "minimal"],
)
def test_validate_shapes(
    sample_tensor_4d, event_ndim, expected_event_shape, batch_shape
):
    validate_shapes(
        sample_tensor_4d,
        event_ndim=event_ndim,
        expected_event_shape=expected_event_shape,
        batch_shape=batch_shape,
    )


def test_validate_shapes_event_error(sample_tensor_4d):
    with pytest.raises(ValueError, match="event_shape mismatch"):
        validate_shapes(sample_tensor_4d, event_ndim=2, expected_event_shape=(4, 4))


def test_validate_shapes_batch_error(sample_tensor_4d):
    with pytest.raises(ValueError, match="batch_shape mismatch"):
        validate_shapes(sample_tensor_4d, event_ndim=2, batch_shape=(2, 4))


@pytest.mark.parametrize(
    ("event_ndim", "match"),
    [
        (-1, "event_ndim must be >= 0"),
        (10, "event_ndim exceeds tensor.ndim"),
    ],
    ids=["negative", "exceeds_ndim"],
)
def test_validate_shapes_ndim_errors(sample_tensor_4d, event_ndim, match):
    with pytest.raises(ValueError, match=match):
        validate_shapes(sample_tensor_4d, event_ndim=event_ndim)


def test_validate_shapes_accepts_spec(sample_tensor_4d):
    spec = TensorSpec(event_shape=(4, 5))
    validate_shapes(sample_tensor_4d, spec, batch_shape=(2, 3))


def test_validate_shapes_spec_event_error(sample_tensor_4d):
    spec = TensorSpec(event_shape=(4, 4))
    with pytest.raises(ValueError, match="event_shape mismatch"):
        validate_shapes(sample_tensor_4d, spec)


def test_validate_shapes_spec_dtype_error(sample_tensor_4d):
    spec = TensorSpec(event_shape=(4, 5), dtype=torch.float64)
    with pytest.raises(TypeError, match="dtype mismatch"):
        validate_shapes(sample_tensor_4d, spec)


# TensorSpec tests


def test_tensor_spec():
    spec = TensorSpec(event_shape=(2, 3), dtype=torch.float32)

    assert spec.event_shape == (2, 3)
    assert spec.event_ndim == 2
    assert spec.numel == 6
    assert spec.dtype == torch.float32


def test_tensor_spec_scalar():
    spec = TensorSpec(event_shape=())

    assert spec.event_ndim == 0
    assert spec.numel == 1
