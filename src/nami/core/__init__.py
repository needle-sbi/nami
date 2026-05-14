"""Core tensor utilities: event-shape specs and shape-aware broadcasting.

Foundation layer used by interpolants, fields, and processes to express
``leading_shape + event_shape`` conventions uniformly across the
library. No dependencies on ML papers — pure plumbing.
"""

from __future__ import annotations



from nami.core.specs import (
    TensorSpec,
    as_tuple,
    event_numel,
    flatten_event,
    split_event,
    unflatten_event,
    validate_shapes,
)

__all__ = [
    "TensorSpec",
    "as_tuple",
    "event_numel",
    "flatten_event",
    "split_event",
    "unflatten_event",
    "validate_shapes",
]
