from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from ..parameterizations import Target, TensorLike


@dataclass(frozen=True)
class InterpolantState:
    r"""Snapshot of an interpolation at time ``t``.

    Carries enough information for any :class:`~nami.parameterizations.Target`
    to be derived without re-sampling.  ``xt`` is the interpolated point;
    ``x_data`` and ``x_noise`` are the endpoint draws that defined this
    realisation (``t = 0`` is noise, ``t = 1`` is data, per nami's
    FM convention); ``noise`` is the latent ``z`` used for stochastic paths and
    is ``None`` for deterministic interpolants.
    """

    xt: torch.Tensor
    x_noise: torch.Tensor
    x_data: torch.Tensor
    t: torch.Tensor
    noise: torch.Tensor | None = None


class Interpolant(Protocol):
    r"""Marginal path between two endpoint distributions.

    Replaces the ``ProbabilityPath`` / ``GeneratorPath`` split.  An
    ``Interpolant`` describes *where* the path goes; what the network learns
    to emit on top of it is the job of
    :class:`~nami.parameterizations.Target` and
    :class:`~nami.parameterizations.Parameterization`.

    Concrete interpolants implement ``sample`` to draw a state at time ``t``
    and ``target`` to produce the regression target for a requested
    :class:`Target` variant.  An interpolant is free to raise
    ``NotImplementedError`` from ``target`` for variants it cannot support
    (e.g. a deterministic path has no well-defined score).
    """

    def sample(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState: ...

    def target(
        self,
        target: Target,
        state: InterpolantState,
    ) -> TensorLike: ...
