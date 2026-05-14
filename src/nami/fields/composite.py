from __future__ import annotations

r"""Composite fields that combine separately trained heads.

The legacy :mod:`nami.interpolants.transforms` module carried two
*two-model combiners* (``DriftFromVelocityScore`` and
``MarkovizationDriftFromVelocityScore``) that wrapped a velocity field
and a separate score field, combining them via gamma-schedule
arithmetic to produce a probability-flow drift or markovisation SDE
drift.  Those wrappers were deleted in stage 4 of the refactor — their
single-model siblings (``ScoreFromEta``, ``ScoreFromRawNoise``,
``MirrorVelocityFromScore``) are subsumed by Process-layer dispatch on
``parameterization.target``, but the *two-model* shape doesn't fit the
single-parameterization-per-field assumption.

This module now holds the concrete replacements for the deleted
two-model wrappers.  They are plain fields: callers pass separately
trained velocity and score models plus the stochastic-interpolant gamma
schedule, and the composite emits the runtime drift a Process needs.
"""


from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from nami.diffusion import expand_like
from nami.interpolants.gamma import GammaSchedule


@runtime_checkable
class TwoHeadField(Protocol):
    r"""Field that combines two trained models into one runtime quantity.

    Implementations consume the outputs of two separately-trained
    networks (typically a velocity head and a score head) plus the
    ambient noise schedule, and emit the runtime quantity a Process
    needs (probability-flow drift, markovisation SDE drift, etc.).

    The legacy classes that this protocol will replace lived in the
    deleted ``nami.interpolants.transforms`` module:

    * ``DriftFromVelocityScore``: ``u = v - \gamma(t)\dot{\gamma}(t) s``
      (probability-flow drift).
    * ``MarkovizationDriftFromVelocityScore``: ``b = v + (-\gamma(t)\dot{\gamma}(t)
      + \frac{1}{2} g(t)^2) s`` (markovisation SDE drift, with ``g(t)^2`` an
      independent diffusion coefficient).

    A future PR will introduce concrete implementations alongside the
    Process that consumes them; the Protocol signature here is the
    contract those implementations must satisfy.
    """

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined runtime quantity at ``(x, t)``."""
        ...

    @property
    def event_ndim(self) -> int | None:
        """Event-tensor rank, matching the underlying head conventions."""
        ...


class DriftFromVelocityScore(nn.Module):
    """Probability-flow drift from separately trained velocity and score heads.

    Computes ``u(x, t) = v(x, t) - gamma(t) * gamma_dot(t) * s(x, t)``.
    """

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        gamma_schedule: GammaSchedule,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule
        _validate_event_ndim(velocity_model, score_model)

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.velocity_model, "event_ndim", None)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_val = self.velocity_model(x, t, c)
        s_val = self.score_model(x, t, c)
        coeff = self.gamma_schedule.gamma_gamma_dot(t)
        coeff = expand_like(coeff, s_val)
        return v_val - coeff * s_val


class MarkovizationDriftFromVelocityScore(nn.Module):
    """Markovization SDE drift from velocity and score heads.

    Computes ``b(x, t) = v(x, t) + (-gamma*gamma_dot + 0.5*g(t)^2) * s(x, t)``.
    ``diffusion2`` is the squared diffusion coefficient ``g(t)^2`` as either a
    constant or a callable ``(t) -> Tensor``.
    """

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        gamma_schedule: GammaSchedule,
        *,
        diffusion2: float | Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule
        self.diffusion2 = diffusion2
        if isinstance(diffusion2, (int, float)):
            self._diffusion2_fn = None
            self._diffusion2_const = float(diffusion2)
        else:
            self._diffusion2_fn: Callable[[torch.Tensor], torch.Tensor] | None = (
                diffusion2
            )
            self._diffusion2_const: float | None = None
        _validate_event_ndim(velocity_model, score_model)

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.velocity_model, "event_ndim", None)

    def _diffusion2(self, t: torch.Tensor) -> torch.Tensor:
        if self._diffusion2_fn is not None:
            return self._diffusion2_fn(t)
        diffusion2_const = self._diffusion2_const
        if diffusion2_const is None:
            msg = "diffusion2 constant is unexpectedly unset"
            raise RuntimeError(msg)
        return torch.full_like(t, diffusion2_const)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_val = self.velocity_model(x, t, c)
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        g2_val = self._diffusion2(t)
        coeff = -gg_val + 0.5 * g2_val
        coeff = expand_like(coeff, s_val)
        return v_val + coeff * s_val


def _validate_event_ndim(velocity_model, score_model) -> None:
    velocity_event_ndim = getattr(velocity_model, "event_ndim", None)
    score_event_ndim = getattr(score_model, "event_ndim", None)
    if (
        velocity_event_ndim is not None
        and score_event_ndim is not None
        and int(velocity_event_ndim) != int(score_event_ndim)
    ):
        msg = (
            "velocity_model.event_ndim and score_model.event_ndim must match; "
            f"got {velocity_event_ndim} and {score_event_ndim}"
        )
        raise ValueError(msg)
