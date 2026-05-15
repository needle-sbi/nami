r"""Composite fields built from separately trained heads.

These modules combine a velocity head ``v(x,t)`` and score head
``s(x,t) = \nabla_x \log p_t(x)`` with a stochastic-interpolant noise
schedule ``\gamma(t)``.  The result is a runtime vector field suitable for
probability-flow or Markovian SDE sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from nami.diffusion import expand_like
from nami.interpolants.gamma import GammaSchedule


@runtime_checkable
class TwoHeadField(Protocol):
    r"""Field that combines two trained models into one runtime quantity.

    Implementations consume two networks, usually a velocity head and a score
    head, and emit a vector field with shape ``lead + event_shape``.
    """


    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the composite field.

        Args:
            x (torch.Tensor): State tensor with shape ``lead + event_shape``.
            t (torch.Tensor): Time tensor broadcastable to ``lead``.
            c (torch.Tensor | None): Optional conditioning tensor.

        Returns:
            torch.Tensor: Composite vector field with the same shape as ``x``.
        """
        ...

    @property
    def event_ndim(self) -> int | None:
        """int | None: Number of trailing event dimensions."""
        ...


class DriftFromVelocityScore(nn.Module):
    r"""Probability-flow drift from separately trained velocity and score heads.

    Args:
        velocity_model (nn.Module): Model returning ``v(x,t)``.
        score_model (nn.Module): Model returning ``s(x,t)``.
        gamma_schedule (GammaSchedule): Schedule providing ``\gamma(t)`` and
            ``\dot{\gamma}(t)``.

    The evaluated drift is

    .. math::

       u(x,t) = v(x,t) - \gamma(t)\dot{\gamma}(t)s(x,t).
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
    r"""Markovization SDE drift from velocity and score heads.

    Args:
        velocity_model (nn.Module): Model returning ``v(x,t)``.
        score_model (nn.Module): Model returning ``s(x,t)``.
        gamma_schedule (GammaSchedule): Schedule providing ``\gamma(t)`` and
            ``\dot{\gamma}(t)``.
        diffusion2 (float | Callable[[torch.Tensor], torch.Tensor]): Squared
            diffusion coefficient ``g^2(t)``.

    The evaluated drift is

    .. math::

       b(x,t) = v(x,t) +
       \left[-\gamma(t)\dot{\gamma}(t) + \frac{1}{2}g^2(t)\right]s(x,t).
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
