from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .gamma import GammaSchedule

# Based on https://github.com/malbergo/stochastic-interpolants/tree/main [https://arxiv.org/abs/2303.08797 Albergo et al.]


def _expand_time_like(
    scale: torch.Tensor, target: torch.Tensor, event_ndim: int | None
) -> torch.Tensor:
    if event_ndim is None:
        while scale.ndim < target.ndim:
            scale = scale.unsqueeze(-1)
        return scale

    lead_ndim = target.ndim - event_ndim
    if scale.ndim > lead_ndim:
        return scale

    n_prepend = lead_ndim - scale.ndim
    shape = (1,) * n_prepend + tuple(scale.shape) + (1,) * event_ndim
    return scale.reshape(shape)


class ScoreFromNoise(nn.Module):
    """Convert a noise-prediction model eta(x, t) into a score model s(x, t).

    Parameters
    ----------
    eta_model : nn.Module
        The noise prediction model that takes (x, t, c) and returns noise.
    gamma_schedule : [GammaSchedule]
        The noise schedule for converting noise to score.
    eps : float, optional
        Small epsilon value to prevent division by zero, by default 1e-12.

    Attributes
    ----------
    eta_model : nn.Module
        The wrapped noise prediction model.
    gamma_schedule : GammaSchedule
        The noise schedule.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self, eta_model: nn.Module, gamma_schedule: GammaSchedule, eps: float = 1e-12
    ):
        super().__init__()
        self.eta_model = eta_model
        self.gamma_schedule = gamma_schedule
        self.eps = float(eps)

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.eta_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        eta_val = self.eta_model(x, t, c)
        gamma_val = torch.clamp(self.gamma_schedule.gamma(t), min=self.eps)
        gamma_val = _expand_time_like(gamma_val, eta_val, self.event_ndim)
        return eta_val / gamma_val


class DriftFromVelocityScore(nn.Module):
    """Combine velocity and score into probability-flow drift.

    Computes ``u(x, t) = v(x, t) - gamma(t) * gamma_dot(t) * s(x, t)``.
    For markovization SDE drift use ``MarkovizationDriftFromVelocityScore``.

    Parameters
    ----------
    velocity_model : nn.Module
        The velocity field model v(x, t, c).
    score_model : nn.Module
        The score field model s(x, t, c).
    gamma_schedule : GammaSchedule
        The noise schedule providing gamma(t) and gamma_dot(t).

    Attributes
    ----------
    velocity_model : nn.Module
        The velocity field model.
    score_model : nn.Module
        The score field model.
    gamma_schedule : GammaSchedule
        The noise schedule.
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

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.velocity_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the probability-flow drift.

        Parameters
        ----------
        x : torch.Tensor
            The state variable.
        t : torch.Tensor
            The time variable.
        c : torch.Tensor or None, optional
            Optional conditioning information, by default None.

        Returns
        -------
        torch.Tensor
            The drift value u(x, t) = v(x, t) - gamma*gamma_dot*s(x, t).
        """
        v_val = self.velocity_model(x, t, c)
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        gg_val = _expand_time_like(gg_val, s_val, self.event_ndim)
        return v_val - gg_val * s_val


class MarkovizationDriftFromVelocityScore(nn.Module):
    """Combine velocity and score into markovization SDE drift.

    Computes
    ``b(x, t) = u(x, t) + 0.5 * g(t)^2 * s(x, t)``,
    where ``u(x, t) = v(x, t) - gamma(t) * gamma_dot(t) * s(x, t)``.
    Equivalently:
    ``b(x, t) = v(x, t) + (-gamma*gamma_dot + 0.5*g^2) * s(x, t)``.

    ``diffusion2`` is the squared diffusion coefficient ``g(t)^2``, given as a
    constant ``float`` or a plain callable ``(Tensor) -> Tensor``.  It is **not**
    registered as an ``nn.Module`` submodule, so learnable diffusion schedules
    should be managed separately.
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

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.velocity_model, "event_ndim", None)

    def _diffusion2(self, t: torch.Tensor) -> torch.Tensor:
        if callable(self.diffusion2):
            return self.diffusion2(t)
        return torch.full_like(t, float(self.diffusion2))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        v_val = self.velocity_model(x, t, c)
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        g2_val = self._diffusion2(t)
        coeff = -gg_val + 0.5 * g2_val
        coeff = _expand_time_like(coeff, s_val, self.event_ndim)
        return v_val + coeff * s_val


class MirrorVelocityFromScore(nn.Module):
    """Create mirror-flow velocity v_mirror(x, t) = gamma*gamma_dot*s(x, t)."""

    def __init__(self, score_model: nn.Module, gamma_schedule: GammaSchedule):
        super().__init__()
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.score_model, "event_ndim", None)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        s_val = self.score_model(x, t, c)
        gg_val = self.gamma_schedule.gamma_gamma_dot(t)
        gg_val = _expand_time_like(gg_val, s_val, self.event_ndim)
        return gg_val * s_val
