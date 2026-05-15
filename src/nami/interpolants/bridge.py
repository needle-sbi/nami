r"""Brownian-bridge interpolant.

The marginal path is

.. math::

   x_t = (1-t)x_{\mathrm{noise}} + t x_{\mathrm{data}}
         + \sigma\sqrt{t(1-t)}z,

where ``z`` is standard Gaussian noise.  The interpolant exposes a single
``sample`` method and a target dispatcher for velocity, score, and generator
parameter targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch

from nami.interpolants._common import broadcast_t as _broadcast_t
from nami.interpolants.protocol import InterpolantState
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Score,
    Target,
    TensorLike,
    Velocity,
    VPrediction,
)


@dataclass(frozen=True)
class BrownianBridgeInterpolant:
    r"""Stochastic Brownian-bridge interpolant.

    Args:
        sigma (float): Bridge noise scale ``\sigma``.
        eps (float): Positive floor applied to endpoint denominators.

    The path is

    .. math::

       x_t = (1-t)x_{\mathrm{noise}} + t x_{\mathrm{data}}
             + \sigma\sqrt{t(1-t)}z.

    When ``noise=None`` is passed to :meth:`sample`, fresh Gaussian noise is
    drawn internally.

    Supported targets:

    * :class:`Velocity` — conditional velocity ``u_t(x_t)`` with the
      Brownian-bridge correction term.
    * :class:`Score` — Stein score ``\nabla_x \log p_t(x_t)``.
    * :class:`GeneratorParams` — packed drift (and optional diffusion)
      for an ``ItoGeneratorOperator``.

    :class:`Epsilon` and :class:`X0` raise ``NotImplementedError`` -
    the bridge's noise term is bound to ``z`` rather than a
    standardised ``\epsilon`` with a clean ``\sigma``-``\alpha`` decomposition, so those targets
    have no canonical formula here.

    Endpoint singularities at ``t=0`` and ``t=1`` (both Velocity and
    Score divide by ``t(1-t)``) are guarded by ``eps``.  Loss callers should
    still avoid exact endpoints when sampling ``t``.
    """

    sigma: float = 1.0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        if self.eps <= 0:
            msg = "eps must be positive"
            raise ValueError(msg)
        if self.eps >= 0.5:
            msg = "eps must be < 0.5"
            raise ValueError(msg)

    def sample(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState:
        tt = _broadcast_t(t, x_data)
        mu = (1.0 - tt) * x_noise + tt * x_data
        if noise is None:
            noise = torch.randn_like(mu)
        std = self.sigma * torch.sqrt(tt * (1.0 - tt))
        xt = mu + std * noise
        return InterpolantState(
            xt=xt,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            noise=noise,
        )

    # ------------------------------------------------------------------
    # Target dispatch.  Each named target is derived from the sampled bridge
    # state so losses share the same formulas.

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Velocity():
                return self._velocity(state)
            case Score():
                return self._score(state)
            case GeneratorParams(operator=op):
                drift = self._velocity(state)
                if getattr(op, "diffusion_mode", "none") == "none":
                    return op.pack_params(drift=drift)
                diffusion = torch.full_like(state.x_data, self.sigma)
                return op.pack_params(drift=drift, diffusion=diffusion)
            case Epsilon() | X0() | VPrediction():
                msg = (
                    f"BrownianBridgeInterpolant does not support {type(target).__name__}: "
                    "the bridge's noise variable is z (Brownian-bridge increment), "
                    "not a standardised \\epsilon with a clean \\alpha/\\sigma decomposition "
                    "(VPrediction additionally needs a NoiseSchedule)."
                )
                raise NotImplementedError(msg)
            case Action():
                # ``\nabla_x s`` regresses against the conditional bridge velocity.
                return self._velocity(state)
        assert_never(target)

    # ------------------------------------------------------------------

    def _velocity(self, state: InterpolantState) -> torch.Tensor:
        x_noise, x_data = state.x_noise, state.x_data
        tt = _broadcast_t(state.t, x_data)
        mu = (1.0 - tt) * x_noise + tt * x_data
        denom = 2.0 * torch.clamp(tt * (1.0 - tt), min=self.eps)
        # Sign flip from chain rule on the (1-2t) coefficient under t -> 1-t.
        coeff = (2.0 * tt - 1.0) / denom
        return (x_data - x_noise) + coeff * (state.xt - mu)

    def _score(self, state: InterpolantState) -> torch.Tensor:
        x_noise, x_data = state.x_noise, state.x_data
        tt = _broadcast_t(state.t, x_data)
        mu = (1.0 - tt) * x_noise + tt * x_data
        var = self.sigma**2 * torch.clamp(tt * (1.0 - tt), min=self.eps)
        return (mu - state.xt) / var
