r"""Brownian bridge interpolant - the headline duplication collapse.

The legacy code carries this path in *two* parallel hierarchies:

* :class:`nami.paths.bridge.BrownianBridgePath` (a ``ProbabilityPath``)
  exposes ``sample_xt`` plus ``target_ut`` (velocity) and
  ``score_target`` (Stein score).
* :class:`nami.generators.paths.BrownianGeneratorPath` (a ``GeneratorPath``)
  exposes the *same* ``sample_xt`` byte-for-byte, plus ``target_params``
  for an :class:`~nami.generators.operators.ItoGeneratorOperator`.

Both are encodings of the same Brownian bridge, expressed in different
target dialects.  ``BrownianBridgeInterpolant`` collapses them: one
interpolant, one ``sample`` method, and a single ``target`` dispatch
that produces velocity, score, *and* generator parameters from the
same state.

The conditional-velocity formula and the score formula are inherited
from Albergo-Vanden-Eijnden's stochastic-interpolant convention with a
``\sigma``-scaled Brownian-bridge noise term; the generator-parameter formula
matches the conditional drift used in nami's existing
``BrownianGeneratorPath`` so the GM training objective is preserved
exactly.
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

    ``x_t = (1-t) x_noise + t x_data + \sigma \sqrt{t (1-t)} z``

    where ``z ~ N(0, I)`` is supplied via ``noise=`` to keep sampling
    deterministic across calls (e.g. for the equivalence tests against
    legacy paths).  When ``noise=None`` is passed, fresh Gaussian noise
    is drawn internally.

    Supported targets:

    * :class:`Velocity` — conditional velocity ``u_t(x_t)`` with the
      Brownian-bridge correction term, matching the legacy
      ``BrownianBridgePath.target_ut(..., xt=...)``.
    * :class:`Score` — Stein score ``\nabla \log p_t(x_t)``, matching
      ``BrownianBridgePath.score_target``.
    * :class:`GeneratorParams` — packed drift (and optional diffusion)
      for an ``ItoGeneratorOperator``, matching
      ``BrownianGeneratorPath.target_params``.

    :class:`Epsilon` and :class:`X0` raise ``NotImplementedError`` -
    the bridge's noise term is bound to ``z`` rather than a
    standardised ``\epsilon`` with a clean ``\sigma``-``\alpha`` decomposition, so those targets
    have no canonical formula here.

    Endpoint singularities at ``t=0`` and ``t=1`` (both Velocity and
    Score divide by ``t(1-t)``) are guarded by an ``eps`` floor that
    matches the legacy path's clamp; ``regression_loss`` callers
    typically also pass ``eps_t`` for sampling discipline.
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
    # Target dispatch.  Each named target reuses the legacy formula
    # exactly; the equivalence tests in tests/interpolants/test_bridge.py
    # pin that the migration is bit-exact.

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
