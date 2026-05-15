r"""Linear interpolants - deterministic and stochastic variants.

``LinearInterpolant``:
    ``x_t = (1 - t) \cdot x_noise + t \cdot x_data`` - the deterministic
    linear path between noise (``t=0``) and data (``t=1``).
    Supports :class:`Velocity` (constant ``u_t = x_data - x_noise``)
    and :class:`GeneratorParams` for an Ito operator.

``StochasticLinearInterpolant``:
    ``x_t = (1 - t) \cdot x_noise + t \cdot x_data + \gamma(t) \cdot z`` - the
    Albergo-Vanden-Eijnden stochastic interpolant variant.  Adds a
    gamma-scaled Gaussian noise term to the deterministic linear path so
    the conditional density has a well-defined score.  Supports
    :class:`Velocity` (with the ``\dot{\gamma}\cdot z`` correction term).

Stochastic targets on the deterministic ``LinearInterpolant``
(``Score``, ``Epsilon``, ``X0``) raise ``NotImplementedError`` -
the path has no noise component, so its conditional density is a
delta function.  Use :class:`StochasticLinearInterpolant` or
:class:`~nami.interpolants.gaussian.GaussianInterpolant` for those
targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_never

import torch

from nami.interpolants._common import broadcast_t as _broadcast_t
from nami.interpolants.gamma import BrownianGamma, GammaSchedule
from nami.interpolants.protocol import InterpolantState
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Target,
    TensorLike,
    Velocity,
    VPrediction,
)


@dataclass(frozen=True)
class LinearInterpolant:
    r"""Deterministic linear interpolant ``x_t = (1-t) x_noise + t x_data``.

    Implements the :class:`~nami.interpolants.protocol.Interpolant`
    protocol.  Supports only :class:`~nami.parameterizations.Velocity`;
    other targets raise ``NotImplementedError`` because the path is
    deterministic.
    """

    def sample(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState:
        if noise is not None:
            msg = (
                "LinearInterpolant is deterministic; passing noise= has no "
                "effect.  Use GaussianInterpolant for stochastic paths."
            )
            raise ValueError(msg)
        tt = _broadcast_t(t, x_data)
        xt = (1.0 - tt) * x_noise + tt * x_data
        return InterpolantState(
            xt=xt,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            noise=None,
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Velocity():
                return state.x_data - state.x_noise
            case Score() | Epsilon() | X0() | VPrediction():
                msg = (
                    f"LinearInterpolant does not support {type(target).__name__}: "
                    "the path is deterministic and has no noise component "
                    "(VPrediction also needs a NoiseSchedule, which the linear path "
                    "does not carry)."
                )
                raise NotImplementedError(msg)
            case Action():
                # Action regresses ``\nabla_x s(x, t)`` against the conditional
                # velocity; for the deterministic linear path that velocity
                # is identical to the Velocity target arm.
                return state.x_data - state.x_noise
            case GeneratorParams(operator=op):
                # Drift is identical to the Velocity target — the linear
                # path's velocity *is* the conditional generator drift
                # for an Ito operator.  Diffusion is zero because the
                # path itself is deterministic.  Replaces the legacy
                # ``LinearGeneratorPath.target_params`` body verbatim.
                drift = state.x_data - state.x_noise
                if getattr(op, "diffusion_mode", "none") == "none":
                    return op.pack_params(drift=drift)
                diffusion = torch.zeros_like(state.x_data)
                return op.pack_params(drift=drift, diffusion=diffusion)
        # ``assert_never`` gives both halves of the discipline:
        # a static checker (mypy/pyright) flags a missing match arm
        # as a type error, *and* the runtime fails crisply instead
        # of silently returning ``None`` if a non-Target value slips
        # through.
        assert_never(target)


def velocity_prediction() -> Parameterization:
    r"""Velocity-prediction with ``\omega(t) = 1`` — the standard FM convention.

    No schedule argument because deterministic linear paths have none;
    if a future Velocity-supporting interpolant introduces a non-trivial
    weighting, that factory will take its own schedule and live
    alongside this one.
    """
    return Parameterization(target=Velocity())


@dataclass(frozen=True)
class StochasticLinearInterpolant:
    r"""Albergo-Vanden-Eijnden stochastic-linear interpolant.

    ``x_t = (1 - t) x_noise + t x_data + \gamma(t) z`` with ``z \sim N(0, I)``.

    Adds a ``\gamma``-scaled Gaussian noise term on top of the deterministic
    linear interpolation so the conditional density at intermediate
    ``t`` has a well-defined score.  ``z`` is sampled internally when
    :meth:`sample` is called with ``noise=None``; pass an explicit ``z``
    via ``noise=`` for reproducibility (used by losses that need to
    pair two trajectory points or share noise across heads).

    Supported targets:

    * :class:`~nami.parameterizations.Velocity` — conditional velocity
      with the ``\dot{\gamma}\cdot z`` correction:
      ``u_t = (x_data - x_noise) + \dot{\gamma}(t) z``.

    Score / Epsilon / X0 raise ``NotImplementedError`` — the noise
    variable here is ``z`` (the bridge increment), not a standardised ``\epsilon``
    with a clean ``\alpha/\sigma`` decomposition. Use ``GaussianInterpolant`` when
    those targets are required.

    The default ``gamma`` is :class:`~nami.interpolants.gamma.BrownianGamma`
    matching the legacy ``stochastic_fm_loss`` default.
    """

    gamma: GammaSchedule = field(default_factory=BrownianGamma)

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
        gamma_t = _broadcast_t(self.gamma.gamma(t), mu)
        xt = mu + gamma_t * noise
        return InterpolantState(
            xt=xt,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            noise=noise,
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Velocity():
                # ``u_t = (x_data - x_noise) + \dot{\gamma}(t) \cdot z``
                gamma_dot = _broadcast_t(self.gamma.gamma_dot(state.t), state.x_data)
                if state.noise is None:
                    msg = (
                        "StochasticLinearInterpolant.target(Velocity) requires "
                        "InterpolantState.noise; sample with noise=z to make "
                        "the conditional velocity well-defined."
                    )
                    raise ValueError(msg)
                return (state.x_data - state.x_noise) + gamma_dot * state.noise
            case Action():
                # Same conditional velocity as the Velocity arm: the
                # action-matching loss regresses ``\nabla_x s`` against ``u_t`` and
                # ``u_t`` carries the ``\dot{\gamma}\cdot z`` correction term.
                gamma_dot = _broadcast_t(self.gamma.gamma_dot(state.t), state.x_data)
                if state.noise is None:
                    msg = (
                        "StochasticLinearInterpolant.target(Action) requires "
                        "InterpolantState.noise; sample with noise=z so the "
                        "conditional velocity (and hence the action gradient "
                        "target) is well-defined."
                    )
                    raise ValueError(msg)
                return (state.x_data - state.x_noise) + gamma_dot * state.noise
            case Score() | Epsilon() | X0() | VPrediction():
                msg = (
                    f"StochasticLinearInterpolant does not support {type(target).__name__}: "
                    "the noise variable is z (the bridge increment), not a "
                    "standardised \\epsilon with a clean \\alpha/\\sigma decomposition. Use "
                    "GaussianInterpolant if you need those targets."
                )
                raise NotImplementedError(msg)
            case GeneratorParams():
                msg = (
                    "StochasticLinearInterpolant does not support GeneratorParams; "
                    "use an operator-aware interpolant instead."
                )
                raise NotImplementedError(msg)
        assert_never(target)
