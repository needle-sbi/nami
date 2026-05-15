r"""Cosine-scheduled deterministic interpolant.

``x_t = \alpha(t) x_noise + \sigma(t) x_data`` with ``\alpha(t) = \cos(\pi t/2)`` and
``\sigma(t) = \sin(\pi t/2)``. At ``t=0`` the path is pure noise
(``\alpha(0)=1, \sigma(0)=0``); at ``t=1`` it is pure data
(``\alpha(1)=0, \sigma(1)=1``). The conditional velocity is

    ``u_t = \alpha'(t) x_noise + \sigma'(t) x_data``

with ``\alpha'(t) = -\pi/2 \cdot \sin(\pi t/2)`` and ``\sigma'(t) = \pi/2 \cdot \cos(\pi t/2)``,
so unlike :class:`~nami.interpolants.linear.LinearInterpolant` the
velocity is *t-dependent*.

The path is deterministic (no noise term), so ``Score``, ``Epsilon``,
``X0`` and ``VPrediction`` raise ``NotImplementedError`` for the same reasons as
``LinearInterpolant``.  Use ``GaussianInterpolant`` if those targets
are required.
"""

from __future__ import annotations

import math
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
class CosineInterpolant:
    r"""Deterministic cosine-scheduled interpolant.

    Supports the :class:`~nami.parameterizations.Velocity` target with the
    closed-form derivative of the cosine path.
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
                "CosineInterpolant is deterministic; passing noise= has no "
                "effect.  Use GaussianInterpolant for stochastic paths."
            )
            raise ValueError(msg)
        a = _broadcast_t(self._alpha(t), x_data)
        s = _broadcast_t(self._sigma(t), x_data)
        xt = a * x_noise + s * x_data
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
                ap = _broadcast_t(self._alpha_prime(state.t), state.x_data)
                sp = _broadcast_t(self._sigma_prime(state.t), state.x_data)
                return ap * state.x_noise + sp * state.x_data
            case Score() | Epsilon() | X0() | VPrediction():
                msg = (
                    f"CosineInterpolant does not support {type(target).__name__}: "
                    "the path is deterministic and has no noise component."
                )
                raise NotImplementedError(msg)
            case Action():
                # \nabla_x s`` regresses against the conditional velocity, which
                # is the same closed-form expression as the Velocity arm.
                ap = _broadcast_t(self._alpha_prime(state.t), state.x_data)
                sp = _broadcast_t(self._sigma_prime(state.t), state.x_data)
                return ap * state.x_noise + sp * state.x_data
            case GeneratorParams():
                msg = (
                    "CosineInterpolant does not support GeneratorParams; "
                    "use an operator-aware interpolant instead."
                )
                raise NotImplementedError(msg)
        assert_never(target)

    @staticmethod
    def _alpha(t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t * math.pi / 2.0)

    @staticmethod
    def _sigma(t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t * math.pi / 2.0)

    @staticmethod
    def _alpha_prime(t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2.0) * torch.sin(t * math.pi / 2.0)

    @staticmethod
    def _sigma_prime(t: torch.Tensor) -> torch.Tensor:
        return (math.pi / 2.0) * torch.cos(t * math.pi / 2.0)
