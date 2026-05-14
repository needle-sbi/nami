from __future__ import annotations

"""Cosine-scheduled deterministic interpolant.

``x_t = \alpha(t) x_target + \sigma(t) x_source`` with ``\alpha(t) = \cos(\pi t/2)`` and
``\sigma(t) = \sin(\pi t/2)``. The conditional velocity is

    ``u_t = \alpha'(t) x_target + \sigma'(t) x_source``

with ``\alpha'(t) = -\pi/2 \cdot \sin(\pi t/2)`` and ``\sigma'(t) = \pi/2 \cdot \cos(\pi t/2)``,
so unlike :class:`~nami.interpolants.linear.LinearInterpolant` the
velocity is *t-dependent*.

The path is deterministic (no noise term), so ``Score``, ``Epsilon``,
``X0`` and ``VPrediction`` raise ``NotImplementedError`` for the same reasons as
``LinearInterpolant``.  Use ``GaussianInterpolant`` if those targets
are required.
"""



import math
from dataclasses import dataclass
from typing import assert_never

import torch

from nami.parameterizations import (
    Action,
    Epsilon,
    GeneratorParams,
    Score,
    Target,
    TensorLike,
    VPrediction,
    Velocity,
    X0,
)
from nami.interpolants._common import broadcast_t as _broadcast_t
from nami.interpolants.protocol import InterpolantState


@dataclass(frozen=True)
class CosineInterpolant:
    """Deterministic cosine-scheduled interpolant.

    Replaces the legacy :class:`~nami.paths.cosine.CosinePath` on the
    unified vocabulary.  Same closed-form math; supports the
    :class:`~nami.parameterizations.Velocity` target.
    """

    def sample(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
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
        a = _broadcast_t(self._alpha(t), x_target)
        s = _broadcast_t(self._sigma(t), x_target)
        xt = a * x_target + s * x_source
        return InterpolantState(
            xt=xt, x_target=x_target, x_source=x_source, t=t, noise=None,
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Velocity():
                ap = _broadcast_t(self._alpha_prime(state.t), state.x_target)
                sp = _broadcast_t(self._sigma_prime(state.t), state.x_target)
                return ap * state.x_target + sp * state.x_source
            case Score() | Epsilon() | X0() | VPrediction():
                msg = (
                    f"CosineInterpolant does not support {type(target).__name__}: "
                    "the path is deterministic and has no noise component."
                )
                raise NotImplementedError(msg)
            case Action():
                # ``\nabla_x s`` regresses against the conditional velocity, which
                # is the same closed-form expression as the Velocity arm.
                ap = _broadcast_t(self._alpha_prime(state.t), state.x_target)
                sp = _broadcast_t(self._sigma_prime(state.t), state.x_target)
                return ap * state.x_target + sp * state.x_source
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
