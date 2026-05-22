r"""Typed prediction targets for transport processes.

The :data:`Target` union names the mathematical object emitted by a model:
conditional velocity, score, standardized noise, clean endpoint,
v-prediction, action potential, or packed generator parameters.
:class:`Parameterization` pairs such a target with a loss weighting
``\omega(t)`` and an output projection from raw network values into the target
space.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .generators.base import GeneratorOperator


# ---------------------------------------------------------------------------
# Note: Every variant is a frozen dataclass so it can carry per-target metadata
# (e.g. the GeneratorOperator that interprets GeneratorParams) without
# leaking into string-keyed registries.  Frozen dataclasses also receive
# ``__match_args__``, which makes ``case Variant(field=x):``
# pattern matching work at call sites.


@dataclass(frozen=True)
class Velocity:
    r"""Conditional velocity target.

    The target is the vector field
    ``u_t(x) = \partial_t x_t`` with shape ``lead + event_shape``.
    """


@dataclass(frozen=True)
class Score:
    r"""Score target.

    The target is the Stein score ``\nabla_x \log p_t(x)`` with shape
    ``lead + event_shape``.
    """


@dataclass(frozen=True)
class Epsilon:
    r"""Standardized-noise target.

    For a Gaussian path, the target is ``\epsilon`` in
    ``x_t = \alpha(t)\epsilon + \sigma(t)x_0``.
    """


@dataclass(frozen=True)
class X0:
    r"""Clean-endpoint target ``x_0``."""


@dataclass(frozen=True)
class GeneratorParams:
    r"""Packed generator parameters interpreted by an operator.

    Args:
        operator (GeneratorOperator): Operator that projects, validates, and
            unpacks the emitted parameter tensor.

    The network emits one tensor in the operator's parameter space.  The
    operator maps that tensor to semantically named coefficients such as drift
    ``b(x,t)`` and diffusion ``g(x,t)``.
    """

    operator: GeneratorOperator


@dataclass(frozen=True)
class VPrediction:
    r"""Salimans-Ho v-prediction target.

    The network emits

    .. math::

       v = \alpha(t)\epsilon - \sigma(t)x_0,

    where ``\alpha`` and ``\sigma`` come from a
    :class:`~nami.schedules.base.NoiseSchedule`.  This target is defined for
    interpolants with a Gaussian noise decomposition.
    """


@dataclass(frozen=True)
class Action:
    r"""Action-matching target (Neklyudov et al., 2023).

    The network emits a scalar potential ``s(x,t)`` with shape ``lead``.  The
    supervised vector target is its gradient,

    .. math::

       \nabla_x s(x,t) = u_t(x),

    so action matching uses autograd rather than direct tensor regression on
    the scalar field output.
    """


Target = Velocity | Score | Epsilon | X0 | VPrediction | Action | GeneratorParams
"""Union of all supported prediction-target marker types."""


# ---------------------------------------------------------------------------
# Parameterization

TensorLike = torch.Tensor
"""Tensor type used for concrete target values."""


def _ones_weighting(t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(t)


def _identity_transform(y: TensorLike) -> TensorLike:
    return y


@dataclass(frozen=True)
class Parameterization:
    r"""Prediction target, loss weighting, and output projection.

    Args:
        target (Target): Mathematical object the network predicts.
        weighting (Callable[[torch.Tensor], torch.Tensor]): Per-sample loss
            weight ``\omega(t)``. Runtime sampling ignores this value.
        output_transform (Callable[[TensorLike], TensorLike]): Projection from
            raw network output to target space.

    Cross-target conversions such as ``\epsilon`` to score require ``x_t``,
    ``t``, and schedule values.  Those conversions are handled by process
    classes such as :class:`~nami.processes.diffusion.Diffusion`, not by the
    projection stored here.
    """

    target: Target
    weighting: Callable[[torch.Tensor], torch.Tensor] = field(default=_ones_weighting)
    output_transform: Callable[[TensorLike], TensorLike] = field(
        default=_identity_transform
    )

    @property
    def is_identity_transform(self) -> bool:
        r"""Return whether ``output_transform`` is the identity projection.

        Returns:
            bool: ``True`` when raw network outputs are already in target space.
        """
        return self.output_transform is _identity_transform
