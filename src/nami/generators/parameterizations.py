r"""Parameterization factory for generator-matching training.

The factory binds the :class:`~nami.parameterizations.GeneratorParams`
target to the operator's projection (``operator.project``), so the raw
network output is constrained to the operator's parameter manifold
(e.g. positive softplus diffusion for
:class:`~nami.generators.operators.ItoGeneratorOperator`).

The default weighting is uniform ``\omega(t) = 1``. GM does not have a
canonical schedule-dependent weighting analogous to diffusion's
``\mathrm{SNR}/\sigma^2`` conventions; if a future generator family introduces one, it
will live in its own factory rather than overloading this one.
"""

from __future__ import annotations

from nami.generators.base import GeneratorOperator
from nami.parameterizations import GeneratorParams, Parameterization


def generator_prediction(operator: GeneratorOperator) -> Parameterization:
    """Create a generator-parameter prediction parameterization.

    Args:
        operator (GeneratorOperator): Operator that interprets the packed
            parameter tensor.

    Returns:
        Parameterization: Target ``GeneratorParams(operator=operator)`` with
        ``operator.project`` as ``output_transform``.
    """
    return Parameterization(
        target=GeneratorParams(operator=operator),
        output_transform=operator.project,
    )
