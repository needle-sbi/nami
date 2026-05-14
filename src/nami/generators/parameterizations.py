from __future__ import annotations

"""Parameterization factory for generator-matching training.

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


from nami.parameterizations import GeneratorParams, Parameterization
from nami.generators.base import GeneratorOperator


def generator_prediction(operator: GeneratorOperator) -> Parameterization:
    """Bind a generator operator into a :class:`Parameterization`.

    Replaces the legacy ``operator.project(field(x, t, c))`` plumbing
    inside ``gm_loss``: the equivalent at the unified-vocabulary layer
    is ``Parameterization(target=GeneratorParams(operator=op),
    output_transform=op.project)``, and this factory is the
    one-liner that produces it.
    """
    return Parameterization(
        target=GeneratorParams(operator=operator),
        output_transform=operator.project,
    )
