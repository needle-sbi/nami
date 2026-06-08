"""Neural fields: heads consumed by Process classes.

Each field is an ``nn.Module`` honouring the ``forward(x, t, c=None)``
contract with an ``event_ndim`` attribute. Concrete heads cover
velocity prediction (flow matching), scalar action potentials, scalar
log-density heads (consistency models), and operator-parameter heads
(generator matching).
"""

from __future__ import annotations

from nami.fields.action import ActionHead
from nami.fields.adaln import AdaLNVelocityField
from nami.fields.composite import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
    TwoHeadField,
)
from nami.fields.consistency import LogDensityHead
from nami.fields.ctmc import CTMCField
from nami.fields.generator import GeneratorField
from nami.fields.scalar_potential import ScalarPotentialField
from nami.fields.transformer_velocity import TransformerVelocityField
from nami.fields.velocity import VelocityField

__all__ = [
    "ActionHead",
    "AdaLNVelocityField",
    "CTMCField",
    "DriftFromVelocityScore",
    "GeneratorField",
    "LogDensityHead",
    "MarkovizationDriftFromVelocityScore",
    "ScalarPotentialField",
    "TransformerVelocityField",
    "TwoHeadField",
    "VelocityField",
]
