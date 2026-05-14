"""Generator operators for generator-matching transports.

An operator interprets a field's raw output as packed parameters
(drift, optional diffusion, jump rates, ...) of a continuous-time
Markov generator. The field stays generic; the operator carries the
semantics.

References
----------
- Holderrieth et al., *Generator Matching*, 2024.
"""

from __future__ import annotations


from nami.generators.base import GeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
from nami.generators.parameterizations import generator_prediction

__all__ = [
    "GeneratorOperator",
    "ItoGeneratorOperator",
    "generator_prediction",
]
