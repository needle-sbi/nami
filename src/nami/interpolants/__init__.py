"""Marginal-path interpolants between source and target distributions.

Unifies the flow-matching, stochastic-interpolant, and bridge-matching
path families under one ``Interpolant`` protocol that dispatches to
target objects (Velocity, Score, Epsilon, X0, VPrediction, Action,
GeneratorParams) per :mod:`nami.parameterizations`.

References
----------
- Lipman et al., *Flow Matching for Generative Modeling*, 2022
  (arXiv:2210.02747).
- Albergo, Boffi, Vanden-Eijnden, *Stochastic Interpolants: A Unifying
  Framework*, 2023 (arXiv:2303.08797).
- Liu et al., *Rectified Flow*, 2022 (arXiv:2209.03003).
- Peluchetti, *Diffusion Bridge Mixture Transports*, 2023.
"""

from __future__ import annotations

from nami.interpolants.bridge import BrownianBridgeInterpolant
from nami.interpolants.cosine import CosineInterpolant
from nami.interpolants.gamma import (
    BrownianGamma,
    GammaSchedule,
    ScaledBrownianGamma,
    ZeroGamma,
)
from nami.interpolants.gaussian import (
    GaussianInterpolant,
    epsilon_prediction,
    score_prediction,
    v_prediction,
    x0_prediction,
)
from nami.interpolants.linear import (
    LinearInterpolant,
    StochasticLinearInterpolant,
    velocity_prediction,
)
from nami.interpolants.protocol import Interpolant, InterpolantState

__all__ = [
    "BrownianBridgeInterpolant",
    "BrownianGamma",
    "CosineInterpolant",
    "GammaSchedule",
    "GaussianInterpolant",
    "Interpolant",
    "InterpolantState",
    "LinearInterpolant",
    "ScaledBrownianGamma",
    "StochasticLinearInterpolant",
    "ZeroGamma",
    "epsilon_prediction",
    "score_prediction",
    "v_prediction",
    "velocity_prediction",
    "x0_prediction",
]
