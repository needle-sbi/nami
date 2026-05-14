"""Curated public API for ``nami``.

The user story is three nouns and one flow:

* **Interpolant + Parameterization -> Loss**
* **Field + Parameterization -> Process**
* **Process -> sample / log_prob**

Top-level imports are restricted to the symbols a user composes those
three sentences with.  Implementation details (lazy adapters, base
protocols, building-block components, internal helpers) live under
their submodules — ``nami.lazy``, ``nami.processes``, ``nami.divergence``,
``nami.components`` — and are reachable from there.

``bridge_matching_loss``, ``masked_fm_loss``, and ``stochastic_fm_loss``
have been migrated to the unified vocabulary and live at their
submodules; they are not re-exported here because they are
specialised variants of ``regression_loss`` for specific use cases
(Brownian bridge two-head training, masked variable-cardinality
inputs, stochastic linear interpolants).
"""
from __future__ import annotations

try:
    from nami._version import version as __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

from nami import diffusion as diffusion  # exposed as ``nami.diffusion`` (submodule)

# ---------------------------------------------------------------------------
# Distributions, solvers, divergence — minimal user-facing surface
# ---------------------------------------------------------------------------
from nami.distributions.normal import DiagonalNormal, StandardNormal
from nami.divergence.exact import ExactDivergence
from nami.divergence.hutchinson import HutchinsonDivergence

# ---------------------------------------------------------------------------
# Fields — bases users build their networks on top of
# ---------------------------------------------------------------------------
from nami.fields.action import ActionHead
from nami.fields.adaln import AdaLNVelocityField
from nami.fields.composite import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
)
from nami.fields.consistency import LogDensityHead
from nami.fields.generator import GeneratorField
from nami.fields.transformer_velocity import TransformerVelocityField
from nami.fields.velocity import VelocityField

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
from nami.generators.base import GeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
from nami.generators.parameterizations import generator_prediction

# ---------------------------------------------------------------------------
# Interpolants — concrete implementations
# ---------------------------------------------------------------------------
from nami.interpolants.bridge import BrownianBridgeInterpolant
from nami.interpolants.cosine import CosineInterpolant
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

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
from nami.losses.action import action_matching_loss, action_prediction
from nami.losses.consistency import consistency_loss
from nami.losses.log_density import log_density_consistency_loss
from nami.losses.regression import regression_loss

# ---------------------------------------------------------------------------
# Vocabulary: targets, parameterizations
# ---------------------------------------------------------------------------
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Velocity,
    VPrediction,
)

# ---------------------------------------------------------------------------
# Processes — the four high-level constructors users instantiate
# ---------------------------------------------------------------------------
from nami.processes.action import ActionMatching
from nami.processes.consistency import ConsistencyFlowMatching
from nami.processes.diffusion import Diffusion
from nami.processes.fm import FlowMatching
from nami.processes.gm import GeneratorMatching

# ---------------------------------------------------------------------------
# Schedules — diffusion building blocks
# ---------------------------------------------------------------------------
from nami.schedules.edm import EDMSchedule
from nami.schedules.ve import VESchedule
from nami.schedules.vp import VPSchedule
from nami.solvers.dpm import DPMSolverPP
from nami.solvers.heun import Heun
from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama

__all__ = [
    "RK4",
    "X0",
    "Action",
    "ActionHead",
    "ActionMatching",
    "AdaLNVelocityField",
    "BrownianBridgeInterpolant",
    "ConsistencyFlowMatching",
    "CosineInterpolant",
    "DPMSolverPP",
    "DiagonalNormal",
    "Diffusion",
    "DriftFromVelocityScore",
    "EDMSchedule",
    "Epsilon",
    "EulerMaruyama",
    "ExactDivergence",
    "FlowMatching",
    "GaussianInterpolant",
    "GeneratorField",
    "GeneratorMatching",
    "GeneratorOperator",
    "GeneratorParams",
    "Heun",
    "HutchinsonDivergence",
    "ItoGeneratorOperator",
    "LinearInterpolant",
    "LogDensityHead",
    "MarkovizationDriftFromVelocityScore",
    "Parameterization",
    "Score",
    "StandardNormal",
    "StochasticLinearInterpolant",
    "TransformerVelocityField",
    "VESchedule",
    "VPSchedule",
    "VPrediction",
    "Velocity",
    "VelocityField",
    "__version__",
    "action_matching_loss",
    "action_prediction",
    "consistency_loss",
    "epsilon_prediction",
    "generator_prediction",
    "log_density_consistency_loss",
    "regression_loss",
    "score_prediction",
    "v_prediction",
    "velocity_prediction",
    "x0_prediction",
]
