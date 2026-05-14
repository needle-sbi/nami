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

try:
    from nami._version import version as __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

from nami import diffusion as diffusion  # exposed as ``nami.diffusion`` (submodule)

# ---------------------------------------------------------------------------
# Vocabulary: targets, parameterizations
# ---------------------------------------------------------------------------
from nami.parameterizations import (
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    VPrediction,
    Velocity,
    X0,
)

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
# Generators
# ---------------------------------------------------------------------------
from nami.generators.base import GeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
from nami.generators.parameterizations import generator_prediction

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
from nami.losses.action import action_matching_loss, action_prediction
from nami.losses.consistency import consistency_loss
from nami.losses.log_density import log_density_consistency_loss
from nami.losses.regression import regression_loss

# ---------------------------------------------------------------------------
# Processes — the four high-level constructors users instantiate
# ---------------------------------------------------------------------------
from nami.processes.action import ActionMatching
from nami.processes.consistency import ConsistencyFlowMatching
from nami.processes.diffusion import Diffusion
from nami.processes.fm import FlowMatching
from nami.processes.gm import GeneratorMatching

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
# Schedules — diffusion building blocks
# ---------------------------------------------------------------------------
from nami.schedules.edm import EDMSchedule
from nami.schedules.ve import VESchedule
from nami.schedules.vp import VPSchedule

# ---------------------------------------------------------------------------
# Distributions, solvers, divergence — minimal user-facing surface
# ---------------------------------------------------------------------------
from nami.distributions.normal import DiagonalNormal, StandardNormal
from nami.divergence.exact import ExactDivergence
from nami.divergence.hutchinson import HutchinsonDivergence
from nami.solvers.dpm import DPMSolverPP
from nami.solvers.heun import Heun
from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama


__all__ = [
    # Vocabulary
    "Action",
    "Epsilon",
    "GeneratorParams",
    "Parameterization",
    "Score",
    "VPrediction",
    "Velocity",
    "X0",
    # Interpolants
    "BrownianBridgeInterpolant",
    "CosineInterpolant",
    "GaussianInterpolant",
    "LinearInterpolant",
    "StochasticLinearInterpolant",
    # Parameterization factories
    "action_prediction",
    "epsilon_prediction",
    "generator_prediction",
    "score_prediction",
    "v_prediction",
    "velocity_prediction",
    "x0_prediction",
    # Generators
    "GeneratorOperator",
    "ItoGeneratorOperator",
    # Losses
    "action_matching_loss",
    "consistency_loss",
    "log_density_consistency_loss",
    "regression_loss",
    # Processes
    "ActionMatching",
    "ConsistencyFlowMatching",
    "Diffusion",
    "FlowMatching",
    "GeneratorMatching",
    # Fields
    "ActionHead",
    "AdaLNVelocityField",
    "DriftFromVelocityScore",
    "GeneratorField",
    "LogDensityHead",
    "MarkovizationDriftFromVelocityScore",
    "TransformerVelocityField",
    "VelocityField",
    # Schedules
    "EDMSchedule",
    "VESchedule",
    "VPSchedule",
    # Distributions, solvers, divergence
    "DPMSolverPP",
    "DiagonalNormal",
    "EulerMaruyama",
    "ExactDivergence",
    "Heun",
    "HutchinsonDivergence",
    "RK4",
    "StandardNormal",
    # Version
    "__version__",
]
