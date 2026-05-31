"""Public surface
TODO: write nice summary docstring for this file
* **Interpolant + Parameterization -> Loss**
* **Field + Parameterization -> Process**
* **Process -> sample / log_prob**

"""

from __future__ import annotations

from nami import diffusion as diffusion
from nami.distributions.mask import AllMask
from nami.distributions.normal import DiagonalNormal, StandardNormal
from nami.divergence.exact import ExactDivergence
from nami.divergence.hutchinson import HutchinsonDivergence
from nami.fields.action import ActionHead
from nami.fields.adaln import AdaLNVelocityField
from nami.fields.composite import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
)
from nami.fields.consistency import LogDensityHead
from nami.fields.ctmc import CTMCField
from nami.fields.generator import GeneratorField
from nami.fields.transformer_velocity import TransformerVelocityField
from nami.fields.velocity import VelocityField
from nami.generators.base import GeneratorOperator
from nami.generators.ctmc import CTMCGeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
from nami.generators.parameterizations import generator_prediction
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
from nami.interpolants.masking import MaskingInterpolant
from nami.losses.action import action_matching_loss, action_prediction
from nami.losses.bregman import (
    BregmanDivergence,
    ItakuraSaito,
    KLDivergence,
    SquaredL2,
)
from nami.losses.cgm import cgm_loss
from nami.losses.consistency import consistency_loss
from nami.losses.log_density import log_density_consistency_loss
from nami.losses.regression import regression_loss
from nami.losses.stochastic_fm import stochastic_fm_loss
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
from nami.processes.action import ActionMatching
from nami.processes.consistency import ConsistencyFlowMatching
from nami.processes.diffusion import Diffusion
from nami.processes.fm import FlowMatching
from nami.processes.gm import GeneratorMatching
from nami.schedules.edm import EDMSchedule
from nami.schedules.ve import VESchedule
from nami.schedules.vp import VPSchedule
from nami.solvers.dpm import DPMSolverPP
from nami.solvers.heun import Heun
from nami.solvers.jump import TauLeapingSampler
from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama

try:
    from nami._version import version as __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"


__all__ = [
    "RK4",
    "X0",
    "Action",
    "ActionHead",
    "ActionMatching",
    "AdaLNVelocityField",
    "AllMask",
    "BregmanDivergence",
    "BrownianBridgeInterpolant",
    "BrownianGamma",
    "CTMCField",
    "CTMCGeneratorOperator",
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
    "GammaSchedule",
    "GaussianInterpolant",
    "GeneratorField",
    "GeneratorMatching",
    "GeneratorOperator",
    "GeneratorParams",
    "Heun",
    "HutchinsonDivergence",
    "ItakuraSaito",
    "ItoGeneratorOperator",
    "KLDivergence",
    "LinearInterpolant",
    "LogDensityHead",
    "MarkovizationDriftFromVelocityScore",
    "MaskingInterpolant",
    "Parameterization",
    "ScaledBrownianGamma",
    "Score",
    "SquaredL2",
    "StandardNormal",
    "StochasticLinearInterpolant",
    "TauLeapingSampler",
    "TransformerVelocityField",
    "VESchedule",
    "VPSchedule",
    "VPrediction",
    "Velocity",
    "VelocityField",
    "ZeroGamma",
    "__version__",
    "action_matching_loss",
    "action_prediction",
    "cgm_loss",
    "consistency_loss",
    "epsilon_prediction",
    "generator_prediction",
    "log_density_consistency_loss",
    "regression_loss",
    "score_prediction",
    "stochastic_fm_loss",
    "v_prediction",
    "velocity_prediction",
    "x0_prediction",
]
