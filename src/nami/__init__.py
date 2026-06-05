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
except ModuleNotFoundError:  # pragma: no cover - build-metadata fallback
    __version__ = "0+unknown"


# __all__ is grouped by the layer each symbol belongs to, so:
# (Field -> Interpolant + Parameterization -> Loss -> Process + Solver),
__all__ = [  # noqa: RUF022  (deliberately grouped by layer, not alphabetised)
    # Fields: something that learns f(x, t, c)
    "VelocityField",  # standard field to use
    "AdaLNVelocityField",
    "TransformerVelocityField",
    "LogDensityHead",  # one-step log-density head
    # more advanced fields: generator matching / CTMC / action / score-drift composites
    "GeneratorField",
    "CTMCField",
    "ActionHead",
    "DriftFromVelocityScore",
    "MarkovizationDriftFromVelocityScore",
    # Base distributions (source / t=0) for the process
    "StandardNormal",
    "DiagonalNormal",
    "AllMask",  # discrete / masking transports
    # Interpolants: the path between source and data
    "LinearInterpolant",  # standard interpolant to use
    "CosineInterpolant",
    "StochasticLinearInterpolant",
    "GaussianInterpolant",
    "BrownianBridgeInterpolant",
    "MaskingInterpolant",  # discrete / CTMC
    # gamma schedules for stochastic interpolants
    "BrownianGamma",
    "ZeroGamma",
    "ScaledBrownianGamma",
    "GammaSchedule",
    # Parameterizations: what the field predicts
    # factories (tie a field output to a training target)
    "velocity_prediction",  # standard parameterization to use
    "epsilon_prediction",
    "score_prediction",
    "x0_prediction",
    "v_prediction",
    "generator_prediction",  # generator matching
    "action_prediction",  # action matching
    # target markers (the types the factories produce) for the parameterizations
    "Velocity",
    "Epsilon",
    "Score",
    "X0",
    "VPrediction",
    "GeneratorParams",
    "Action",
    "Parameterization",
    # Losses: pure training objectives
    "regression_loss",  # standard loss to use
    "consistency_loss",
    "stochastic_fm_loss",
    "log_density_consistency_loss",
    "cgm_loss",  # conditional generator matching
    "action_matching_loss",
    # Bregman divergences (loss geometry for generator matching)
    "BregmanDivergence",
    "KLDivergence",
    "SquaredL2",
    "ItakuraSaito",
    # Processes: sampling/likelihood(density)
    "FlowMatching",  # standard process to use
    "Diffusion",
    "ConsistencyFlowMatching",
    "GeneratorMatching",  # generator matching
    "ActionMatching",
    # Solvers: numerical schemes for the process
    "RK4",  # solver to use for ODEs
    "Heun",  # second-order solver to use for ODEs
    "EulerMaruyama",  # solver to use for SDEs
    "DPMSolverPP",  # fast solver for diffusion ODEs
    "TauLeapingSampler",  # solver to use for discrete / CTMC
    # Diffusion noise schedules
    "VPSchedule",
    "VESchedule",
    "EDMSchedule",
    # Divergence estimators (for log_prob)
    "HutchinsonDivergence",  # for high-dimensional estimator
    "ExactDivergence",  # for low-dimensional estimator
    # Generator operators (advanced: generator matching)
    "GeneratorOperator",
    "ItoGeneratorOperator",
    "CTMCGeneratorOperator",
    # Meta: version information
    "__version__",
]
