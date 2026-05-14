r"""Training-time losses on the unified Interpolant + Parameterization vocabulary.

Aggregates the regression-style objective
(:func:`~nami.losses.regression.regression_loss`, specialising to Flow
Matching, Rectified Flow, score / ``\epsilon``-prediction, EDM, and stochastic
interpolants) together with the structurally distinct trajectory-pair
and gradient-regression losses: consistency
(:func:`~nami.losses.consistency.consistency_loss`; Song et al.,
*Consistency Models*, 2023), log-density consistency
(:func:`~nami.losses.log_density.log_density_consistency_loss`; Chen et
al. / Grathwohl et al. instantaneous change-of-variables), action
matching (:func:`~nami.losses.action.action_matching_loss`; Neklyudov
et al., 2023), and Schrödinger-bridge matching
(:func:`~nami.losses.bridge.bridge_matching_loss`; Tong et al., 2024).
"""

from __future__ import annotations

from nami.losses.action import action_matching_loss
from nami.losses.bridge import bridge_matching_loss
from nami.losses.consistency import consistency_loss
from nami.losses.log_density import log_density_consistency_loss
from nami.losses.regression import regression_loss
from nami.losses.stochastic_fm import stochastic_fm_loss

__all__ = [
    "action_matching_loss",
    "bridge_matching_loss",
    "consistency_loss",
    "log_density_consistency_loss",
    "regression_loss",
    "stochastic_fm_loss",
]
