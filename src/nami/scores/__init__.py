"""Score-estimator protocol and suppliers."""

from __future__ import annotations

from nami.scores.base import ScoreEstimator
from nami.scores.ctsm import CTSMJointScore
from nami.scores.dsm import DSMSpatialScore
from nami.scores.mined import MinedJointScore
from nami.scores.oracle import OracleScore

__all__ = [
    "CTSMJointScore",
    "DSMSpatialScore",
    "MinedJointScore",
    "OracleScore",
    "ScoreEstimator",
]
