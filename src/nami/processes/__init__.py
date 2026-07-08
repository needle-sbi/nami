"""Concrete transport processes built on the unified vocabulary.

Each pair is ``(LazyProcess, ConcreteProcess)``; the lazy variant is
the constructor users build, and ``forward(c)`` produces the concrete
runtime object exposing ``sample`` / ``rsample`` / ``log_prob``.

Process subpackage entry points are also re-exported at top-level
``nami`` for the user API.
"""

from __future__ import annotations

from nami.processes.action import ActionMatching, ActionMatchingProcess
from nami.processes.consistency import (
    ConsistencyFlowMatching,
    ConsistencyFlowMatchingProcess,
)
from nami.processes.diffusion import Diffusion, DiffusionProcess
from nami.processes.fm import FlowMatching, FlowMatchingProcess
from nami.processes.gm import GeneratorMatching, GeneratorMatchingProcess
from nami.processes.parameter_flow import ParameterFlow, ParameterFlowProcess

__all__ = [
    "ActionMatching",
    "ActionMatchingProcess",
    "ConsistencyFlowMatching",
    "ConsistencyFlowMatchingProcess",
    "Diffusion",
    "DiffusionProcess",
    "FlowMatching",
    "FlowMatchingProcess",
    "GeneratorMatching",
    "GeneratorMatchingProcess",
    "ParameterFlow",
    "ParameterFlowProcess",
]
