"""Toy 2-D datasets and a lightweight ``ToyDataset`` container.

Classic generative-model targets — checkerboards, two-moons, two-spirals,
Gaussian ring / shell / mixture — used in notebooks and unit tests as
fast, visualisable benchmarks for transport methods.
"""

from __future__ import annotations

from .checkerboard import Checkerboard
from .dataset import ToyDataset
from .gaussian import GaussianMixture
from .moons import TwoMoons
from .parameterised import ParameterisedGaussian
from .ring import GaussianRing
from .rng import make_generator
from .shell import GaussianShell
from .spirals import TwoSpirals
from .standardise import Standardiser

__all__ = [
    "Checkerboard",
    "GaussianMixture",
    "GaussianRing",
    "GaussianShell",
    "ParameterisedGaussian",
    "Standardiser",
    "ToyDataset",
    "TwoMoons",
    "TwoSpirals",
    "make_generator",
]
