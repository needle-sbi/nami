from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import ToyDataset


@dataclass(frozen=True)
class GaussianRing:
    """Isotropic Gaussian modes arranged in a circle.

    Parameters
    ----------
    n_modes : int
        Number of equally-spaced modes.
    radius : float
        Distance of each mode centre from the origin.
    std : float
        Standard deviation of each isotropic Gaussian mode.
    """

    n_modes: int = 8
    radius: float = 3.0
    std: float = 0.3

    def generate(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw *n* samples from a ring of Gaussian modes."""
        angles = torch.linspace(0, 2 * math.pi, self.n_modes + 1)[: self.n_modes]
        centres = torch.stack(
            [self.radius * angles.cos(), self.radius * angles.sin()], dim=1
        )

        mode_idx = torch.randint(self.n_modes, (n,), generator=generator)
        x = centres[mode_idx] + torch.empty(n, 2).normal_(
            0, self.std, generator=generator
        )

        return ToyDataset(
            x=x,
            y=mode_idx,
            meta={"n_modes": self.n_modes, "radius": self.radius, "std": self.std},
        )
