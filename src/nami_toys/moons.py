from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import ToyDataset


@dataclass(frozen=True)
class TwoMoons:
    """Two interleaving crescents in 2-D.

    Parameters
    ----------
    noise : float
        Std-dev of isotropic Gaussian noise added to the arc positions.
    """

    noise: float = 0.1

    def generate(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw *n* labelled samples from two crescents."""
        n_upper = n // 2
        n_lower = n - n_upper

        # upper moon: arc from 0 to pi at origin
        t_up = torch.empty(n_upper).uniform_(0, math.pi, generator=generator)
        upper = torch.stack([t_up.cos(), t_up.sin()], dim=1)

        # lower moon: arc from 0 to pi, shifted right and down
        t_lo = torch.empty(n_lower).uniform_(0, math.pi, generator=generator)
        lower = torch.stack([1.0 - t_lo.cos(), 0.5 - t_lo.sin()], dim=1)

        x = torch.cat([upper, lower], dim=0)
        x = x + torch.empty_like(x).normal_(0, self.noise, generator=generator)

        y = torch.cat(
            [
                torch.zeros(n_upper, dtype=torch.long),
                torch.ones(n_lower, dtype=torch.long),
            ]
        )

        perm = torch.randperm(n, generator=generator)
        return ToyDataset(x=x[perm], y=y[perm], meta={"noise": self.noise})
