from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import ToyDataset


@dataclass(frozen=True)
class TwoSpirals:
    """Two Archimedean spirals winding in opposite directions.

    Parameters
    ----------
    noise : float
        Std-dev of isotropic Gaussian noise.
    n_turns : float
        Number of full turns each spiral makes.
    """

    noise: float = 0.1
    n_turns: float = 1.5

    def generate(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw *n* labelled samples from two spirals."""
        n_a = n // 2
        n_b = n - n_a
        max_angle = self.n_turns * 2 * math.pi

        def _arm(count: int, offset: float) -> torch.Tensor:
            # sqrt spacing gives uniform density along the arm
            t = torch.empty(count).uniform_(0, 1, generator=generator).sqrt()
            theta = t * max_angle + offset
            r = t * max_angle / (2 * math.pi)  # radius grows with arc length
            pts = torch.stack([r * theta.cos(), r * theta.sin()], dim=1)
            return pts + torch.empty_like(pts).normal_(
                0, self.noise, generator=generator
            )

        x = torch.cat([_arm(n_a, 0.0), _arm(n_b, math.pi)], dim=0)
        y = torch.cat(
            [
                torch.zeros(n_a, dtype=torch.long),
                torch.ones(n_b, dtype=torch.long),
            ]
        )

        perm = torch.randperm(n, generator=generator)
        return ToyDataset(
            x=x[perm],
            y=y[perm],
            meta={"noise": self.noise, "n_turns": self.n_turns},
        )
