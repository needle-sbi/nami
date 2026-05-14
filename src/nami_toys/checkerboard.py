from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import ToyDataset


@dataclass(frozen=True)
class Checkerboard:
    """Uniform density on the "on" cells of a 2-D checkerboard.

    Parameters
    ----------
    cells : int
        Number of cells per side (total grid is *cells* x *cells*).
    bound : float
        The grid spans [-bound, bound] along each axis.
    """

    cells: int = 4
    bound: float = 2.0

    def generate(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw *n* samples uniformly from the filled squares."""
        on_cells = [
            (i, j)
            for i in range(self.cells)
            for j in range(self.cells)
            if (i + j) % 2 == 0
        ]
        centres = torch.tensor(on_cells, dtype=torch.float)  # (K, 2)

        idx = torch.randint(len(on_cells), (n,), generator=generator)
        offsets = torch.empty(n, 2).uniform_(0, 1, generator=generator)

        cell_size = 2.0 * self.bound / self.cells
        x = -self.bound + (centres[idx] + offsets) * cell_size

        return ToyDataset(
            x=x,
            meta={"cells": self.cells, "bound": self.bound},
        )
