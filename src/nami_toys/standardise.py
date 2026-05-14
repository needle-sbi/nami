from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import ToyDataset


@dataclass(frozen=True)
class Standardiser:
    """Shift-and-scale transform fitted from data.

    Parameters
    ----------
    mean : torch.Tensor
        Per-feature mean, shape ``(d,)``.
    std : torch.Tensor
        Per-feature standard deviation, shape ``(d,)``.
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, *, eps: float = 1e-8) -> Standardiser:
        """Compute mean and std from a data tensor ``(N, d)``.

        Features with zero variance are given ``std = 1`` to avoid division
        by zero.
        """
        return cls(mean=x.mean(dim=0), std=x.std(dim=0).clamp(min=eps))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise *x* to zero mean and unit variance."""
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map standardised *x* back to the original scale."""
        return x * self.std + self.mean

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`transform`."""
        return self.transform(x)

    def transform_dataset(self, ds: ToyDataset) -> ToyDataset:
        """Return a copy of *ds* with standardised features."""
        return ToyDataset(x=self.transform(ds.x), y=ds.y, meta={**ds.meta})
