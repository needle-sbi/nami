from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ToyDataset:
    """Lightweight container for toy simulation data.

    Parameters
    ----------
    x : torch.Tensor
        Data points with shape ``(N, d)``.
    y : torch.Tensor | None
        Optional integer labels with shape ``(N,)``.
    meta : dict[str, Any]
        Metadata from the generation call.
    """

    x: torch.Tensor
    y: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __repr__(self) -> str:
        n, d = self.x.shape
        labels = (
            f", labels={set(self.y.unique().tolist())}" if self.y is not None else ""
        )
        meta = f", meta={self.meta}" if self.meta else ""
        return f"ToyDataset(n={n}, d={d}{labels}{meta})"

    def subset(self, mask: torch.Tensor) -> ToyDataset:
        """Return a new dataset containing only entries where *mask* is True."""
        y_sub = self.y[mask] if self.y is not None else None
        return ToyDataset(x=self.x[mask], y=y_sub, meta={**self.meta})

    def limit(self, n: int) -> ToyDataset:
        """Return a dataset containing at most *n* entries."""
        n = min(n, len(self))
        y_lim = self.y[:n] if self.y is not None else None
        return ToyDataset(x=self.x[:n], y=y_lim, meta={**self.meta})
