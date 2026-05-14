from __future__ import annotations

import torch


def make_generator(seed: int | None = None) -> torch.Generator | None:
    """Create a seeded :class:`torch.Generator`, or return ``None`` if no seed."""
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)
