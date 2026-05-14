from __future__ import annotations


import torch


def broadcast_t(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Right-pad ``t`` with singleton dims so it broadcasts against ``like``."""
    return t.reshape(t.shape + (1,) * (like.ndim - t.ndim))
