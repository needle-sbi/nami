"""Reusable neural building blocks: MLPs, time embeddings, transformer blocks.

These are framework primitives shared across fields — no transport
semantics live here.
"""

from __future__ import annotations

from nami.components.activation import get_activation
from nami.components.mlp import MLPBackbone
from nami.components.time import ScalarTimeEmbedding, SinusoidalTimeEmbedding
from nami.components.transformer import TransformerBackbone, TransformerBlock

__all__ = [
    "MLPBackbone",
    "ScalarTimeEmbedding",
    "SinusoidalTimeEmbedding",
    "TransformerBackbone",
    "TransformerBlock",
    "get_activation",
]
