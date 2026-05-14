"""Transformer blocks with optional cross-attention.

Standard pre-norm encoder block (self-attention + MLP) with an
optional cross-attention sub-layer for context tokens. Followed by a
:class:`TransformerBackbone` stacking ``depth`` blocks.

References
----------
- Vaswani et al., *Attention Is All You Need*, 2017
  (arXiv:1706.03762).
"""

from __future__ import annotations


import torch
from torch import nn

from nami.components.activation import get_activation


# -------------------------------------------------------------------------------------
#
def _flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        msg = "expected at least 2 dimensions: (..., seq, dim)"
        raise ValueError(msg)
    lead_shape = tuple(x.shape[:-2])
    seq_len, width = x.shape[-2:]
    return x.reshape(-1, seq_len, width), lead_shape


def _restore_tokens(x: torch.Tensor, lead_shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*lead_shape, *x.shape[-2:])


class TransformerBlock(nn.Module):
    """Transformer block for token sequences, with optional cross-attention."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        cross_attention: bool = False,
    ):
        super().__init__()
        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        if num_heads <= 0:
            msg = f"num_heads must be positive, got {num_heads}"
            raise ValueError(msg)

        if dim % num_heads != 0:
            msg = f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}"
            raise ValueError(msg)

        if mlp_ratio < 1.0:
            msg = f"mlp_ratio must be >= 1, got {mlp_ratio}"
            raise ValueError(msg)

        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0, 1), got {dropout}"
            raise ValueError(msg)

        hidden_dim = int(dim * mlp_ratio)
        self.dim = int(dim)
        self.cross_attention = bool(cross_attention)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if self.cross_attention:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attend ``x`` (and cross-attend ``context`` if provided)."""
        tokens, lead_shape = _flatten_tokens(x)

        if context is not None and not self.cross_attention:
            msg = "context was provided but cross_attention is disabled"
            raise ValueError(msg)

        y = self.norm_self(tokens)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        tokens = tokens + y

        if context is not None:
            context_tokens, _ = _flatten_tokens(context)
            y = self.norm_cross(tokens)
            y, _ = self.cross_attn(
                y, context_tokens, context_tokens, need_weights=False
            )
            tokens = tokens + y

        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return _restore_tokens(tokens, lead_shape)


class TransformerBackbone(nn.Module):
    """A stack of transformer blocks operating on sequences/sets."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        cross_attention: bool = False,
    ):
        super().__init__()
        if depth <= 0:
            msg = f"depth must be positive, got {depth}"
            raise ValueError(msg)
        self._dim = int(dim)
        self._cross_attention = bool(cross_attention)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation=activation,
                    cross_attention=cross_attention,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def _validate_context_tokens(
        self,
        context: torch.Tensor | None,
        lead_shape: tuple[int, ...],
    ) -> None:
        """Validate projected context token shape before passing through blocks."""
        if context is not None:
            if not self._cross_attention:
                msg = "context was provided but cross_attention is disabled"
                raise ValueError(msg)
            if context.shape[:-2] != lead_shape:
                msg = (
                    f"context shape mismatch: expected leading shape {lead_shape}, "
                    f"got {tuple(context.shape[:-2])}"
                )
                raise ValueError(msg)
            if context.shape[-1] != self._dim:
                msg = (
                    f"context width mismatch: expected {self._dim}, "
                    f"got {context.shape[-1]}"
                )
                raise ValueError(msg)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ``depth`` transformer blocks then a final layer norm."""
        lead_shape = tuple(x.shape[:-2])
        self._validate_context_tokens(context, lead_shape)
        for block in self.blocks:
            x = block(x, context=context)
        return self.norm(x)
