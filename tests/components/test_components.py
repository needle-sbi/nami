from __future__ import annotations

import pytest
import torch

from nami.components import (
    MLPBackbone,
    ScalarTimeEmbedding,
    SinusoidalTimeEmbedding,
    TransformerBackbone,
    TransformerBlock,
    get_activation,
)

# ---------------------------------------------------------
# activation


@pytest.mark.parametrize(
    "name",
    [
        "relu",
        "silu",
        "gelu",
        "tanh",
        "elu",
        "leaky_relu",
        "selu",
        "swish",
        "mish",
        "hard_swish",
        "hard_sigmoid",
    ],
)
def test_get_activation_supports_extended_registry(name: str):
    activation = get_activation(name)
    x = torch.randn(2, 3, 4)

    out = activation(x)

    assert out.shape == x.shape


def test_get_activation_unknown_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("nonexistent")

# ---------------------------------------------------------
# MLP


def test_mlp_backbone_supports_multiple_leading_dims():
    backbone = MLPBackbone(5, 3, hidden=16, layers=2, layer_norm=True)
    x = torch.randn(2, 4, 6, 5)

    out = backbone(x)

    assert out.shape == (2, 4, 6, 3)


def test_mlp_backbone_with_dropout():
    backbone = MLPBackbone(4, 2, hidden=8, layers=2, dropout=0.1)
    x = torch.randn(3, 4)

    out = backbone(x)

    assert out.shape == (3, 2)


def test_mlp_backbone_zero_layers():
    backbone = MLPBackbone(4, 2, hidden=8, layers=0)
    x = torch.randn(3, 4)

    out = backbone(x)

    assert out.shape == (3, 2)


def test_validate_mlp_config_hidden_non_positive():
    with pytest.raises(ValueError, match="hidden must be positive"):
        MLPBackbone(4, 2, hidden=0)


def test_validate_mlp_config_layers_negative():
    with pytest.raises(ValueError, match="layers must be non-negative"):
        MLPBackbone(4, 2, layers=-1)


def test_validate_mlp_config_dropout_out_of_range():
    with pytest.raises(ValueError, match="dropout must be in"):
        MLPBackbone(4, 2, dropout=1.0)


# ---------------------------------------------------------
# time emb.


def test_scalar_and_sinusoidal_time_embeddings_match_leading_shape():
    leading_shape = (2, 3, 4)
    t = torch.rand(2, 3, 4)

    scalar = ScalarTimeEmbedding()(t, leading_shape=leading_shape, device=t.device, dtype=t.dtype)
    sinusoidal = SinusoidalTimeEmbedding(7)(
        t,
        leading_shape=leading_shape,
        device=t.device,
        dtype=t.dtype,
    )

    assert scalar.shape == (2, 3, 4, 1)
    assert sinusoidal.shape == (2, 3, 4, 7)
    assert torch.equal(sinusoidal[..., -1], t)


def test_sinusoidal_dim_1_returns_raw_scalar():
    emb = SinusoidalTimeEmbedding(1)
    t = torch.tensor([0.5])

    out = emb(t, leading_shape=(1,), device=t.device, dtype=t.dtype)

    assert out.shape == (1, 1)
    assert torch.allclose(out.squeeze(), t)


def test_sinusoidal_even_dim():
    emb = SinusoidalTimeEmbedding(8)
    t = torch.rand(2, 3)

    out = emb(t, leading_shape=(2, 3), device=t.device, dtype=t.dtype)

    assert out.shape == (2, 3, 8)


def test_sinusoidal_dim_non_positive_raises():
    with pytest.raises(ValueError, match="dim must be positive"):
        SinusoidalTimeEmbedding(0)


def test_sinusoidal_max_period_non_positive_raises():
    with pytest.raises(ValueError, match="max_period must be positive"):
        SinusoidalTimeEmbedding(8, max_period=0.0)


def test_sinusoidal_out_dim_property():
    emb = SinusoidalTimeEmbedding(16)
    assert emb.out_dim == 16


# ---------------------------------------------------------
# transformer


def test_transformer_backbone_preserves_shape_with_context_tokens():
    backbone = TransformerBackbone(
        dim=8,
        depth=2,
        num_heads=2,
        cross_attention=True,
    )
    x = torch.randn(3, 5, 8)
    context = torch.randn(3, 2, 8)

    out = backbone(x, context=context)

    assert out.shape == x.shape


def test_transformer_block_self_attention_only():
    block = TransformerBlock(dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)

    out = block(x)

    assert out.shape == x.shape


def test_transformer_block_context_without_cross_attention_raises():
    block = TransformerBlock(dim=8, num_heads=2, cross_attention=False)
    x = torch.randn(2, 4, 8)
    ctx = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="cross_attention is disabled"):
        block(x, context=ctx)


def test_transformer_block_dim_non_positive_raises():
    with pytest.raises(ValueError, match="dim must be positive"):
        TransformerBlock(dim=0, num_heads=1)


def test_transformer_block_num_heads_non_positive_raises():
    with pytest.raises(ValueError, match="num_heads must be positive"):
        TransformerBlock(dim=8, num_heads=0)


def test_transformer_block_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="dim must be divisible by num_heads"):
        TransformerBlock(dim=7, num_heads=2)


def test_transformer_block_mlp_ratio_too_low_raises():
    with pytest.raises(ValueError, match="mlp_ratio must be >= 1"):
        TransformerBlock(dim=8, num_heads=2, mlp_ratio=0.5)


def test_transformer_block_dropout_out_of_range_raises():
    with pytest.raises(ValueError, match="dropout must be in"):
        TransformerBlock(dim=8, num_heads=2, dropout=1.0)


def test_transformer_block_1d_input_raises():
    block = TransformerBlock(dim=8, num_heads=2)
    x = torch.randn(8)

    with pytest.raises(ValueError, match="expected at least 2 dimensions"):
        block(x)


def test_transformer_backbone_depth_non_positive_raises():
    with pytest.raises(ValueError, match="depth must be positive"):
        TransformerBackbone(dim=8, depth=0, num_heads=2)


def test_transformer_backbone_context_shape_mismatch_raises():
    backbone = TransformerBackbone(dim=8, depth=1, num_heads=2, cross_attention=True)
    x = torch.randn(3, 5, 8)
    ctx = torch.randn(4, 2, 8)  # wrong batch dim

    with pytest.raises(ValueError, match="context shape mismatch"):
        backbone(x, context=ctx)


def test_transformer_backbone_context_width_mismatch_raises():
    backbone = TransformerBackbone(dim=8, depth=1, num_heads=2, cross_attention=True)
    x = torch.randn(3, 5, 8)
    ctx = torch.randn(3, 2, 4)  # wrong width

    with pytest.raises(ValueError, match="context width mismatch"):
        backbone(x, context=ctx)


def test_transformer_backbone_context_without_cross_attention_raises():
    backbone = TransformerBackbone(dim=8, depth=1, num_heads=2, cross_attention=False)
    x = torch.randn(3, 5, 8)
    ctx = torch.randn(3, 2, 8)

    with pytest.raises(ValueError, match="cross_attention is disabled"):
        backbone(x, context=ctx)


def test_transformer_block_extra_leading_dims():
    block = TransformerBlock(dim=8, num_heads=2)
    x = torch.randn(2, 3, 4, 8)

    out = block(x)

    assert out.shape == x.shape
