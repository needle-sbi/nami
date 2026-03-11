from __future__ import annotations

import pytest
import torch

from nami.components import (
    MLPBackbone,
    ScalarTimeEmbedding,
    SinusoidalTimeEmbedding,
    TransformerBackbone,
    get_activation,
)


def test_mlp_backbone_supports_multiple_leading_dims():
    backbone = MLPBackbone(5, 3, hidden=16, layers=2, layer_norm=True)
    x = torch.randn(2, 4, 6, 5)

    out = backbone(x)

    assert out.shape == (2, 4, 6, 3)


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
