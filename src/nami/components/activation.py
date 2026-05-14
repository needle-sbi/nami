"""Activation-function registry keyed by lowercase name."""

from __future__ import annotations



from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "mish": nn.Mish,
    "hard_swish": nn.Hardswish,
    "hard_sigmoid": nn.Hardsigmoid,
}


def get_activation(name: str) -> nn.Module:
    """Return a freshly constructed activation module by name."""
    activation = _ACTIVATIONS.get(name)
    if activation is None:
        msg = f"Unknown activation: {name!r}. Available: {sorted(_ACTIVATIONS)}"
        raise ValueError(msg)
    return activation()


__all__ = ["get_activation"]
