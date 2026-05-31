from __future__ import annotations

import torch

from nami.generators.base import GeneratorOperator
from nami.losses.bregman import SquaredL2


class _MinimalOperator(GeneratorOperator):
    """Smallest concrete operator: exercises only the inherited base methods."""

    def __init__(self):
        super().__init__(runtime_kind="ode")


def test_base_decompose_is_single_all_component():
    """The base operator treats the whole tensor as one ``"all"`` component."""
    op = _MinimalOperator()
    params = torch.randn(4, 3)
    parts = op.decompose(params)
    assert set(parts) == {"all"}
    assert parts["all"] is params


def test_base_default_divergence_is_squared_l2():
    """A Euclidean operator defaults to squared-L2 (MSE)."""
    op = _MinimalOperator()
    assert isinstance(op.default_divergence(), SquaredL2)
