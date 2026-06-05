from __future__ import annotations

import torch

from nami.core.specs import TensorSpec
from nami.generators.base import GeneratorOperator
from nami.generators.ctmc import CTMCGeneratorOperator
from nami.generators.operators import ItoGeneratorOperator
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


def test_operators_expose_consistent_tensor_spec():
    """``spec`` is the single source of shape truth for operators."""
    ito = ItoGeneratorOperator((2, 3), diffusion="diagonal")
    assert isinstance(ito.spec, TensorSpec)
    assert ito.spec.event_shape == ito.event_shape == (2, 3)
    assert ito.spec.event_ndim == ito.event_ndim == 2

    ctmc = CTMCGeneratorOperator(5, (4,))
    assert isinstance(ctmc.spec, TensorSpec)
    assert ctmc.spec.event_shape == ctmc.event_shape == (4,)
    assert ctmc.parameter_shape == (4, 5)


class _ShapedOperator(GeneratorOperator):
    """Concrete operator relying on the base-class ``spec`` derivation."""

    def __init__(self):
        super().__init__(runtime_kind="ode")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (3,)


def test_base_spec_derives_from_event_shape():
    """Without an override, ``spec`` wraps the subclass's event_shape."""
    op = _ShapedOperator()
    assert op.spec == TensorSpec((3,))


def test_base_project_is_identity():
    """The default projection leaves raw parameters untouched."""
    op = _MinimalOperator()
    params = torch.randn(4, 3)
    assert op.project(params) is params
