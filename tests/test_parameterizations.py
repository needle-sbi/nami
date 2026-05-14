"""Stage-0 invariants for the parameterization vocabulary.

These tests do not exercise any concrete model — they pin the *shape* of
the new vocabulary so that later stages cannot silently drift back into a
parallel hierarchy.  Two invariants are protected here:

1. ``Target`` enumerates exactly the variants the design committed to.
   Adding a new variant must be a deliberate choice that updates the
   union, this test, and every consumer's pattern match.
2. Pattern-matching dispatch over ``Target`` works structurally.  The
   sentinel test below mirrors how concrete interpolants will dispatch
   in Stage 1 onward; if it stops compiling cleanly, the protocol has
   regressed.
"""
from __future__ import annotations

from typing import get_args

import pytest
import torch

from nami.interpolants.protocol import InterpolantState
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Target,
    Velocity,
    VPrediction,
)


def test_target_union_is_exactly_the_committed_variants() -> None:
    expected = {Velocity, Score, Epsilon, X0, VPrediction, Action, GeneratorParams}
    assert set(get_args(Target)) == expected


def test_parameterization_defaults_are_neutral() -> None:
    p = Parameterization(target=Velocity())
    t = torch.linspace(0.0, 1.0, 4)
    assert torch.equal(p.weighting(t), torch.ones_like(t))
    y = torch.randn(3)
    assert p.output_transform(y) is y


def test_pattern_match_dispatch_over_target_variants() -> None:
    """Mirror the dispatch shape concrete Interpolants will use in Stage 1.

    Each variant must be reachable by a structural pattern; missing arms
    fall through to the explicit failure.  This is the test that breaks
    when someone adds a new variant without updating consumers.
    """

    def name(t: Target) -> str:
        match t:
            case Velocity():
                return "velocity"
            case Score():
                return "score"
            case Epsilon():
                return "epsilon"
            case X0():
                return "x0"
            case VPrediction():
                return "v"
            case Action():
                return "action"
            case GeneratorParams(operator=_op):
                return "generator_params"
            case _:
                pytest.fail("non-exhaustive Target match")

    class _Op:
        pass

    assert name(Velocity()) == "velocity"
    assert name(Score()) == "score"
    assert name(Epsilon()) == "epsilon"
    assert name(X0()) == "x0"
    assert name(VPrediction()) == "v"
    assert name(Action()) == "action"
    assert name(GeneratorParams(operator=_Op())) == "generator_params"  # type: ignore[arg-type]


def test_interpolant_state_carries_endpoints_and_noise() -> None:
    xt = torch.zeros(2, 3)
    state = InterpolantState(
        xt=xt,
        x_data=torch.ones(2, 3),
        x_noise=torch.full((2, 3), -1.0),
        t=torch.tensor([0.25, 0.75]),
        noise=None,
    )
    assert state.xt.shape == (2, 3)
    assert state.noise is None
