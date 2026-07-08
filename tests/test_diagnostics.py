from __future__ import annotations

import pytest
import torch

from nami import RK4, ExactDivergence, FlowMatching, StandardNormal
from nami.diagnostics import (
    describe,
    divergence_stats,
    field_stats,
    reversibility_error,
    score_projection,
)


class _LinearField:
    event_ndim = 1

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = x + t.unsqueeze(-1)
        if c is not None:
            out = out + c.sum(dim=-1, keepdim=True)
        return out


class _NoDivergenceField:
    event_ndim = 1

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = (t, c)
        return x


class _NotImplementedDivergenceField(_NoDivergenceField):
    def call_and_divergence(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (x, t, c)
        raise NotImplementedError


class _ZeroDriftField:
    event_ndim = 1

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = (t, c)
        return torch.zeros_like(x)


class _ScalarField:
    event_ndim = 0

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = c
        return x + t


class _ScalarZeroField:
    event_ndim = 0

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = (t, c)
        return torch.zeros_like(x)


class _NoOpSolver:
    requires_steps = False

    def integrate(
        self, f, x: torch.Tensor, *, t0: float, t1: float, **kwargs
    ) -> torch.Tensor:
        _ = (t1, kwargs)
        _ = f(x, t0)
        return x


class _NoOpStepSolver:
    requires_steps = True

    def __init__(self, steps: int = 0) -> None:
        self.steps = steps
        self.seen_steps: list[int] = []

    def integrate(
        self, f, x: torch.Tensor, *, t0: float, t1: float, steps: int
    ) -> torch.Tensor:
        _ = t1
        self.seen_steps.append(steps)
        _ = f(x, t0)
        return x


def test_describe_includes_shape_dtype_device_and_event_split() -> None:
    x = torch.zeros(2, 3, 4)
    c = torch.zeros(2, 3, 5)
    t = torch.zeros(2, 3)

    text = describe(x=x, c=c, t=t, event_ndim=1)

    assert "x: shape=(2, 3, 4)" in text
    assert "lead=(2, 3) event_shape=(4,)" in text
    assert "c: shape=(2, 3, 5)" in text
    assert "t: shape=(2, 3)" in text


def test_field_stats_computes_norm_summary_with_broadcasting() -> None:
    field = _LinearField()
    x = torch.tensor([[[3.0, 4.0]], [[0.0, 5.0]]])
    t = torch.tensor([[0.0, 1.0, 2.0]])
    c = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])

    stats = field_stats(field, x, t, c)

    x_b = x.expand(2, 3, 2)
    t_b = t.expand(2, 3)
    v = x_b + t_b.unsqueeze(-1) + c.sum(dim=-1, keepdim=True)
    norms = v.norm(dim=-1)

    torch.testing.assert_close(stats["mean"], norms.mean())
    torch.testing.assert_close(stats["std"], norms.std(unbiased=False))
    torch.testing.assert_close(stats["min"], norms.min())
    torch.testing.assert_close(stats["max"], norms.max())


def test_field_stats_supports_scalar_event_ndim_zero() -> None:
    field = _ScalarField()
    x = torch.tensor([[1.0], [2.0]])
    t = torch.tensor([0.5, 1.5, 2.5])

    stats = field_stats(field, x, t)
    values = x.expand(2, 3) + t.expand(2, 3)

    torch.testing.assert_close(stats["mean"], values.abs().mean())
    torch.testing.assert_close(stats["std"], values.abs().std(unbiased=False))
    torch.testing.assert_close(stats["min"], values.abs().min())
    torch.testing.assert_close(stats["max"], values.abs().max())


def test_divergence_stats_uses_estimator() -> None:
    field = _NoDivergenceField()
    x = torch.randn(2, 1, 4)
    t = torch.randn(1, 3)
    c = torch.randn(2, 3, 1)

    def estimator(
        _field, x_in: torch.Tensor, t_in: torch.Tensor, c_in: torch.Tensor | None
    ) -> torch.Tensor:
        assert _field is field
        assert x_in.shape == (2, 3, 4)
        assert t_in.shape == (2, 3)
        assert c_in is not None
        assert c_in.shape == (2, 3, 1)
        return x_in.sum(dim=-1) + t_in + c_in.squeeze(-1)

    stats = divergence_stats(field, x, t, c, estimator=estimator)
    div = estimator(field, x.expand(2, 3, 4), t.expand(2, 3), c)

    torch.testing.assert_close(stats["mean"], div.mean())
    torch.testing.assert_close(stats["std"], div.std(unbiased=False))
    torch.testing.assert_close(stats["min"], div.min())
    torch.testing.assert_close(stats["max"], div.max())


def test_divergence_stats_requires_estimator_or_call_and_divergence() -> None:
    with pytest.raises(
        TypeError, match=r"divergence_stats requires either `estimator=\.\.\.`"
    ):
        divergence_stats(_NoDivergenceField(), x=torch.randn(2, 3), t=torch.randn(2))


def test_divergence_stats_handles_not_implemented_call_and_divergence() -> None:
    with pytest.raises(
        TypeError, match=r"divergence_stats requires either `estimator=\.\.\.`"
    ):
        divergence_stats(
            _NotImplementedDivergenceField(),
            x=torch.randn(2, 3),
            t=torch.randn(2),
        )


def test_reversibility_error_is_zero_for_noop_solver_and_zero_drift() -> None:
    x = torch.randn(4, 3)
    stats = reversibility_error(_ZeroDriftField(), _NoOpSolver(), x, t0=1.0, t1=0.0)

    torch.testing.assert_close(stats["mean"], torch.tensor(0.0))
    torch.testing.assert_close(stats["std"], torch.tensor(0.0))
    torch.testing.assert_close(stats["max"], torch.tensor(0.0))


def test_reversibility_error_requires_positive_steps_when_solver_demands_them() -> None:
    with pytest.raises(ValueError, match="solver requires steps"):
        reversibility_error(
            _ZeroDriftField(), _NoOpStepSolver(steps=0), torch.randn(3, 2)
        )


def test_reversibility_error_supports_scalar_event_ndim_zero_with_steps_solver() -> (
    None
):
    solver = _NoOpStepSolver(steps=2)
    x = torch.randn(2, 3)

    stats = reversibility_error(_ScalarZeroField(), solver, x, steps=4)

    assert solver.seen_steps == [4, 4]
    torch.testing.assert_close(stats["mean"], torch.tensor(0.0))
    torch.testing.assert_close(stats["std"], torch.tensor(0.0))
    torch.testing.assert_close(stats["max"], torch.tensor(0.0))


class _ConstantContextField:
    """Velocity equal to the context: the flow is the shift x -> x + c."""

    event_ndim = 1

    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        _ = t
        if c is None:
            return torch.zeros_like(x)
        return (x * 0) + c


def test_score_projection_matches_analytic_toy() -> None:
    process = FlowMatching(
        _ConstantContextField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()
    x = torch.randn(5, 2)
    theta = torch.randn(5, 2)

    score = score_projection(
        process,
        x,
        theta,
        estimator=ExactDivergence(max_dim=8, create_graph=True),
    )

    assert score.shape == theta.shape
    assert torch.allclose(score, x - theta, atol=1e-5)
    assert not theta.requires_grad  # caller's tensor untouched
