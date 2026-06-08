from __future__ import annotations

import math

import pytest
import torch

from nami import RK4
from nami.core.specs import TensorSpec
from nami.divergence import ExactDivergence
from nami.fields.scalar_potential import ScalarPotentialField
from nami.losses.parameter_flow import parameter_flow_loss
from nami.paths import LinearParameterPath
from nami.processes.parameter_flow import ParameterFlow, ParameterFlowProcess
from nami.scores import OracleScore


class _AnalyticPotential:
    spec = TensorSpec((1,))
    event_ndim = 1
    event_shape = (1,)

    def __call__(self, x, t=None, c=None):
        del t, c
        return x.sum(dim=-1)

    def velocity(self, x, theta, *, create_graph=True):
        del theta, create_graph
        return torch.ones_like(x)

    def velocity_field(self):
        return _AnalyticGradient()


class _AnalyticGradient:
    spec = TensorSpec((1,))
    event_ndim = 1
    event_shape = (1,)

    def __call__(self, x, t=None, c=None):
        del t, c
        # x * 0 keeps the output attached to x's graph for divergence
        # estimators.
        return x * 0 + 1.0


def _unit_path() -> LinearParameterPath:
    return LinearParameterPath(torch.tensor([0.0]), torch.tensor([1.0]))


def test_lazy_forward_binds_path():
    lazy = ParameterFlow(_AnalyticPotential(), RK4(steps=4))
    process = lazy(_unit_path())
    assert isinstance(process, ParameterFlowProcess)
    assert process.path is not None


def test_forward_requires_path():
    lazy = ParameterFlow(_AnalyticPotential(), RK4(steps=4))
    with pytest.raises(ValueError, match="requires a ParameterPath"):
        lazy()


def test_event_shape_property():
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=4))(_unit_path())
    assert process.event_shape == (1,)


def test_transport_shifts_gaussian_mean():
    # With the analytic potential, transport along theta: 0 -> 1 is the
    # exact shift x -> x + 1 (velocity theta-dot * grad(phi) = 1).
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=8))(_unit_path())
    x0 = torch.randn(64, 1)

    x1 = process.transport(x0)

    torch.testing.assert_close(x1, x0 + 1.0)


def test_transport_with_logp_is_exact_on_gaussian_mean():
    # The shift map has zero divergence contribution only if Laplacian
    # of phi is 0 — true here, so log p is carried unchanged and equals
    # the analytic N(1, 1) log-density at the endpoint.
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=8))(_unit_path())
    x0 = torch.randn(32, 1)
    logp0 = -0.5 * (x0.squeeze(-1) ** 2) - 0.5 * math.log(2 * math.pi)

    x1, logp1 = process.transport_with_logp(
        x0, logp0, divergence_estimator=ExactDivergence(max_dim=4)
    )

    expected = -0.5 * ((x1.squeeze(-1) - 1.0) ** 2) - 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(x1, x0 + 1.0)
    torch.testing.assert_close(logp1, expected)


def test_multi_theta_selects_pinned_mode():
    # A path with d_theta > 1 puts the process in path-pinned mode: no
    # theta-dot scaling, phi is the per-unit-s potential.
    path = LinearParameterPath(torch.zeros(2), torch.ones(2))
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=2))(path)
    assert process.pinned is True


def test_dim1_path_is_not_pinned():
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=2))(_unit_path())
    assert process.pinned is False


def test_score_supply_matches_oracle_on_analytic_potential():
    # score_supply inverts the PDE: -Lap(phi) - grad(phi) . spatial =
    # 0 - 1 * (theta - x) = x - theta, the exact joint score of
    # p_theta = N(theta, 1).
    process = ParameterFlow(_AnalyticPotential(), RK4(steps=2))(_unit_path())
    theta = torch.randn(16, 1)
    x = theta + torch.randn(16, 1)
    spatial = OracleScore(lambda x, theta: theta - x)

    supplied = process.score_supply(x, theta, spatial_score=spatial)

    torch.testing.assert_close(supplied, x - theta)


def _p0(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _p_theta(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return _p0(x) * (1 + theta * torch.tanh(x))


def _joint_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # Elementwise on broadcastable (x, theta); for (N, 1) inputs the
    # output is (N, 1) = (*lead, d_theta) with d_theta = 1.
    return torch.tanh(x) / (1 + theta * torch.tanh(x))


def _spatial_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    sech2 = 1.0 / torch.cosh(x) ** 2
    return -x + theta * sech2 / (1 + theta * torch.tanh(x))


def _grid(n: int = 4001, lo: float = -8.0, hi: float = 8.0) -> torch.Tensor:
    return torch.linspace(lo, hi, n, dtype=torch.float64)


def _v_analytic_on_grid(
    grid: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    integrand = _p0(grid) * torch.tanh(grid)
    integral = torch.cat(
        [
            torch.zeros(1, dtype=grid.dtype),
            torch.cumulative_trapezoid(integrand, grid),
        ]
    )
    p = _p_theta(grid, torch.tensor(theta, dtype=grid.dtype))
    return grid, -integral / p


def _sample_p_theta(theta: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Inverse-CDF sampling of p_theta on a grid, batched over theta."""
    grid = _grid()
    pdf = _p_theta(grid, theta.double().unsqueeze(-1))  # (n, grid)
    cdf = torch.cat(
        [
            torch.zeros(pdf.shape[0], 1, dtype=pdf.dtype),
            torch.cumulative_trapezoid(pdf, grid, dim=-1),
        ],
        dim=-1,
    )
    cdf = cdf / cdf[:, -1:]
    u = torch.rand(theta.shape[0], 1, dtype=cdf.dtype, generator=generator)
    idx = torch.searchsorted(cdf, u).clamp(1, grid.numel() - 1)
    c0 = cdf.gather(-1, idx - 1)
    c1 = cdf.gather(-1, idx)
    frac = (u - c0) / (c1 - c0).clamp_min(1e-12)
    x = grid[idx - 1] + frac * (grid[idx] - grid[idx - 1])
    return x.float()


@pytest.mark.slow
def test_linear_tilt_recovers_analytic_velocity():
    # Train phi on the PDE residual with oracle scores; the recovered
    # velocity grad(phi) must match the continuity-equation solution in
    # p_theta-weighted squared error, and score_supply must agree with
    # the joint-score oracle on data.
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(7)

    field = ScalarPotentialField(1, theta_dim=1, hidden=64, layers=3)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)

    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    for step in range(3000):
        theta = torch.rand(512, 1, generator=generator) - 0.5  # U(-0.5, 0.5)
        x = _sample_p_theta(theta.squeeze(-1), generator)
        loss = parameter_flow_loss(
            field,
            x=x,
            theta=theta,
            joint_score=joint,
            spatial_score=spatial,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.3e}")

    # Evaluate p_theta-weighted squared velocity error on a grid.
    for theta_eval in (-0.3, 0.0, 0.4):
        grid, v_true = _v_analytic_on_grid(_grid(801, -5.0, 5.0), theta_eval)
        xs = grid.float().unsqueeze(-1)
        thetas = torch.full((grid.numel(), 1), theta_eval)
        v_model = field.velocity(xs, thetas, create_graph=False).squeeze(-1).double()
        weights = _p_theta(grid, torch.tensor(theta_eval, dtype=grid.dtype))
        weights = weights / torch.trapezoid(weights, grid)
        err = torch.trapezoid((v_model - v_true) ** 2 * weights, grid)
        print(f"theta={theta_eval}: weighted L2 err {err.item():.3e}")
        assert err.item() < 1e-3

    # The PDE inversion must agree with the joint-score oracle on data.
    process = ParameterFlow(field, RK4(steps=32))(
        LinearParameterPath(torch.tensor([0.0]), torch.tensor([0.4]))
    )
    theta = torch.zeros(2048, 1) + 0.2
    x = _sample_p_theta(theta.squeeze(-1), generator)
    supplied = process.score_supply(x, theta, spatial_score=spatial)
    target = _joint_score(x, theta)
    rms = (supplied - target).pow(2).mean().sqrt()
    print(f"score_supply RMS error {rms.item():.3e}")
    assert rms.item() < 0.05
