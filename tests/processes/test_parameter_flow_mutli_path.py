r"""path-pinned multi-theta parameter flow.
"""

from __future__ import annotations

import math

import pytest
import torch

from nami import RK4
from nami.core.specs import TensorSpec
from nami.divergence import ExactDivergence
from nami.fields.scalar_potential import ScalarPotentialField
from nami.losses.parameter_flow import path_pinned_parameter_flow_loss
from nami.paths import LinearParameterPath
from nami.processes.parameter_flow import ParameterFlow
from nami.scores import CTSMJointScore, OracleScore

# Gaussian-mean family p_theta = N(theta, I) on R^2.
THETA_0 = torch.tensor([0.0, 0.0])
THETA_1 = torch.tensor([1.0, 1.0])


def _diagonal_path() -> LinearParameterPath:
    return LinearParameterPath(THETA_0, THETA_1)


def _joint_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # d/dtheta log N(x; theta, I) = x - theta, shape (*lead, d_theta).
    return x - theta


def _spatial_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # d/dx log N(x; theta, I) = theta - x, shape (*lead, d_x).
    return theta - x


class _AnalyticPerUnitSPotential:
    r"""phi(x, theta) = (theta_1 - theta_0) . x — the per-unit-s potential."""

    spec = TensorSpec((2,))
    event_ndim = 1
    event_shape = (2,)

    def __init__(self, delta: torch.Tensor):
        self._delta = delta

    def __call__(self, x, t=None, c=None):
        del t, c
        return (x * self._delta).sum(dim=-1)

    def velocity(self, x, theta, *, create_graph=True):
        del theta, create_graph
        return self._delta.expand_as(x).clone()

    def velocity_field(self):
        delta = self._delta

        class _Grad:
            spec = TensorSpec((2,))
            event_ndim = 1
            event_shape = (2,)

            def __call__(self, x, t=None, c=None):
                del t, c
                return x * 0 + delta

        return _Grad()

def test_pinned_loss_zero_on_analytic_potential():
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)

    s = torch.rand(128)
    theta_s = path.theta(s)
    x = theta_s + torch.randn(128, 2)

    loss = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=joint,
        spatial_score=spatial,
    )
    assert loss.item() < 1e-10


def test_pinned_loss_accepts_ctsm_directional_score():
    # CTSMJointScore(directional=True) returns the already-contracted
    # d/ds log p; with directional_score=True the pinned loss consumes it
    # directly (no second tangent contraction) and the analytic potential's
    # exact cancellation still drives the residual to zero.
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    spatial = OracleScore(_spatial_score)

    def time_net(x, s):
        # d/ds log p = theta-dot . (x - theta(s)) for N(theta(s), I).
        return (delta * (x - path.theta(s))).sum(-1, keepdim=True)

    directional = CTSMJointScore(time_net, path, directional=True)

    s = torch.rand(128)
    x = path.theta(s) + torch.randn(128, 2)

    loss = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=directional,
        spatial_score=spatial,
        directional_score=True,
    )
    assert loss.item() < 1e-10


def test_pinned_transport_is_exact_shift():
    # The analytic per-unit-s potential transports theta_0 -> theta_1 as
    # the exact constant shift x -> x + (theta_1 - theta_0), with NO
    # theta-dot multiplication (pinned mode).
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    process = ParameterFlow(field, RK4(steps=8))(path)
    assert process.pinned is True

    x0 = torch.randn(64, 2)
    x1 = process.transport(x0)

    torch.testing.assert_close(x1, x0 + delta)


def test_pinned_transport_with_logp_is_exact():
    # Zero Laplacian => log-density carried unchanged; endpoint matches
    # N(theta_1, I) evaluated at the shifted points.
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    process = ParameterFlow(field, RK4(steps=8))(path)

    x0 = torch.randn(32, 2)
    logp0 = -0.5 * (x0**2).sum(-1) - math.log(2 * math.pi)
    x1, logp1 = process.transport_with_logp(
        x0, logp0, divergence_estimator=ExactDivergence(max_dim=4)
    )

    expected = -0.5 * ((x1 - THETA_1) ** 2).sum(-1) - math.log(2 * math.pi)
    torch.testing.assert_close(x1, x0 + delta)
    torch.testing.assert_close(logp1, expected)


def test_pinned_score_supply_returns_directional_scalar():
    # In pinned mode score_supply returns the directional (along-path)
    # score d/ds log p, a scalar per sample, shape (*lead, 1).  For the
    # analytic potential: -Lap - grad phi . spatial = 0 - delta.(theta-x)
    # = delta.(x - theta) = theta-dot . joint_score.
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    process = ParameterFlow(field, RK4(steps=2))(path)
    spatial = OracleScore(_spatial_score)

    s = torch.rand(16)
    theta_s = path.theta(s)
    x = theta_s + torch.randn(16, 2)

    supplied = process.score_supply(x, s=s, spatial_score=spatial)
    expected = (delta * _joint_score(x, theta_s)).sum(-1, keepdim=True)

    assert supplied.shape == (16, 1)
    torch.testing.assert_close(supplied, expected)


def test_pinned_score_supply_requires_s():
    path = _diagonal_path()
    field = _AnalyticPerUnitSPotential(THETA_1 - THETA_0)
    process = ParameterFlow(field, RK4(steps=2))(path)
    spatial = OracleScore(_spatial_score)
    with pytest.raises(ValueError, match="requires the path parameter s"):
        process.score_supply(torch.randn(4, 2), spatial_score=spatial)


def test_pinned_loss_rejects_bad_s_shape():
    path = _diagonal_path()
    field = _AnalyticPerUnitSPotential(THETA_1 - THETA_0)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)
    x = torch.randn(8, 2)
    with pytest.raises(ValueError, match="leading shape of x"):
        path_pinned_parameter_flow_loss(
            field,
            x=x,
            s=torch.rand(8, 1),  # wrong: should be (8,)
            path=path,
            joint_score=joint,
            spatial_score=spatial,
        )


def test_pinned_loss_rejects_non_flat_event():
    class _Event2:
        event_ndim = 2

    with pytest.raises(ValueError, match="event_ndim"):
        path_pinned_parameter_flow_loss(
            _Event2(),
            x=torch.randn(4, 1),
            s=torch.rand(4),
            path=_diagonal_path(),
            joint_score=OracleScore(_joint_score),
            spatial_score=OracleScore(_spatial_score),
        )


def test_pinned_loss_rejects_theta_dtheta_shape_mismatch():
    class _MismatchedPath:
        def theta(self, s):
            return s.unsqueeze(-1).expand(*s.shape, 2)

        def dtheta_ds(self, s):
            return s.unsqueeze(-1)  # (*lead, 1) != theta's (*lead, 2)

    with pytest.raises(ValueError, match="must share a shape"):
        path_pinned_parameter_flow_loss(
            _AnalyticPerUnitSPotential(THETA_1 - THETA_0),
            x=torch.randn(4, 2),
            s=torch.rand(4),
            path=_MismatchedPath(),
            joint_score=OracleScore(_joint_score),
            spatial_score=OracleScore(_spatial_score),
        )


def test_pinned_loss_rejects_spatial_score_shape_mismatch():
    with pytest.raises(ValueError, match="spatial_score returned shape"):
        path_pinned_parameter_flow_loss(
            _AnalyticPerUnitSPotential(THETA_1 - THETA_0),
            x=torch.randn(4, 2),
            s=torch.rand(4),
            path=_diagonal_path(),
            joint_score=OracleScore(_joint_score),
            spatial_score=OracleScore(lambda x, _theta: x[..., :1]),
        )


def test_pinned_loss_accepts_flat_directional_score():
    # directional_score=True with a score returning the flat (*lead,) shape
    # is used directly with no tangent contraction.
    path = _diagonal_path()
    delta = THETA_1 - THETA_0
    field = _AnalyticPerUnitSPotential(delta)
    s = torch.rand(64)
    x = path.theta(s) + torch.randn(64, 2)

    def flat_directional(x, theta):
        return (delta * (x - theta)).sum(-1)  # shape (*lead,)

    loss = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=flat_directional,
        spatial_score=OracleScore(_spatial_score),
        directional_score=True,
    )
    assert loss.item() < 1e-10


def test_pinned_loss_rejects_bad_directional_score_shape():
    with pytest.raises(ValueError, match="directional joint_score must return"):
        path_pinned_parameter_flow_loss(
            _AnalyticPerUnitSPotential(THETA_1 - THETA_0),
            x=torch.randn(4, 2),
            s=torch.rand(4),
            path=_diagonal_path(),
            joint_score=lambda x, _theta: torch.zeros(*x.shape[:-1], 3),
            spatial_score=OracleScore(_spatial_score),
            directional_score=True,
        )


def test_pinned_loss_rejects_joint_score_shape_mismatch():
    with pytest.raises(ValueError, match="joint_score returned shape"):
        path_pinned_parameter_flow_loss(
            _AnalyticPerUnitSPotential(THETA_1 - THETA_0),
            x=torch.randn(4, 2),
            s=torch.rand(4),
            path=_diagonal_path(),
            joint_score=lambda x, _theta: x[..., :1],
            spatial_score=OracleScore(_spatial_score),
        )


def test_pinned_loss_reduction_modes():
    path = _diagonal_path()
    field = ScalarPotentialField(2, theta_dim=2, hidden=16, layers=2)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)
    s = torch.rand(10)
    x = path.theta(s) + torch.randn(10, 2)

    none = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=joint,
        spatial_score=spatial,
        reduction="none",
    )
    assert none.shape == (10,)
    mean = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=joint,
        spatial_score=spatial,
        reduction="mean",
    )
    torch.testing.assert_close(mean, none.mean())


def test_pinned_loss_backward_reaches_field():
    path = _diagonal_path()
    field = ScalarPotentialField(2, theta_dim=2, hidden=16, layers=2)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)
    s = torch.rand(10)
    x = path.theta(s) + torch.randn(10, 2)
    loss = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=joint,
        spatial_score=spatial,
    )
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)

def _energy_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Energy distance between two samples (pure torch).

    E = 2 E||A - B|| - E||A - A'|| - E||B - B'||, all pairwise Euclidean.
    Zero iff the distributions match; non-negative.
    """

    def _mean_pdist(u, v):
        d = torch.cdist(u, v)
        return d.mean()

    return 2 * _mean_pdist(a, b) - _mean_pdist(a, a) - _mean_pdist(b, b)


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One-dimensional two-sample Kolmogorov-Smirnov statistic (pure torch)."""
    a_sorted = torch.sort(a)[0]
    b_sorted = torch.sort(b)[0]
    grid = torch.cat([a_sorted, b_sorted])
    cdf_a = torch.searchsorted(a_sorted, grid, right=True).float() / a.numel()
    cdf_b = torch.searchsorted(b_sorted, grid, right=True).float() / b.numel()
    return (cdf_a - cdf_b).abs().max()


@pytest.mark.slow
def test_gaussian_mean_path_pinned():
    """Path-pinned multi-theta training on the Gaussian-mean family.

    Train ScalarPotentialField(2, theta_dim=2) with the path-pinned loss
    and oracle scores on p_theta = N(theta, I) along a fixed diagonal
    path; the recovered grad(phi) must match the analytic per-unit-s
    velocity (the constant theta-dot) in p-weighted L2 < 1e-3 at held-out
    s, and transported samples from N(theta_0, I) must match N(theta_1, I)
    by mean / variance / energy-distance / per-marginal KS checks.
    """
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)

    path = _diagonal_path()
    delta = THETA_1 - THETA_0  # the analytic per-unit-s velocity (constant)
    field = ScalarPotentialField(2, theta_dim=2, hidden=64, layers=3)
    joint = OracleScore(_joint_score)
    spatial = OracleScore(_spatial_score)

    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    for step in range(3000):
        s = torch.rand(512, generator=generator)
        theta_s = path.theta(s)
        x = theta_s + torch.randn(512, 2, generator=generator)
        loss = path_pinned_parameter_flow_loss(
            field,
            x=x,
            s=s,
            path=path,
            joint_score=joint,
            spatial_score=spatial,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.3e}")

    # Criterion (a): p-weighted L2 velocity error at held-out s.
    for s_eval in (0.25, 0.5, 0.75):
        s_t = torch.full((4096,), s_eval)
        theta_s = path.theta(s_t)
        x = theta_s + torch.randn(4096, 2, generator=generator)
        v_model = field.velocity(x, theta_s, create_graph=False)
        v_true = delta.expand_as(v_model)
        err = (v_model - v_true).pow(2).sum(-1).mean()
        print(f"s={s_eval}: p-weighted L2 velocity err {err.item():.3e}")
        assert err.item() < 1e-3

    # Criterion (b): transported samples match the truth at theta_1.
    process = ParameterFlow(field, RK4(steps=64))(path)
    n = 8192
    x0 = THETA_0 + torch.randn(n, 2, generator=generator)
    x1 = process.transport(x0)
    truth = THETA_1 + torch.randn(n, 2, generator=generator)

    mean_err = (x1.mean(0) - THETA_1).abs()
    se = x1.std(0) / (n**0.5)
    print(f"transported mean {x1.mean(0).tolist()}, |err| {mean_err.tolist()}")
    assert torch.all(mean_err < 3 * se)

    var_err = (x1.var(0) - 1.0).abs()
    print(f"transported var {x1.var(0).tolist()}, |err| {var_err.tolist()}")
    assert torch.all(var_err < 0.1)

    # Energy distance against fresh truth samples (sub-sample for cdist).
    ed = _energy_distance(x1[:2048], truth[:2048])
    ed_ref = _energy_distance(
        truth[:2048], (THETA_1 + torch.randn(2048, 2, generator=generator))
    )
    print(
        f"energy distance transported-vs-truth {ed.item():.3e}, "
        f"truth-vs-truth ref {ed_ref.item():.3e}"
    )
    assert ed.item() < 5 * ed_ref.item().__abs__() + 1e-2

    # Per-marginal KS check.
    for d in range(2):
        ks = _ks_statistic(x1[:, d], truth[:, d])
        print(f"dim {d}: KS statistic {ks.item():.3e}")
        assert ks.item() < 0.05


VAR_0 = torch.tensor([1.0, 1.0])
VAR_1 = torch.tensor([2.25, 0.49])  # dim 0 inflates, dim 1 shrinks


def _variance_path() -> LinearParameterPath:
    return LinearParameterPath(VAR_0, VAR_1)


def _var_joint_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return -1.0 / (2 * theta) + x.pow(2) / (2 * theta.pow(2))


def _var_spatial_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return -x / theta


def _var_velocity(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return (VAR_1 - VAR_0) * x / (2 * theta)


def _var_laplacian(theta: torch.Tensor) -> torch.Tensor:
    return ((VAR_1 - VAR_0) / (2 * theta)).sum(-1)


class _AnalyticCurvedPotential:
    r"""phi(x, theta) = sum_i Delta_i x_i^2 / (4 theta_i) — curved."""

    spec = TensorSpec((2,))
    event_ndim = 1
    event_shape = (2,)

    def __call__(self, x, t=None, c=None):
        del t
        return ((VAR_1 - VAR_0) * x.pow(2) / (4 * c)).sum(dim=-1)

    def velocity(self, x, theta, *, create_graph=True):
        del create_graph
        return _var_velocity(x, theta)

    def velocity_field(self):
        class _Grad:
            spec = TensorSpec((2,))
            event_ndim = 1
            event_shape = (2,)

            def __call__(self, x, t=None, c=None):
                del t
                return _var_velocity(x, c)

        return _Grad()


def test_pinned_loss_zero_on_curved_analytic_potential():
    # The residual must vanish through genuine cancellation of three
    # NON-zero terms (directional joint score, Laplacian, coupling).
    path = _variance_path()
    field = _AnalyticCurvedPotential()
    s = torch.rand(256)
    theta_s = path.theta(s)
    x = theta_s.sqrt() * torch.randn(256, 2)

    loss = path_pinned_parameter_flow_loss(
        field,
        x=x,
        s=s,
        path=path,
        joint_score=OracleScore(_var_joint_score),
        spatial_score=OracleScore(_var_spatial_score),
    )
    # Sanity that the cancellation is non-trivial: the Laplacian alone is
    # far from zero.
    assert _var_laplacian(theta_s).abs().mean() > 0.01
    assert loss.item() < 1e-10


def test_analytic_transport_is_exact_anisotropic_scaling():
    # The flow of v_i = Delta_i x_i / (2 theta_i(s)) integrates exactly to
    # x_i -> x_i sqrt(theta_i(1)/theta_i(0)).
    path = _variance_path()
    field = _AnalyticCurvedPotential()
    process = ParameterFlow(field, RK4(steps=64))(path)
    assert process.pinned is True

    x0 = torch.randn(128, 2)
    x1 = process.transport(x0)

    torch.testing.assert_close(x1, x0 * (VAR_1 / VAR_0).sqrt(), atol=1e-5, rtol=1e-5)


@pytest.mark.slow
def test_anisotropic_variance_path_pinned():
    """Path-pinned training on a curved potential exercised non-trivially.

    Train ScalarPotentialField on the anisotropic variance family; the
    recovered velocity must match the x-DEPENDENT analytic velocity, the
    recovered Laplacian must match the NON-zero analytic Laplacian, and
    transport must reproduce the exact anisotropic scaling map both
    sample-wise and distributionally.
    """
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)

    path = _variance_path()
    field = ScalarPotentialField(2, theta_dim=2, hidden=64, layers=3)
    joint = OracleScore(_var_joint_score)
    spatial = OracleScore(_var_spatial_score)

    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
    for step in range(4000):
        s = torch.rand(512, generator=generator)
        theta_s = path.theta(s)
        x = theta_s.sqrt() * torch.randn(512, 2, generator=generator)
        loss = path_pinned_parameter_flow_loss(
            field,
            x=x,
            s=s,
            path=path,
            joint_score=joint,
            spatial_score=spatial,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.3e}")

    estimator = ExactDivergence(max_dim=4)
    grad_field = field.velocity_field()
    for s_eval in (0.25, 0.5, 0.75):
        s_t = torch.full((4096,), s_eval)
        theta_s = path.theta(s_t)
        x = theta_s.sqrt() * torch.randn(4096, 2, generator=generator)

        # (a) x-dependent velocity recovered.
        v_model = field.velocity(x, theta_s, create_graph=False)
        v_true = _var_velocity(x, theta_s)
        err = (v_model - v_true).pow(2).sum(-1).mean()
        print(f"s={s_eval}: p-weighted L2 velocity err {err.item():.3e}")
        assert err.item() < 1e-3

        # (a') the elliptic term is genuinely exercised: trained Laplacian
        # matches the non-zero analytic one.
        t_dummy = torch.zeros((), dtype=x.dtype)
        lap_model = estimator(grad_field, x[:512], t_dummy, theta_s[:512])
        lap_true = _var_laplacian(theta_s[:512])
        lap_err = (lap_model - lap_true).abs().mean()
        print(
            f"s={s_eval}: Laplacian err {lap_err.item():.3e} "
            f"(analytic terms {((VAR_1 - VAR_0) / (2 * path.theta(torch.tensor([s_eval])))).squeeze().tolist()})"
        )
        assert lap_err.item() < 0.05

    # (b) transport: exact anisotropic scaling, sample-wise and in law.
    process = ParameterFlow(field, RK4(steps=64))(path)
    n = 8192
    x0 = VAR_0.sqrt() * torch.randn(n, 2, generator=generator)
    x1 = process.transport(x0)
    exact = x0 * (VAR_1 / VAR_0).sqrt()

    samplewise_rms = (x1 - exact).pow(2).sum(-1).mean().sqrt()
    print(f"sample-wise RMS vs exact scaling map: {samplewise_rms.item():.3e}")
    assert samplewise_rms.item() < 0.05

    std_err = (x1.std(0) - VAR_1.sqrt()).abs()
    print(f"transported std {x1.std(0).tolist()} (target {VAR_1.sqrt().tolist()})")
    assert torch.all(std_err < 0.05)

    mean_err = x1.mean(0).abs()
    se = x1.std(0) / (n**0.5)
    assert torch.all(mean_err < 3 * se)

    truth = VAR_1.sqrt() * torch.randn(n, 2, generator=generator)
    for d in range(2):
        ks = _ks_statistic(x1[:, d], truth[:, d])
        print(f"dim {d}: KS statistic {ks.item():.3e}")
        assert ks.item() < 0.05
