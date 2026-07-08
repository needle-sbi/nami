r"""End-to-end no-oracle loop: trained score estimators in place of oracles."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nami.components import SinusoidalTimeEmbedding
from nami.fields.scalar_potential import ScalarPotentialField
from nami.losses.parameter_flow import parameter_flow_loss
from nami.losses.score_matching import (
    denoising_score_matching_loss,
    time_score_matching_loss,
)
from nami.paths import LinearParameterPath
from nami.scores import CTSMJointScore, DSMSpatialScore, OracleScore

# theta(s) = -0.4 -> +0.4, within the family's validity range;
# CTSMJointScore is path-locked to this segment.
THETA_LO, THETA_HI = -0.4, 0.4
DELTA = THETA_HI - THETA_LO


def _p0(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _p_theta(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return _p0(x) * (1 + theta * torch.tanh(x))


def _joint_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) / (1 + theta * torch.tanh(x))


def _spatial_score(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    sech2 = 1.0 / torch.cosh(x) ** 2
    return -x + theta * sech2 / (1 + theta * torch.tanh(x))


def _v_analytic_on_grid(grid: torch.Tensor, theta: float) -> torch.Tensor:
    integrand = _p0(grid) * torch.tanh(grid)
    integral = torch.cat(
        [torch.zeros(1, dtype=grid.dtype), torch.cumulative_trapezoid(integrand, grid)]
    )
    return -integral / _p_theta(grid, torch.tensor(theta, dtype=grid.dtype))


def _sample_component(sign: float, n: int, generator: torch.Generator) -> torch.Tensor:
    """Rejection-sample q_pm = 2 p0(x) sigma(pm 2x) from the N(0,1) proposal."""
    out = []
    have = 0
    while have < n:
        prop = torch.randn(2 * n + 64, generator=generator)
        u = torch.rand(prop.shape, generator=generator)
        acc = prop[u < torch.sigmoid(2 * sign * prop)]
        out.append(acc)
        have += acc.numel()
    return torch.cat(out)[:n]


def _simulate(theta: torch.Tensor, generator: torch.Generator):
    """Draw (x, z) from the mixture; returns x (N,1) and z_pm (N,1) in {-1,+1}."""
    lam = (1 + theta) / 2
    z = (torch.rand(theta.shape, generator=generator) < lam).float() * 2 - 1
    x = torch.empty_like(theta)
    for sign in (1.0, -1.0):
        mask = z.squeeze(-1) == sign
        n = int(mask.sum())
        if n:
            x[mask, 0] = _sample_component(sign, n, generator)
    return x, z


def _latent_joint_score(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Latent score: d/dtheta log p_theta(z) = z / (1 + theta z)."""
    return z / (1 + theta * z)


def test_simulator_matches_marginal():
    # The mixture sampler must reproduce the analytic marginal (KS check),
    # and the latent score's posterior mean must equal the marginal joint
    # score (binned check).
    generator = torch.Generator().manual_seed(0)
    theta = torch.full((20000, 1), 0.3)
    x, z = _simulate(theta, generator)

    # KS against the analytic CDF on a grid.
    grid = torch.linspace(-8, 8, 4001, dtype=torch.float64)
    pdf = _p_theta(grid, torch.tensor(0.3, dtype=torch.float64))
    cdf = torch.cat([torch.zeros(1), torch.cumulative_trapezoid(pdf, grid)])
    cdf = cdf / cdf[-1]
    xs = torch.sort(x.squeeze(-1))[0].double()
    emp = torch.arange(1, xs.numel() + 1, dtype=torch.float64) / xs.numel()
    idx = torch.searchsorted(grid, xs).clamp(0, grid.numel() - 1)
    ks = (emp - cdf[idx]).abs().max()
    assert ks.item() < 0.02

    # Binned conditional mean of the latent score ~= marginal joint score.
    targets = _latent_joint_score(z, theta).squeeze(-1)
    edges = torch.linspace(-2.5, 2.5, 26)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bins = torch.bucketize(x.squeeze(-1), edges).clamp(1, 25) - 1
    err = []
    for b in range(25):
        sel = bins == b
        if sel.sum() > 200:
            binned = targets[sel].mean()
            truth = _joint_score(centres[b], torch.tensor(0.3))
            err.append((binned - truth).abs())
    assert torch.stack(err).mean() < 0.05


@pytest.mark.slow
def test_no_oracle_loop_with_trained_scores():
    """Samples in, transport out: CTSM + DSM in place of oracle scores.

    Stage metrics are printed so the error propagation (score error ->
    velocity error) is inspectable.
    """
    torch.manual_seed(23)
    generator = torch.Generator().manual_seed(23)
    path = LinearParameterPath(torch.tensor([THETA_LO]), torch.tensor([THETA_HI]))

    def draw_theta(n: int) -> torch.Tensor:
        s = torch.rand(n, generator=generator)
        return path.theta(s), s

    sigma_dsm = 0.15
    dsm_net = nn.Sequential(
        nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)
    )

    def dsm_wrapped(x, theta):
        return dsm_net(torch.cat([x, theta], dim=-1))

    opt = torch.optim.Adam(dsm_net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-5)
    for _ in range(4000):
        theta, _ = draw_theta(1024)
        x, _ = _simulate(theta, generator)
        loss = denoising_score_matching_loss(
            dsm_wrapped, x=x, theta=theta, sigma=sigma_dsm
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    spatial_trained = DSMSpatialScore(dsm_wrapped)

    t_emb = SinusoidalTimeEmbedding(16, max_period=100.0)
    ts_net = nn.Sequential(
        nn.Linear(17, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)
    )

    def ts_wrapped(x, s):
        feats = t_emb(s, leading_shape=x.shape[:-1], device=x.device, dtype=x.dtype)
        return ts_net(torch.cat([x, feats], dim=-1))

    opt = torch.optim.Adam(ts_net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000, eta_min=1e-5)
    for _ in range(6000):
        s = torch.rand(2048, generator=generator)
        theta = path.theta(s)
        x, z = _simulate(theta, generator)
        # d/ds log p = theta-dot * (latent joint score).
        target = DELTA * _latent_joint_score(z, theta).squeeze(-1)
        loss = time_score_matching_loss(ts_wrapped, x=x, s=s, target=target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    joint_trained = CTSMJointScore(ts_wrapped, path)

    for theta_eval in (-0.3, 0.0, 0.3):
        n = 8192
        theta = torch.full((n, 1), theta_eval)
        x, _ = _simulate(theta, generator)
        with torch.no_grad():
            dsm_rms = (
                (spatial_trained(x, theta) - _spatial_score(x, theta))
                .pow(2)
                .mean()
                .sqrt()
            )
            ctsm_rms = (
                (joint_trained(x, theta) - _joint_score(x, theta)).pow(2).mean().sqrt()
            )
        print(
            f"theta={theta_eval}: DSM spatial RMS {dsm_rms.item():.3e}, "
            f"CTSM joint RMS {ctsm_rms.item():.3e}"
        )
        assert dsm_rms.item() < 0.15
        assert ctsm_rms.item() < 0.10

    def train_phi(joint, spatial, seed: int) -> ScalarPotentialField:
        torch.manual_seed(seed)
        field = ScalarPotentialField(1, theta_dim=1, hidden=64, layers=3)
        optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
        for _ in range(3000):
            theta, _ = draw_theta(512)
            x, _ = _simulate(theta, generator)
            loss = parameter_flow_loss(
                field, x=x, theta=theta, joint_score=joint, spatial_score=spatial
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return field

    field_no_oracle = train_phi(joint_trained, spatial_trained, seed=7)
    field_oracle = train_phi(
        OracleScore(_joint_score), OracleScore(_spatial_score), seed=7
    )

    grid = torch.linspace(-5.0, 5.0, 801, dtype=torch.float64)
    for theta_eval in (-0.3, 0.0, 0.3):
        v_true = _v_analytic_on_grid(grid, theta_eval)
        xs = grid.float().unsqueeze(-1)
        thetas = torch.full((grid.numel(), 1), theta_eval)
        weights = _p_theta(grid, torch.tensor(theta_eval, dtype=grid.dtype))
        weights = weights / torch.trapezoid(weights, grid)

        def werr(a, b, w=weights):
            return torch.trapezoid((a - b) ** 2 * w, grid).item()

        v_no = field_no_oracle.velocity(xs, thetas, create_graph=False)
        v_no = v_no.squeeze(-1).double()
        v_or = field_oracle.velocity(xs, thetas, create_graph=False)
        v_or = v_or.squeeze(-1).double()

        e_or = werr(v_or, v_true)
        e_no = werr(v_no, v_true)
        e_gap = werr(v_no, v_or)
        print(
            f"theta={theta_eval}: oracle {e_or:.3e}, NO-ORACLE {e_no:.3e}, "
            f"gap {e_gap:.3e}"
        )
        assert e_or < 1e-3
        assert e_no < 1e-2
        assert e_gap < 1e-2
