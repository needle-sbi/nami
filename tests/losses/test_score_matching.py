r"""Tests for the upstream score-matching training losses."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nami.components import SinusoidalTimeEmbedding
from nami.losses.score_matching import (
    _conditional_time_score,
    _schedule_rates,
    ctsm_loss,
    denoising_score_matching_loss,
    time_score_matching_loss,
)
from nami.schedules.vp import VPSchedule
from nami.scores import DSMSpatialScore


class _ThetaConditionedNet(nn.Module):
    """Tiny net (x, theta) -> grad_x log p, conditioned on theta."""

    def __init__(self, d_x: int, d_theta: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_theta, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_x),
        )

    def forward(self, x, theta):
        return self.net(torch.cat([x, theta], dim=-1))

def test_dsm_loss_reductions():
    net = _ThetaConditionedNet(2, 2)
    x = torch.randn(8, 2)
    theta = torch.randn(8, 2)
    noise = torch.randn(8, 2)
    none = denoising_score_matching_loss(
        net, x=x, theta=theta, sigma=0.5, noise=noise, reduction="none"
    )
    assert none.shape == (8,)
    mean = denoising_score_matching_loss(
        net, x=x, theta=theta, sigma=0.5, noise=noise, reduction="mean"
    )
    torch.testing.assert_close(mean, none.mean())
    s = denoising_score_matching_loss(
        net, x=x, theta=theta, sigma=0.5, noise=noise, reduction="sum"
    )
    torch.testing.assert_close(s, none.sum())


def test_dsm_loss_backward():
    net = _ThetaConditionedNet(2, 2)
    x = torch.randn(8, 2)
    theta = torch.randn(8, 2)
    loss = denoising_score_matching_loss(net, x=x, theta=theta, sigma=0.3)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_dsm_loss_target_matches_analytic():
    # The DSM target is exactly (x - x_tilde)/sigma^2 = -noise/sigma; a net
    # returning that target gives zero loss.
    sigma = 0.4
    noise = torch.randn(16, 2)

    class _Exact:
        def __init__(self, x):
            self._x = x

        def __call__(self, x_tilde, _theta):
            return (self._x - x_tilde) / sigma**2

    x = torch.randn(16, 2)
    theta = torch.randn(16, 2)
    loss = denoising_score_matching_loss(
        _Exact(x), x=x, theta=theta, sigma=sigma, noise=noise
    )
    assert loss.item() < 1e-12


def test_dsm_loss_rejects_shape_mismatch():
    def bad(x_tilde, _theta):
        return x_tilde[..., :1]

    with pytest.raises(ValueError, match="expected x's shape"):
        denoising_score_matching_loss(
            bad, x=torch.randn(4, 2), theta=torch.randn(4, 2), sigma=0.5
        )

def test_time_score_loss_zero_on_exact_net():
    target = torch.randn(10)

    class _Exact:
        def __call__(self, _x, _s):
            return target.unsqueeze(-1)

    x = torch.randn(10, 1)
    s = torch.rand(10)
    loss = time_score_matching_loss(_Exact(), x=x, s=s, target=target)
    assert loss.item() < 1e-12


def test_time_score_loss_squeezes_trailing_one_target():
    target = torch.randn(10, 1)

    class _Exact:
        def __call__(self, _x, _s):
            return target

    loss = time_score_matching_loss(
        _Exact(), x=torch.randn(10, 1), s=torch.rand(10), target=target
    )
    assert loss.item() < 1e-12


def test_time_score_loss_rejects_shape_mismatch():
    def net(x, _s):
        return x.expand(*x.shape[:-1], 2)

    with pytest.raises(ValueError, match="does not match"):
        time_score_matching_loss(
            net, x=torch.randn(6, 1), s=torch.rand(6), target=torch.randn(6)
        )


def test_time_score_loss_reductions_and_backward():
    net = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))

    def wrapped(x, s):
        return net(torch.cat([x, s.unsqueeze(-1)], dim=-1))

    x = torch.randn(12, 1)
    s = torch.rand(12)
    target = torch.randn(12)
    none = time_score_matching_loss(wrapped, x=x, s=s, target=target, reduction="none")
    assert none.shape == (12,)
    mean = time_score_matching_loss(wrapped, x=x, s=s, target=target)
    torch.testing.assert_close(mean, none.mean())
    mean.backward()
    assert any(p.grad is not None for p in net.parameters())


@pytest.mark.slow
def test_dsm_trained_recovers_gaussian_spatial_score():
    """A DSM net trained on N(theta, I) recovers the smoothed spatial score.

    The DSM minimiser is the score of the sigma-smoothed density: for
    N(theta, I) corrupted by N(0, sigma^2 I) that is N(theta, (1+sigma^2)I),
    whose score is (theta - x) / (1 + sigma^2).  We use a small sigma so
    this is close to the clean score theta - x, and check both.
    """
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(3)
    sigma = 0.25

    net = _ThetaConditionedNet(2, 2, hidden=128)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(4000):
        theta = torch.randn(1024, 2, generator=generator)
        x = theta + torch.randn(1024, 2, generator=generator)
        loss = denoising_score_matching_loss(net, x=x, theta=theta, sigma=sigma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"step {step}: dsm loss {loss.item():.3e}")

    dsm = DSMSpatialScore(net)
    theta = torch.randn(4096, 2, generator=generator)
    x = theta + torch.randn(4096, 2, generator=generator)
    with torch.no_grad():
        pred = dsm(x, theta)
    smoothed = (theta - x) / (1 + sigma**2)  # DSM minimiser
    clean = theta - x  # analytic clean score
    rms_smoothed = (pred - smoothed).pow(2).mean().sqrt()
    rms_clean = (pred - clean).pow(2).mean().sqrt()
    print(
        f"DSM RMS vs smoothed score {rms_smoothed.item():.3e}, "
        f"vs clean score {rms_clean.item():.3e}"
    )
    assert rms_smoothed.item() < 0.1
    assert rms_clean.item() < 0.15


def _wrap_time_net(net):
    def wrapped(x, t):
        return net(torch.cat([x, t.unsqueeze(-1)], dim=-1))

    return wrapped


def test_ctsm_conditional_target_matches_autograd():
    # The closed-form conditional time score must equal the autograd
    # t-derivative of log N(x; alpha_t z, sigma_t^2 I) at fixed x.
    torch.manual_seed(0)
    schedule = VPSchedule(0.1, 8.0)
    t = torch.rand(16, dtype=torch.float64) * 0.8 + 0.1
    z = torch.randn(16, 3, dtype=torch.float64)
    eps = torch.randn(16, 3, dtype=torch.float64)

    a, s, a_dot, s_dot = _schedule_rates(schedule, t)
    x_fixed = (a.unsqueeze(-1) * z + s.unsqueeze(-1) * eps).detach()
    closed_form = _conditional_time_score(z, eps, a_dot, s, s_dot)

    t_var = t.detach().clone().requires_grad_(True)
    aa = schedule.alpha(t_var).unsqueeze(-1)
    ss = schedule.sigma(t_var).unsqueeze(-1)
    logp = (
        -0.5 * 3 * math.log(2 * math.pi)
        - 3 * torch.log(ss.squeeze(-1))
        - (x_fixed - aa * z).pow(2).sum(-1) / (2 * ss.squeeze(-1) ** 2)
    )
    (autograd_score,) = torch.autograd.grad(logp.sum(), t_var)

    torch.testing.assert_close(closed_form, autograd_score, atol=1e-9, rtol=1e-9)


def test_ctsm_loss_shapes_reductions_backward():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
    wrapped = _wrap_time_net(net)
    schedule = VPSchedule(0.1, 8.0)
    x_data = torch.randn(12, 1)

    none = ctsm_loss(wrapped, x_data=x_data, schedule=schedule, reduction="none")
    assert none.shape == (12,)
    assert torch.isfinite(none).all()

    mean = ctsm_loss(
        wrapped,
        x_data=x_data,
        t=torch.full((12,), 0.5),
        noise=torch.randn(12, 1),
        schedule=schedule,
    )
    mean.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_ctsm_rejects_net_shape_mismatch():
    schedule = VPSchedule(0.1, 8.0)
    x_data = torch.randn(6, 1)

    def bad(x, _t):
        return x.expand(*x.shape[:-1], 2)

    with pytest.raises(ValueError, match="does not match"):
        ctsm_loss(bad, x_data=x_data, t=torch.full((6,), 0.4), schedule=schedule)


def test_ctsm_callable_weighting_broadcasts_scalar():
    net = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    wrapped = _wrap_time_net(net)
    schedule = VPSchedule(0.1, 8.0)
    kwargs = {
        "x_data": torch.randn(8, 1),
        "t": torch.full((8,), 0.4),
        "noise": torch.randn(8, 1),
        "schedule": schedule,
    }

    scalar_w = ctsm_loss(wrapped, weighting=lambda _t: torch.tensor(0.5), **kwargs)
    uniform = ctsm_loss(wrapped, weighting="uniform", **kwargs)
    assert torch.isfinite(scalar_w)
    torch.testing.assert_close(scalar_w, 0.5 * uniform)


def test_ctsm_weighting_modes():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
    wrapped = _wrap_time_net(net)
    schedule = VPSchedule(0.1, 8.0)
    x_data = torch.randn(8, 1)
    t = torch.full((8,), 0.4)
    noise = torch.randn(8, 1)
    kwargs = {"x_data": x_data, "t": t, "noise": noise, "schedule": schedule}

    normalised = ctsm_loss(wrapped, weighting="normalised", **kwargs)
    uniform = ctsm_loss(wrapped, weighting="uniform", **kwargs)
    assert torch.isfinite(normalised)
    assert torch.isfinite(uniform)
    assert normalised.item() != uniform.item()

    zero = ctsm_loss(wrapped, weighting=torch.zeros_like, **kwargs)
    assert zero.item() == 0.0

    with pytest.raises(ValueError, match="weighting"):
        ctsm_loss(wrapped, weighting="banana", **kwargs)


def _marginal_logp(x, t, schedule, mu1, sig1):
    """Closed-form marginal log-density of the VP path over N(mu1, sig1^2)."""
    a = schedule.alpha(t)
    s = schedule.sigma(t)
    m = a * mu1
    v = a**2 * sig1**2 + s**2
    return -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(v) - (x - m) ** 2 / (2 * v)


def _marginal_time_score(x, t, schedule, mu1, sig1):
    t = t.detach().clone().requires_grad_(True)
    logp = _marginal_logp(x, t, schedule, mu1, sig1)
    (g,) = torch.autograd.grad(logp.sum(), t)
    return g


@pytest.mark.slow
def test_ctsm_recovers_marginal_time_score_and_density_ratio():
    """Train on conditional targets, recover the marginal time score.

    The targets are high-variance per-sample noise around the marginal,
    so matching them in expectation is the denoising trick working.  Then
    verify the trained net satisfies the integrated density-ratio identity
    log[p_t1/p_t0] = int net dt.
    """
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)
    schedule = VPSchedule(0.1, 8.0)
    mu1, sig1 = 1.0, 0.7

    t_emb = SinusoidalTimeEmbedding(16, max_period=100.0)
    net = nn.Sequential(
        nn.Linear(1 + 16, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 1),
    )

    def wrapped(x, t):
        feats = t_emb(t, leading_shape=x.shape[:-1], device=x.device, dtype=x.dtype)
        return net(torch.cat([x, feats], dim=-1))

    # The conditional targets carry several times the scale of the
    # marginal signal (worst at small t where sigma_t -> 0), so
    # convergence is statistics-limited: large batches + decaying lr
    # average the noise into the conditional mean.
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000, eta_min=1e-5)
    for step in range(8000):
        z = mu1 + sig1 * torch.randn(4096, 1, generator=generator)
        loss = ctsm_loss(wrapped, x_data=z, schedule=schedule, eps_t=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step}: ctsm loss {loss.item():.3e}")

    # (1) Marginal time score on a p_t-weighted grid at held-out times.
    for t_eval in (0.25, 0.5, 0.75):
        a = schedule.alpha(torch.tensor(t_eval))
        s = schedule.sigma(torch.tensor(t_eval))
        m = (a * mu1).item()
        sd = float(torch.sqrt(a**2 * sig1**2 + s**2))
        grid = torch.linspace(m - 3.5 * sd, m + 3.5 * sd, 201)
        tt = torch.full_like(grid, t_eval)
        with torch.no_grad():
            pred = wrapped(grid.unsqueeze(-1), tt).squeeze(-1)
        truth = _marginal_time_score(grid, tt, schedule, mu1, sig1)
        w = torch.exp(_marginal_logp(grid, tt, schedule, mu1, sig1))
        w = w / w.sum()
        rms = ((pred - truth).pow(2) * w).sum().sqrt()
        scale = (truth.pow(2) * w).sum().sqrt()
        print(
            f"t={t_eval}: weighted RMS err {rms.item():.3e} "
            f"(truth scale {scale.item():.3e})"
        )
        # Within 10% of signal scale despite per-sample targets missing
        # the marginal by ~350%.
        assert rms.item() < 0.10 * max(scale.item(), 1.0)

    # (2) Integrated identity: int_{t0}^{t1} net(x, t) dt = log p_t1 - log p_t0.
    t0, t1 = 0.15, 0.9
    taus = torch.linspace(t0, t1, 201)
    for x_val in (0.0, 0.5, 1.0, 1.5):
        x_rep = torch.full_like(taus, x_val)
        with torch.no_grad():
            integrand = wrapped(x_rep.unsqueeze(-1), taus).squeeze(-1)
        integral = torch.trapezoid(integrand, taus)
        truth = _marginal_logp(
            torch.tensor(x_val), torch.tensor(t1), schedule, mu1, sig1
        ) - _marginal_logp(torch.tensor(x_val), torch.tensor(t0), schedule, mu1, sig1)
        err = (integral - truth).abs().item()
        print(
            f"x={x_val}: int net dt = {integral:.4f}, truth {truth:.4f}, err {err:.3e}"
        )
        assert err < 0.1
