"""Tests for the nami_toys package."""

from __future__ import annotations

import torch

from nami_toys import (
    Checkerboard,
    GaussianMixture,
    GaussianRing,
    GaussianShell,
    ParameterisedGaussian,
    Standardiser,
    ToyDataset,
    TwoMoons,
    TwoSpirals,
    make_generator,
)

# ------------------------------------------------------------------------
# Gaussian mixtures


def test_gaussian_generate():
    sim = GaussianMixture()
    ds = sim.generate(1000, 0.3)

    assert isinstance(ds, ToyDataset)
    assert ds.x.ndim == 2
    assert ds.x.shape[1] == 2
    assert ds.y is not None
    assert ds.y.shape == (ds.x.shape[0],)
    assert len(ds) == ds.x.shape[0]
    assert ds.y.dtype == torch.long


def test_gaussian_custom():
    loc = torch.zeros(5)
    cov = torch.eye(5)
    sim = GaussianMixture(sig_loc=loc, sig_cov=cov, bkg_loc=loc, bkg_cov=cov)

    assert sim.d == 5
    ds = sim.generate(500, 0.5)
    assert ds.x.shape[1] == 5


def test_parameterised_generate():
    sim = ParameterisedGaussian()
    ds = sim.generate(theta=1.5, n_expected=1000)
    assert ds.x.ndim == 2
    assert ds.x.shape[1] == 2
    assert ds.y is not None
    assert ds.meta["theta"] == 1.5


def test_parameterised_theta_shifts_signal():
    sim = ParameterisedGaussian(sig_frac=1.0)  # all signal
    ds_lo = sim.generate(theta=-3.0, n_expected=2000)
    ds_hi = sim.generate(theta=3.0, n_expected=2000)
    # signal mean in dim 0 should differ noticeably
    assert ds_hi.x[:, 0].mean() > ds_lo.x[:, 0].mean() + 1.0


def test_parameterised_log_prob():
    sim = ParameterisedGaussian()
    x = torch.randn(50, 2)
    lp = sim.log_prob(x, theta=1.0)
    assert lp.shape == (50,)
    assert lp.isfinite().all()


def test_parameterised_log_likelihood_ratio():
    sim = ParameterisedGaussian()
    x = torch.randn(10, 2)
    llr = sim.log_likelihood_ratio(x, theta=0.0)
    assert llr.shape == (10,)
    assert llr.isfinite().all()


def test_shell_generate():
    sim = GaussianShell()
    ds = sim.generate(1000, 0.5)

    assert isinstance(ds, ToyDataset)
    assert ds.x.ndim == 2
    assert ds.x.shape[1] == 2
    assert ds.y is not None
    assert ds.y.dtype == torch.long
    assert len(ds) == ds.x.shape[0]


def test_dataset_subset():
    x = torch.randn(100, 3)
    y = torch.randint(0, 2, (100,))
    ds = ToyDataset(x=x, y=y, meta={"info": "test"})

    mask = y == 1
    sub = ds.subset(mask)
    assert len(sub) == int(mask.sum())
    assert sub.y is not None
    assert (sub.y == 1).all()


def test_dataset_limit():
    ds = ToyDataset(x=torch.randn(200, 4), y=torch.zeros(200, dtype=torch.long))
    limited = ds.limit(50)
    assert len(limited) == 50
    assert limited.x.shape == (50, 4)


# ------------------------------------------------------------------------
# two-moons, checkerboard, ring, spiral


def test_moons_generate():
    ds = TwoMoons().generate(500)
    assert ds.x.shape == (500, 2)
    assert ds.y is not None
    assert set(ds.y.unique().tolist()) == {0, 1}


def test_moons_noise():
    ds_lo = TwoMoons(noise=0.01).generate(1000)
    ds_hi = TwoMoons(noise=1.0).generate(1000)
    assert ds_lo.x.std() < ds_hi.x.std()


def test_checkerboard_generate():
    ds = Checkerboard().generate(1000)
    assert ds.x.shape == (1000, 2)
    assert ds.y is None  # no natural labels


def test_checkerboard_bounds():
    cb = Checkerboard(cells=4, bound=2.0)
    ds = cb.generate(5000)
    assert ds.x[:, 0].min() >= -2.0
    assert ds.x[:, 0].max() <= 2.0


def test_ring_generate():
    ds = GaussianRing().generate(800)
    assert ds.x.shape == (800, 2)
    assert ds.y is not None
    assert ds.y.unique().numel() <= 8


def test_ring_custom_modes():
    ds = GaussianRing(n_modes=4, radius=5.0, std=0.1).generate(2000)
    assert ds.y.unique().numel() <= 4
    # with tight std, points should cluster near radius=5
    norms = ds.x.norm(dim=1)
    assert (norms.mean() - 5.0).abs() < 0.5


def test_spirals_generate():
    ds = TwoSpirals().generate(600)
    assert ds.x.shape == (600, 2)
    assert ds.y is not None
    assert set(ds.y.unique().tolist()) == {0, 1}


# ------------------------------------------------------------------------
# standardiser


def test_standardiser_zero_mean_unit_var():
    x = torch.randn(5000, 3) * 5.0 + 10.0
    s = Standardiser.fit(x)
    z = s.transform(x)
    assert z.mean(dim=0).abs().max() < 0.1
    assert (z.std(dim=0) - 1.0).abs().max() < 0.1


def test_standardiser_roundtrip():
    x = torch.randn(200, 4) * 3.0 - 2.0
    s = Standardiser.fit(x)
    assert torch.allclose(s.inverse(s.transform(x)), x, atol=1e-5)


def test_standardiser_call_is_transform():
    x = torch.randn(50, 2)
    s = Standardiser.fit(x)
    assert torch.equal(s(x), s.transform(x))


def test_standardiser_transform_dataset():
    ds = ToyDataset(x=torch.randn(100, 3) * 5.0, y=torch.ones(100, dtype=torch.long))
    s = Standardiser.fit(ds.x)
    ds_z = s.transform_dataset(ds)
    assert ds_z.x.shape == ds.x.shape
    assert ds_z.y is not None
    assert torch.equal(ds_z.y, ds.y)
    assert ds_z.x.mean(dim=0).abs().max() < 0.5


def test_standardiser_constant_feature():
    x = torch.zeros(100, 2)
    x[:, 1] = 3.0  # constant column
    s = Standardiser.fit(x)
    z = s.transform(x)
    assert z.isfinite().all()


# ------------------------------------------------------------------------
# rng helpers


def test_make_generator_seeded():
    g1 = make_generator(42)
    g2 = make_generator(42)
    assert torch.equal(torch.randn(5, generator=g1), torch.randn(5, generator=g2))


def test_make_generator_none():
    assert make_generator(None) is None


def test_make_generator_with_toy():
    ds1 = TwoMoons().generate(200, generator=make_generator(0))
    ds2 = TwoMoons().generate(200, generator=make_generator(0))
    assert torch.equal(ds1.x, ds2.x)
