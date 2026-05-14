from __future__ import annotations


from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch.distributions import Binomial, MultivariateNormal

from .dataset import ToyDataset

_DEFAULT_SIG_LOC = torch.tensor([0.0, 0.0])
_DEFAULT_SIG_COV = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
_DEFAULT_BKG_LOC = torch.tensor([0.0, 0.0])
_DEFAULT_BKG_COV = torch.tensor([[2.0, -0.2], [-0.2, 2.0]])


@dataclass(frozen=True)
class ParameterisedGaussian:
    r"""Gaussian mixture whose signal location depends on a parameter :math:`\theta`.

    The background distribution is fixed while the signal mean is set to
    :math:`\theta` along ``param_dim``, keeping the base mean elsewhere.

    Parameters
    ----------
    sig_loc : torch.Tensor
        Base signal mean ``(d,)``; entry at *param_dim* is replaced by theta.
    sig_cov : torch.Tensor
        Signal covariance ``(d, d)`` (fixed).
    bkg_loc, bkg_cov : torch.Tensor
        Background mean and covariance (fixed).
    sig_frac : float
        Expected signal fraction.
    param_dim : int
        Dimension of the mean vector that theta controls.
    """

    sig_loc: torch.Tensor = field(default_factory=lambda: _DEFAULT_SIG_LOC.clone())
    sig_cov: torch.Tensor = field(default_factory=lambda: _DEFAULT_SIG_COV.clone())
    bkg_loc: torch.Tensor = field(default_factory=lambda: _DEFAULT_BKG_LOC.clone())
    bkg_cov: torch.Tensor = field(default_factory=lambda: _DEFAULT_BKG_COV.clone())
    sig_frac: float = 0.3
    param_dim: int = 0

    @cached_property
    def bkg(self) -> MultivariateNormal:
        return MultivariateNormal(self.bkg_loc, self.bkg_cov)

    @property
    def d(self) -> int:
        return self.sig_loc.shape[0]

    def sig_at(self, theta: float) -> MultivariateNormal:
        """Return the signal distribution at a given *theta*."""
        mu = self.sig_loc.clone()
        mu[self.param_dim] = theta
        return MultivariateNormal(mu, self.sig_cov)

    def log_prob(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        r"""Mixture log-probability :math:`\log p(x \mid \theta)`.

        Parameters
        ----------
        x : torch.Tensor
            Events ``(N, d)`` or ``(d,)``.
        theta : float
            Parameter value.
        """
        sig = self.sig_at(theta)
        p = (
            self.sig_frac * sig.log_prob(x).exp()
            + (1 - self.sig_frac) * self.bkg.log_prob(x).exp()
        )
        return p.log()

    def log_likelihood_ratio(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        r"""Per-event log-likelihood ratio :math:`\log p(x \mid \text{sig}, \theta) - \log p(x \mid \text{bkg})`."""
        return self.sig_at(theta).log_prob(x) - self.bkg.log_prob(x)

    def generate(
        self,
        theta: float,
        n_expected: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw a Poisson-fluctuated dataset at the given *theta*."""
        n_total = int(torch.poisson(torch.tensor(float(n_expected))).item())
        n_sig = int(
            Binomial(n_total, probs=torch.tensor(float(self.sig_frac))).sample().item()
        )
        n_bkg = n_total - n_sig

        sig = self.sig_at(theta)
        x = torch.cat([sig.sample((n_sig,)), self.bkg.sample((n_bkg,))], dim=0)

        y = torch.cat(
            [
                torch.ones(n_sig, dtype=torch.long),
                torch.zeros(n_bkg, dtype=torch.long),
            ]
        )

        perm = torch.randperm(n_total, generator=generator)
        return ToyDataset(
            x=x[perm],
            y=y[perm],
            meta={"n_expected": n_expected, "sig_frac": self.sig_frac, "theta": theta},
        )
