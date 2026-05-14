from __future__ import annotations



from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch.distributions import Binomial, MultivariateNormal

from .dataset import ToyDataset

_DEFAULT_SIG_LOC = torch.tensor([1.0, 0.0])
_DEFAULT_SIG_COV = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
_DEFAULT_BKG_LOC = torch.tensor([0.0, 0.0])
_DEFAULT_BKG_COV = torch.tensor([[2.0, -0.2], [-0.2, 2.0]])


@dataclass(frozen=True)
class GaussianMixture:
    """N-dimensional Gaussian signal + background simulator.

    Parameters
    ----------
    sig_loc, sig_cov : torch.Tensor
        Mean ``(d,)`` and covariance ``(d, d)`` of the signal component.
    bkg_loc, bkg_cov : torch.Tensor
        Mean ``(d,)`` and covariance ``(d, d)`` of the background component.
    """

    sig_loc: torch.Tensor = field(default_factory=lambda: _DEFAULT_SIG_LOC.clone())
    sig_cov: torch.Tensor = field(default_factory=lambda: _DEFAULT_SIG_COV.clone())
    bkg_loc: torch.Tensor = field(default_factory=lambda: _DEFAULT_BKG_LOC.clone())
    bkg_cov: torch.Tensor = field(default_factory=lambda: _DEFAULT_BKG_COV.clone())

    @cached_property
    def sig(self) -> MultivariateNormal:
        return MultivariateNormal(self.sig_loc, self.sig_cov)

    @cached_property
    def bkg(self) -> MultivariateNormal:
        return MultivariateNormal(self.bkg_loc, self.bkg_cov)

    @property
    def d(self) -> int:
        return self.sig_loc.shape[0]

    def generate(
        self,
        n_expected: int,
        sig_frac: float,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw a Poisson-fluctuated dataset of signal + background events."""
        n_total = int(torch.poisson(torch.tensor(float(n_expected))).item())
        n_sig = int(
            Binomial(n_total, probs=torch.tensor(float(sig_frac))).sample().item()
        )
        n_bkg = n_total - n_sig

        sig_data = self.sig.sample((n_sig,))
        bkg_data = self.bkg.sample((n_bkg,))
        x = torch.cat([sig_data, bkg_data], dim=0)

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
            meta={"n_expected": n_expected, "sig_frac": sig_frac},
        )
