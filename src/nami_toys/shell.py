from __future__ import annotations


import math
from dataclasses import dataclass

import torch
from torch.distributions import Binomial

from .dataset import ToyDataset


@dataclass(frozen=True)
class GaussianShell:
    """2-D Gaussian shell (ring) signal with isotropic normal background.

    Parameters
    ----------
    radius : float
        Mean radius of the signal ring.
    width : float
        Standard deviation of the radial spread.
    bkg_scale : float
        Standard deviation of the isotropic background Gaussian.
    """

    radius: float = 2.5
    width: float = 0.25
    bkg_scale: float = 1.5

    def generate(
        self,
        n_expected: int,
        sig_frac: float,
        *,
        generator: torch.Generator | None = None,
    ) -> ToyDataset:
        """Draw a Poisson-fluctuated 2-D shell dataset."""
        n_total = int(torch.poisson(torch.tensor(float(n_expected))).item())
        n_sig = int(
            Binomial(n_total, probs=torch.tensor(float(sig_frac))).sample().item()
        )
        n_bkg = n_total - n_sig

        # signal: points on a noisy ring
        if n_sig > 0:
            angles = torch.empty(n_sig).uniform_(
                0.0, 2.0 * math.pi, generator=generator
            )
            radii = (
                torch.empty(n_sig)
                .normal_(self.radius, self.width, generator=generator)
                .clamp(min=0.0)
            )
            signal = torch.stack([radii * angles.cos(), radii * angles.sin()], dim=1)
        else:
            signal = torch.empty(0, 2)

        # background: isotropic 2-D Gaussian
        if n_bkg > 0:
            background = torch.empty(n_bkg, 2).normal_(
                0.0, self.bkg_scale, generator=generator
            )
        else:
            background = torch.empty(0, 2)

        x = torch.cat([signal, background], dim=0)
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
            meta={
                "n_expected": n_expected,
                "sig_frac": sig_frac,
                "radius": self.radius,
                "width": self.width,
                "bkg_scale": self.bkg_scale,
            },
        )
