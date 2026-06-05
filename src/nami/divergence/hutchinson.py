"""Hutchinson stochastic-trace divergence estimator.

Single-sample unbiased estimate ``\\mathbb{E}_\\epsilon[\\epsilon^T J \\epsilon] = \\mathrm{tr}(J)``
with ``\\epsilon`` drawn from a Rademacher or Gaussian probe. One vJp call
per evaluation regardless of event size.

References
----------
- Hutchinson, 1989 — original trace estimator.
- Grathwohl et al., *FFJORD*, 2018 (arXiv:1810.01367) — Hutchinson
  trace integrated through a continuous flow.
"""

from __future__ import annotations

import torch

from nami.core.specs import split_event
from nami.divergence.base import DivergenceEstimator


def _rademacher_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)


class HutchinsonDivergence(DivergenceEstimator):
    """Hutchinson stochastic trace estimator.

    Parameters
    ----------
    probe : {"rademacher", "gaussian"}
        Probe distribution. Rademacher has the lower variance for
        diagonally dominant Jacobians and is the default.
    create_graph : bool
        Retain the autograd graph through the estimate.
    """

    def __init__(self, probe: str = "rademacher", *, create_graph: bool = False):
        if probe not in {"rademacher", "gaussian"}:
            msg = "probe must be 'rademacher' or 'gaussian'"
            raise ValueError(msg)
        self.probe = probe
        self.create_graph = bool(create_graph)

    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor:
        spec = getattr(field, "spec", None)
        event_ndim = spec.event_ndim if spec is not None else getattr(field, "event_ndim", None)
        if event_ndim is None:
            msg = "field.event_ndim is required for divergence"
            raise ValueError(msg)
        lead, _event_shape = split_event(x, event_ndim)

        with torch.enable_grad():
            # clone to avoid mutating input tensor's grad state
            x_req = x.detach().clone().requires_grad_(True)

            # field call must be inside enable_grad() to build computation graph
            v = field(x_req, t, c)

            if self.probe == "gaussian":
                eps = torch.randn_like(x_req)
            else:
                eps = _rademacher_like(x_req)

            dot = (v * eps).sum()
            grad = torch.autograd.grad(
                dot,
                x_req,
                create_graph=self.create_graph,
            )[0]

        return (grad * eps).reshape(*lead, -1).sum(dim=-1)
