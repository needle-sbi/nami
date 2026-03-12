from __future__ import annotations

import torch

from ..core.specs import event_numel, split_event
from .base import DivergenceEstimator


class ExactDivergence(DivergenceEstimator):
    def __init__(self, max_dim: int = 16, *, create_graph: bool = False):
        self.max_dim = int(max_dim)
        self.create_graph = bool(create_graph)

    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor:
        event_ndim = getattr(field, "event_ndim", None)
        if event_ndim is None:
            msg = "field.event_ndim is required for divergence"
            raise ValueError(msg)
        lead, event_shape = split_event(x, event_ndim)
        numel = event_numel(event_shape)
        if numel > self.max_dim:
            msg = "event_numel exceeds ExactDivergence.max_dim"
            raise ValueError(msg)

        with torch.enable_grad():
            # Clone to avoid mutating input tensor's grad state
            x_req = x.detach().clone().requires_grad_(True)

            # Field call must be inside enable_grad() to build computation graph
            v = field(x_req, t, c)
            v_flat = v.reshape(*lead, numel)
            div = torch.zeros(lead, device=x.device, dtype=x.dtype)

            for i in range(numel):
                grad = torch.autograd.grad(
                    v_flat[..., i].sum(),
                    x_req,
                    create_graph=self.create_graph,
                    # When create_graph=True we must retain the graph on every
                    # component pull, otherwise downstream backward through
                    # log_prob can fail with "backward through graph a second time".
                    retain_graph=(self.create_graph or i < numel - 1),
                )[0]
                grad_flat = grad.reshape(*lead, numel)
                div = div + grad_flat[..., i]

        return div
