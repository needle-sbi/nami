"""Concrete generator operators.

Currently implements :class:`ItoGeneratorOperator` — continuous-time
generators with drift and optional diagonal diffusion. The diffusion
mode toggles between ODE (``"none"``) and SDE (``"diagonal"``)
runtime; the parameter layout grows from event-shape to
``(2, *event_shape)`` accordingly.

References
----------
- Holderrieth et al., *Generator Matching*, 2024.
"""

from __future__ import annotations




import torch
import torch.nn.functional as F

from nami.core.specs import validate_shapes
from nami.fields._common import normalise_event_shape
from nami.generators.base import GeneratorOperator


class ItoGeneratorOperator(GeneratorOperator):
    """Continuous Ito generator with drift and optional diagonal diffusion.

    Parameters
    ----------
    event_shape : int or tuple[int, ...]
        Shape of a single event.
    diffusion : {"none", "diagonal"}
        ``"none"`` runs as an ODE (drift only); ``"diagonal"`` adds a
        positive diagonal diffusion via softplus projection.
    min_scale : float
        Floor added to the softplus diffusion for numerical stability.
    """


    def __init__(
        self,
        event_shape: int | tuple[int, ...],
        *,
        diffusion: str = "none",
        min_scale: float = 1e-4,
    ):
        if diffusion not in {"none", "diagonal"}:
            msg = "diffusion must be 'none' or 'diagonal'"
            raise ValueError(msg)
        if min_scale < 0:
            msg = f"min_scale must be non-negative, got {min_scale}"
            raise ValueError(msg)

        self._event_shape = normalise_event_shape(event_shape)
        self.diffusion_mode = diffusion
        self.min_scale = float(min_scale)
        super().__init__(runtime_kind="ode" if diffusion == "none" else "sde")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        if self.diffusion_mode == "none":
            return self.event_shape
        return (2, *self.event_shape)

    def _split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.validate_params(params)
        if self.diffusion_mode == "none":
            return params, None
        drift, diffusion = torch.unbind(params, dim=-(self.event_ndim + 1))
        return drift, diffusion

    def pack_params(
        self,
        *,
        drift: torch.Tensor,
        diffusion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_shapes(drift, self.event_ndim, expected_event_shape=self.event_shape)
        if self.diffusion_mode == "none":
            if diffusion is not None:
                msg = "diffusion parameters are not used when diffusion='none'"
                raise ValueError(msg)
            return drift

        if diffusion is None:
            msg = "diffusion tensor is required when diffusion='diagonal'"
            raise ValueError(msg)
        validate_shapes(
            diffusion,
            self.event_ndim,
            expected_event_shape=self.event_shape,
        )
        return torch.stack((drift, diffusion), dim=-(self.event_ndim + 1))

    def project(self, params: torch.Tensor) -> torch.Tensor:
        self.validate_params(params)
        drift, diffusion = self._split_params(params)
        if diffusion is None:
            return drift
        projected_diffusion = F.softplus(diffusion) + self.min_scale
        return self.pack_params(drift=drift, diffusion=projected_diffusion)

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        _ = x, t
        drift, _ = self._split_params(params)
        return drift

    def diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        _ = x, t
        _, diffusion = self._split_params(params)
        if diffusion is None:
            msg = "diffusion is unavailable when diffusion='none'"
            raise NotImplementedError(msg)
        return diffusion
