"""Abstract base class for generator operators.

A generator operator declares its ``runtime_kind`` (ode / sde / jump)
and exposes the contract to ``pack_params``, ``project`` raw outputs
into the valid parameter set, and produce ``drift`` / ``diffusion``
coefficients for the chosen simulator.

References
----------
- Holderrieth et al., *Generator Matching*, 2024.
"""

from __future__ import annotations

import torch

_RUNTIME_KINDS = {"ode", "sde", "jump"}


class GeneratorOperator:
    """Base class for parameterized generators.

    Args:
        runtime_kind (str): Simulator family, one of ``"ode"``, ``"sde"``, or
            ``"jump"``.

    Operators define how a field output ``F_t(x)`` is interpreted at runtime.
    The field stays agnostic and simply predicts parameters with the standard
    ``forward(x, t, c=None)`` contract.
    """

    def __init__(self, *, runtime_kind: str):
        if runtime_kind not in _RUNTIME_KINDS:
            msg = f"runtime_kind must be one of {_RUNTIME_KINDS}, got {runtime_kind}"
            raise ValueError(msg)
        self._runtime_kind = runtime_kind

    @property
    def runtime_kind(self) -> str:
        return self._runtime_kind

    @property
    def event_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def event_ndim(self) -> int:
        return len(self.event_shape)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def project(self, params: torch.Tensor) -> torch.Tensor:
        """Project raw field outputs into the valid parameter set.

        Args:
            params (torch.Tensor): Raw parameter tensor.

        Returns:
            torch.Tensor: Projected parameter tensor.
        """
        return params

    def validate_params(
        self, params: torch.Tensor, *, leading_shape: tuple[int, ...] | None = None
    ) -> None:
        """Validate packed parameter shape.

        Args:
            params (torch.Tensor): Packed parameter tensor.
            leading_shape (tuple[int, ...] | None): Optional expected leading
                shape before ``parameter_shape``.

        Raises:
            ValueError: If either shape does not match the operator contract.
        """
        param_shape = tuple(params.shape[-len(self.parameter_shape) :])
        if param_shape != self.parameter_shape:
            msg = (
                f"parameter_shape mismatch: expected {self.parameter_shape}, "
                f"got {param_shape}"
            )
            raise ValueError(msg)
        if leading_shape is not None:
            actual_leading = (
                tuple(params.shape[: -len(self.parameter_shape)])
                if self.parameter_shape
                else tuple(params.shape)
            )
            if actual_leading != leading_shape:
                msg = (
                    f"parameter leading shape mismatch: expected {leading_shape}, "
                    f"got {actual_leading}"
                )
                raise ValueError(msg)

    def pack_params(
        self,
        *,
        drift: torch.Tensor,
        diffusion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack named coefficients into the operator parameter layout.

        Args:
            drift (torch.Tensor): Drift coefficient ``b(x,t)``.
            diffusion (torch.Tensor | None): Optional diffusion coefficient
                ``g(x,t)``.

        Returns:
            torch.Tensor: Packed parameter tensor.
        """
        raise NotImplementedError

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the drift coefficient from packed parameters.

        Args:
            x (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.
            params (torch.Tensor): Packed parameter tensor.

        Returns:
            torch.Tensor: Drift coefficient ``b(x,t)``.
        """
        raise NotImplementedError

    def diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the diffusion coefficient from packed parameters.

        Args:
            x (torch.Tensor): State tensor.
            t (torch.Tensor): Time tensor.
            params (torch.Tensor): Packed parameter tensor.

        Returns:
            torch.Tensor: Diffusion coefficient ``g(x,t)``.
        """
        raise NotImplementedError
