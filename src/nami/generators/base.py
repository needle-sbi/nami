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
        """Project raw field outputs into the operator's valid parameter set."""
        return params

    def validate_params(
        self, params: torch.Tensor, *, leading_shape: tuple[int, ...] | None = None
    ) -> None:
        """Assert that ``params`` matches ``parameter_shape`` (and optionally ``leading_shape``)."""
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
        """Pack semantically named tensors into the operator's parameter layout."""
        raise NotImplementedError

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the drift coefficient ``b(x, t)`` from packed parameters."""
        raise NotImplementedError

    def diffusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the diffusion coefficient ``g(x, t)`` from packed parameters."""
        raise NotImplementedError
