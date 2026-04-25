from __future__ import annotations

from typing import Protocol

import torch


class DivergenceEstimator(Protocol):
    """Interface for computing divergence of velocity field in log-likelihood calc. via change of variables."""

    def __call__(
        self, field, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor: ...


class ProbabilityPath(Protocol):
    """Interface for interpolation paths in the fm models

    Methods:
    --------
    - `sample_xt`: given data, noise and time, return the interpolated point along the path
    - `target_ut`: ground truth velocity field used in the loss
    """

    def sample_xt(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor: ...

    def target_ut(
        self, x_target: torch.Tensor, x_source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor: ...


class NoiseSchedule(Protocol):
    """Interface for diffusion model noise schedules with forward process
    x_t = alpha(t)*x_0 + sigma(t) * epsilon

    Methods:
    --------
    - `alpha(t)`: signal scaling coeff. at time t
    - `sigma(t)`: noise scaling coeff. at time t
    - `snr`: signal-to-noise ratio (alpha^2/sigma^2)
    - `drift(x,t)`: term in SDE dx = f(x,t)dt + g(t) dW
    - `diffusion`: the diffusion coeff. g(t)
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor: ...

    def sigma(self, t: torch.Tensor) -> torch.Tensor: ...

    def snr(self, t: torch.Tensor) -> torch.Tensor: ...

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...

    def diffusion(self, t: torch.Tensor) -> torch.Tensor: ...


class ODESolver(Protocol):
    """Interface for ODE integrators.

    Methods:
    --------
    - `integrate`: solve dx/dt = f(x,t) from t0 to t1 given initial state x0
    - `integrate_augmented`: jointly solve for state and the log-prob change
    """

    # does the solver require a fixed number of step counts to be specified?
    requires_steps: bool

    # can the solver produce reparameterised samples (i.e. gradients flow through the sampling)
    supports_rsample: bool

    is_sde: bool  # should be False for ODE solvers

    def integrate(
        self,
        f,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        steps: int | None = None,
    ) -> torch.Tensor: ...

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class SDESolver(Protocol):
    """Interface for SDE integrators.

    Methods:
    - `integrate`
    """

    is_sde: bool  # should be True for SDE solvers

    def integrate(
        self,
        drift,
        diffusion,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int,
    ) -> torch.Tensor: ...
