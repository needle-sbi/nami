from __future__ import annotations

import math

import torch

from .base import ProbabilityPath


class CosinePath(ProbabilityPath):
    def sample_xt(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = z
        alpha_t = self._alpha(t)
        sigma_t = self._sigma(t)
        alpha_t_reshaped = alpha_t.reshape(
            alpha_t.shape + (1,) * (x_target.ndim - alpha_t.ndim)
        )
        sigma_t_reshaped = sigma_t.reshape(
            sigma_t.shape + (1,) * (x_target.ndim - sigma_t.ndim)
        )

        return alpha_t_reshaped * x_target + sigma_t_reshaped * x_source

    def target_ut(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        xt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = xt
        alpha_prime = self._alpha_prime(t)
        sigma_prime = self._sigma_prime(t)

        alpha_prime_reshaped = alpha_prime.reshape(
            alpha_prime.shape + (1,) * (x_target.ndim - alpha_prime.ndim)
        )
        sigma_prime_reshaped = sigma_prime.reshape(
            sigma_prime.shape + (1,) * (x_target.ndim - sigma_prime.ndim)
        )

        return alpha_prime_reshaped * x_target + sigma_prime_reshaped * x_source

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t * math.pi / 2)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t * math.pi / 2)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2) * torch.sin(t * math.pi / 2)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return (math.pi / 2) * torch.cos(t * math.pi / 2)
