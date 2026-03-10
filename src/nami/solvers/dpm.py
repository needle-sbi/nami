from __future__ import annotations

import torch

from ..fields.diffusion import _expand_like


class DPMSolverPP:
    """DPM-Solver++ inspired ODE solver, with a diffusion-specific fast path.

    The specialised ``integrate_diffusion`` method implements a 1st/2nd-order
    DPM-Solver++ update in log-SNR space.  The generic ``integrate`` and
    ``integrate_augmented`` methods fall back to a Heun (improved-Euler) scheme
    so the solver can be used as a drop-in replacement for other fixed-step
    solvers.
    """

    requires_steps = True
    supports_rsample = True
    is_sde = False

    def __init__(
        self,
        steps: int = 20,
        *,
        order: int = 2,
        skip_type: str = "time_uniform",
        sigma_min: float = 1e-12,
    ):
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)
        
        if order not in {1, 2}:
            msg = f"order must be 1 or 2, got {order}"
            raise ValueError(msg)
        
        if skip_type not in {"time_uniform", "logsnr"}:
            msg = "skip_type must be 'time_uniform' or 'logsnr'"
            raise ValueError(msg)
        
        if sigma_min <= 0:
            msg = f"sigma_min must be positive, got {sigma_min}"
            raise ValueError(msg)

        self.steps = int(steps)
        self.order = int(order)
        self.skip_type = skip_type
        self.sigma_min = float(sigma_min)

    def integrate(
        self,
        f,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,  # unused
        rtol: float = 1e-5,  # unused
        steps: int | None = None,
    ) -> torch.Tensor:
        # generic fallback (Heun)
        _ = atol, rtol
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)

        dt = (t1 - t0) / steps
        x = x0
        t = t0
        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + dt * k1, t + dt)
            x = x + 0.5 * dt * (k1 + k2)
            t = t + dt
        return x

    def integrate_augmented(
        self,
        f_aug,
        x0: torch.Tensor,
        logp0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        atol: float = 1e-6,  # unused
        rtol: float = 1e-5,  # unused
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # generic fallback (Heun), specifically for augmented states.
        _ = atol, rtol
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)

        dt = (t1 - t0) / steps
        x = x0
        logp = logp0
        t = t0
        for _ in range(steps):
            k1, l1 = f_aug(x, t)
            k2, l2 = f_aug(x + dt * k1, t + dt)
            x = x + 0.5 * dt * (k1 + k2)
            logp = logp + 0.5 * dt * (l1 + l2)
            t = t + dt
        return x, logp

    def integrate_diffusion(
        self,
        predict_eps,
        schedule,
        x0: torch.Tensor,
        *,
        t0: float,
        t1: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Fast diffusion ODE integration using a 1st/2nd-order DPM-Solver++ update."""
        steps = int(steps or self.steps)
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)

        times = self._time_steps(schedule, t0=t0, t1=t1, steps=steps, like=x0)
        x = x0

        t_curr = times[0]
        lambda_curr = self._lambda(schedule, t_curr)
        x0_curr = self._data_prediction(predict_eps, schedule, x, t_curr)

        x0_prev: torch.Tensor | None = None
        lambda_prev: torch.Tensor | None = None

        for idx in range(steps):
            t_next = times[idx + 1]
            lambda_next = self._lambda(schedule, t_next)
            h = lambda_next - lambda_curr

            alpha_next = _expand_like(self._alpha(schedule, t_next), x)
            sigma_curr = _expand_like(self._sigma(schedule, t_curr), x)
            sigma_next = _expand_like(self._sigma(schedule, t_next), x)
            phi_1 = _expand_like(torch.expm1(-h), x)

            if self.order == 1 or x0_prev is None or lambda_prev is None:
                x_next = (sigma_next / sigma_curr) * x - alpha_next * phi_1 * x0_curr
            else:
                h0 = lambda_curr - lambda_prev
                h_min = torch.where(
                    h >= 0,
                    torch.full_like(h, self.sigma_min),
                    torch.full_like(h, -self.sigma_min),
                )
                h_safe = torch.where(
                    torch.abs(h) < self.sigma_min,
                    h_min,
                    h,
                )
                r0 = h0 / h_safe
                d1 = (x0_curr - x0_prev) / _expand_like(r0, x)
                x_next = (
                    (sigma_next / sigma_curr) * x
                    - alpha_next * phi_1 * x0_curr
                    - 0.5 * alpha_next * phi_1 * d1
                )

            x = x_next
            x0_prev = x0_curr
            lambda_prev = lambda_curr
            t_curr = t_next
            lambda_curr = lambda_next

            if idx < steps - 1:
                x0_curr = self._data_prediction(predict_eps, schedule, x, t_curr)

        return x

    def _data_prediction(
        self, predict_eps, schedule, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        eps = predict_eps(x, t)
        alpha = _expand_like(self._alpha(schedule, t), x)
        sigma = _expand_like(self._sigma(schedule, t), x)
        return (x - sigma * eps) / alpha

    def _alpha(self, schedule, t: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(schedule.alpha(t), device=t.device, dtype=t.dtype)
        return torch.clamp(alpha, min=self.sigma_min)

    def _sigma(self, schedule, t: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(schedule.sigma(t), device=t.device, dtype=t.dtype)
        return torch.clamp(sigma, min=self.sigma_min)

    def _lambda(self, schedule, t: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(schedule, t)
        sigma = self._sigma(schedule, t)
        return torch.log(alpha) - torch.log(sigma)

    def _time_steps(
        self,
        schedule,
        *,
        t0: float,
        t1: float,
        steps: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if self.skip_type == "time_uniform":
            return torch.linspace(
                t0, t1, steps + 1, device=like.device, dtype=like.dtype
            )

        t0_tensor = torch.tensor(t0, device=like.device, dtype=like.dtype)
        t1_tensor = torch.tensor(t1, device=like.device, dtype=like.dtype)
        lambda_0 = self._lambda(schedule, t0_tensor).detach().item()
        lambda_1 = self._lambda(schedule, t1_tensor).detach().item()
        lambdas = torch.linspace(
            lambda_0, lambda_1, steps + 1, device=like.device, dtype=like.dtype
        )
        return self._inverse_lambda(schedule, lambdas, t0=t0, t1=t1)

    def _inverse_lambda(
        self, schedule, target_lambda: torch.Tensor, *, t0: float, t1: float
    ) -> torch.Tensor:
        lo_val = min(t0, t1)
        hi_val = max(t0, t1)
        lo = torch.full_like(target_lambda, lo_val)
        hi = torch.full_like(target_lambda, hi_val)

        lo_l = self._lambda(schedule, lo)
        hi_l = self._lambda(schedule, hi)
        increasing = bool((hi_l.mean() - lo_l.mean()) >= 0)

        for _ in range(64):
            mid = 0.5 * (lo + hi)
            mid_l = self._lambda(schedule, mid)
            if increasing:
                go_right = mid_l < target_lambda
            else:
                go_right = mid_l > target_lambda
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)

        return 0.5 * (lo + hi)
