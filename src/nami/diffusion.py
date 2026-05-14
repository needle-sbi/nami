r"""Algebraic conversions between diffusion prediction targets.

Given a Gaussian conditional path ``x_t = \alpha(t) x_0 + \\sigma(t) \epsilon`` and its
named target objects (``\\epsilon``, \\score ``\nabla \log p_t``, clean ``x_0``,
v-prediction ``v = \alpha \epsilon - \sigma x_0``), these helpers convert one target
into another using only ``(x_t, \alpha, \sigma)``.  Used by:

* :class:`~nami.processes.diffusion.DiffusionProcess` —
  pattern-matches on ``parameterization.target`` and dispatches the
  appropriate conversion to produce the ``\epsilon`` the integrator
  consumes.
* :class:`~nami.interpolants.gaussian.GaussianInterpolant` —
  derives Score / V targets at training time from the closed-form
  formulas.
* :class:`~nami.solvers.dpm.DPMSolverPP` — uses
  :func:`expand_like` to broadcast schedule scalars over batched
  state tensors.

These are algebraic identities between target spaces, so schedule-adjacent but not
schedules in themselves, since they don't define ``\alpha(t), \sigma(t)``; they
consume them.
"""

from __future__ import annotations

import torch


def expand_like(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Broadcast ``scale`` to match ``target``'s rank by appending
    trailing singleton dims.

    Used to broadcast schedule scalars ``\alpha(t), \sigma(t)`` (shape ``lead``)
    across event dimensions of state tensors (shape
    ``lead + event_shape``).
    """
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def eps_to_score(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""``score = -\epsilon / \sigma(t)``.

    Diffusion-convention conversion (``x_t = \alpha(t)x_0 + \sigma(t)\epsilon``).
    Used by the score-based reverse-time PF-ODE inside
    :class:`nami.processes.diffusion.Diffusion`, which deliberately
    retains the diffusion convention even after the broader Phase 3 flip.
    """
    return -eps / expand_like(sigma, eps)


def score_to_eps(score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""``\epsilon = -\sigma(t) \cdot score`` (diffusion convention)."""
    return -score * expand_like(sigma, score)


def eps_to_x0(
    x: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""``x_0 = (x_t - \sigma(t) \epsilon) / \alpha(t)`` (diffusion convention)."""
    return (x - expand_like(sigma, x) * eps) / expand_like(alpha, x)


def x0_to_eps(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""``\epsilon = (x_t - \alpha(t) x_0) / \sigma(t)`` (diffusion convention)."""
    return (x - expand_like(alpha, x) * x0) / expand_like(sigma, x)


def score_to_x0(
    x: torch.Tensor, score: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""``x_0 = (x_t + \sigma^2(t) \cdot score) / \alpha(t)`` (diffusion convention)."""
    sigma_exp = expand_like(sigma, x)
    return (x + (sigma_exp**2) * score) / expand_like(alpha, x)


def x0_to_score(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""``score = (\alpha(t) x_0 - x_t) / \sigma^2(t)`` (diffusion convention)."""
    sigma_exp = expand_like(sigma, x)
    return (expand_like(alpha, x) * x0 - x) / (sigma_exp**2)


def v_to_eps(
    x: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    r"""Salimans-Ho v-prediction to ``\epsilon`` conversion (diffusion convention).

    From the system ``v = \alpha \epsilon - \sigma x_0`` and ``x_t = \alpha x_0 + \sigma \epsilon``,
    Cramer's rule gives::

        \epsilon = (\sigma \cdot x_t + \alpha \cdot v) / (\alpha^2 + \sigma^2)

    The denominator is exactly 1 for variance-preserving schedules
    (VP, EDM with the standard reparameterisation) but is generally
    ``1 + \sigma^2`` for variance-exploding schedules; the formula handles
    both without assuming variance preservation.
    """
    alpha_exp = expand_like(alpha, x)
    sigma_exp = expand_like(sigma, x)
    denom = alpha_exp.pow(2) + sigma_exp.pow(2)
    return (sigma_exp * x + alpha_exp * v) / denom


__all__ = [
    "eps_to_score",
    "eps_to_x0",
    "expand_like",
    "score_to_eps",
    "score_to_x0",
    "v_to_eps",
    "x0_to_eps",
    "x0_to_score",
]
