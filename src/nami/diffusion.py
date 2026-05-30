r"""Algebraic conversions between diffusion prediction targets.

For a Gaussian conditional path

.. math::

   x_t = \alpha(t) x_0 + \sigma(t)\epsilon,

the functions in this module convert between standard diffusion targets:
noise ``\epsilon``, score ``\nabla_x \log p_t(x_t)``, clean endpoint
``x_0``, and v-prediction
``v = \alpha(t)\epsilon - \sigma(t)x_0``.  The schedule values
``\alpha(t)`` and ``\sigma(t)`` are inputs; this module does not define a
schedule.
"""

from __future__ import annotations

import torch


def expand_like(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Broadcast a schedule tensor over event dimensions.

    Args:
        scale (torch.Tensor): Tensor with leading shape ``lead``.
        target (torch.Tensor): Tensor with shape ``lead + event_shape``.

    Returns:
        torch.Tensor: ``scale`` reshaped with trailing singleton dimensions so
        it broadcasts with ``target``.
    """
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def eps_to_score(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""Convert noise prediction to score prediction.

    The conversion is

    .. math::

       \nabla_x \log p_t(x_t) = -\epsilon / \sigma(t).

    Args:
        eps (torch.Tensor): Predicted standardized noise ``\epsilon``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)`` with leading shape
            matching ``eps``.

    Returns:
        torch.Tensor: Score tensor with the same shape as ``eps``.
    """
    return -eps / expand_like(sigma, eps)


def score_to_eps(score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""Convert score prediction to noise prediction.

    Args:
        score (torch.Tensor): Score ``\nabla_x \log p_t(x_t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Standardized noise
        ``\epsilon = -\sigma(t)\nabla_x \log p_t(x_t)``.
    """
    return -score * expand_like(sigma, score)


def eps_to_x0(
    x: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""Convert noise prediction to clean-endpoint prediction.

    Args:
        x (torch.Tensor): Current state ``x_t``.
        eps (torch.Tensor): Predicted standardized noise ``\epsilon``.
        alpha (torch.Tensor): Signal scale ``\alpha(t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Clean endpoint
        ``x_0 = (x_t - \sigma(t)\epsilon) / \alpha(t)``.
    """
    return (x - expand_like(sigma, x) * eps) / expand_like(alpha, x)


def x0_to_eps(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""Convert clean-endpoint prediction to noise prediction.

    Args:
        x (torch.Tensor): Current state ``x_t``.
        x0 (torch.Tensor): Predicted clean endpoint ``x_0``.
        alpha (torch.Tensor): Signal scale ``\alpha(t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Standardized noise
        ``\epsilon = (x_t - \alpha(t)x_0) / \sigma(t)``.
    """
    return (x - expand_like(alpha, x) * x0) / expand_like(sigma, x)


def score_to_x0(
    x: torch.Tensor, score: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""Convert score prediction to clean-endpoint prediction.

    Args:
        x (torch.Tensor): Current state ``x_t``.
        score (torch.Tensor): Score ``\nabla_x \log p_t(x_t)``.
        alpha (torch.Tensor): Signal scale ``\alpha(t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Clean endpoint
        ``x_0 = (x_t + \sigma^2(t)\nabla_x\log p_t(x_t)) / \alpha(t)``.
    """
    sigma_exp = expand_like(sigma, x)
    return (x + (sigma_exp**2) * score) / expand_like(alpha, x)


def x0_to_score(
    x: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    r"""Convert clean-endpoint prediction to score prediction.

    Args:
        x (torch.Tensor): Current state ``x_t``.
        x0 (torch.Tensor): Predicted clean endpoint ``x_0``.
        alpha (torch.Tensor): Signal scale ``\alpha(t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Score
        ``\nabla_x\log p_t(x_t) = (\alpha(t)x_0 - x_t) / \sigma^2(t)``.
    """
    sigma_exp = expand_like(sigma, x)
    return (expand_like(alpha, x) * x0 - x) / (sigma_exp**2)


def v_to_eps(
    x: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    r"""Convert v-prediction to noise prediction.

    From

    .. math::

       v = \alpha(t)\epsilon - \sigma(t)x_0,\qquad
       x_t = \alpha(t)x_0 + \sigma(t)\epsilon,

    Cramer's rule gives

    .. math::

       \epsilon =
       \frac{\sigma(t)x_t + \alpha(t)v}{\alpha^2(t)+\sigma^2(t)}.

    Args:
        x (torch.Tensor): Current state ``x_t``.
        v (torch.Tensor): Predicted v-target.
        alpha (torch.Tensor): Signal scale ``\alpha(t)``.
        sigma (torch.Tensor): Noise scale ``\sigma(t)``.

    Returns:
        torch.Tensor: Standardized noise ``\epsilon`` with the same shape as
        ``x``.
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
