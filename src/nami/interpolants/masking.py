r"""Masking interpolant for the masking-CTMC generator.

Implements the absorbing-mask conditional path: at time ``t`` each token
coordinate is independently either still ``MASK`` (with probability
``\alpha(t)``) or revealed to its clean value. The conditional generator target
``F_t^z`` is the clean-token categorical distribution (a one-hot over the data
tokens), matched with the KL / cross-entropy Bregman divergence.

This interpolant is operator-aware: it reads the vocabulary size, the mask
index, and the masking schedule from a
:class:`~nami.generators.ctmc.CTMCGeneratorOperator`, so the forward masking and
the sampler's unmasking schedule cannot drift out of sync.
"""

from __future__ import annotations

import torch

from nami.generators.ctmc import CTMCGeneratorOperator
from nami.interpolants.protocol import InterpolantState
from nami.parameterizations import GeneratorParams, Target, TensorLike


def _broadcast_alpha(alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # alpha has leading (sample) shape; x is lead + event_shape. Append singleton
    # event dims so the per-coordinate Bernoulli mask broadcasts.
    while alpha.dim() < x.dim():
        alpha = alpha.unsqueeze(-1)
    return alpha


class MaskingInterpolant:
    r"""Absorbing-mask conditional path for :class:`CTMCGeneratorOperator`.

    Args:
        operator (CTMCGeneratorOperator): Operator supplying ``num_states``,
            ``mask_index``, and the masking schedule ``\alpha(t)``.
    """

    def __init__(self, operator: CTMCGeneratorOperator):
        if not isinstance(operator, CTMCGeneratorOperator):
            msg = "MaskingInterpolant requires a CTMCGeneratorOperator"
            raise TypeError(msg)
        self.operator = operator

    def sample(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState:
        r"""Mask ``x_data`` to time ``t``.

        Args:
            x_noise (torch.Tensor): Ignored except for shape; the noise end is
                the all-mask state, supplied implicitly by ``mask_index``.
            x_data (torch.Tensor): Clean integer token indices in
                ``{0, ..., num_states - 1}``, shape ``lead + event_shape``.
            t (torch.Tensor): Times of shape ``lead``.
            noise (torch.Tensor | None): Optional uniform draws in ``[0, 1)``
                with the shape of ``x_data`` for reproducible masking.

        Returns:
            InterpolantState: ``xt`` is the masked token state; ``noise`` carries
            the boolean mask (``True`` where masked) for downstream use.
        """
        _ = x_noise
        alpha_t = _broadcast_alpha(self.operator.alpha(t).to(x_data.device), x_data)
        draw = torch.rand_like(alpha_t.expand_as(x_data)) if noise is None else noise
        masked = draw < alpha_t
        xt = torch.where(
            masked, torch.full_like(x_data, self.operator.mask_index), x_data
        )
        return InterpolantState(
            xt=xt, x_noise=x_noise, x_data=x_data, t=t, noise=masked
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        r"""Conditional generator target ``F_t^z``: the clean-token distribution.

        For :class:`~nami.parameterizations.GeneratorParams` this is the one-hot
        encoding of the clean tokens over the ``num_states`` data tokens — the
        denoiser target the KL/cross-entropy CGM loss regresses against.
        """
        match target:
            case GeneratorParams(operator=op):
                _ = op
                return torch.nn.functional.one_hot(
                    state.x_data.long(), num_classes=self.operator.num_states
                ).to(torch.get_default_dtype())
            case _:
                msg = (
                    "MaskingInterpolant supports only the GeneratorParams target "
                    f"(masking CTMC); got {type(target).__name__}."
                )
                raise NotImplementedError(msg)
