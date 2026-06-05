r"""Masking Continuous-Time Markov Chain (CTMC) generator.

A discrete-state-space instance of Generator Matching: a pure-jump
generator on a finite state space that unmasks tokens. It is the main
setting in which the Conditional Generator Matching loss is not a squared-
``L_2`` regression, because the conditional generator's tractable parameterisation is a
categorical denoiser distribution on the probability simplex, matched with the
**KL / cross-entropy Bregman divergence** (:class:`~nami.losses.bregman.KLDivergence`).

Construction (absorbing-mask diffusion)
---------------------------------------
The state space has ``num_states`` data tokens ``{0, ..., K-1}`` plus one
absorbing ``MASK`` token at index ``K``. The conditional probability path for a
coordinate with clean value ``z`` is

.. math::

   p_t(x \mid z) = \alpha(t)\,\delta_{\mathrm{MASK}}(x) + (1 - \alpha(t))\,\delta_z(x),

with masking fraction ``\alpha(0) = 1`` (fully masked at the noise end) and
``\alpha(1) = 0`` (fully revealed at the data end); the default schedule is the
linear ``\alpha(t) = 1 - t``. Sampling runs forward in ``t`` by unmasking: a
masked coordinate is revealed in ``[t, t+dt]`` with probability
``(\alpha(t) - \alpha(t+dt)) / \alpha(t)`` and, when revealed, draws its token
from the network's denoiser posterior ``p_\theta(z \mid x_t)``.

The network emits per-coordinate logits over the ``K`` data "tokens"; the
operator's :meth:`project` maps them to the simplex (softmax) so that every
downstream method (the CGM loss, :meth:`decompose`, and :meth:`jump_step`)
sees a categorical distribution.

In the masking limit this recovers masked-diffusion language models; see the
schedule-conditioned-discrete-diffusion concept note for the broader CTMC
family.

References
----------
- Holderrieth et al., *Generator Matching*, 2024 (§7.2, discrete state space).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nami.core.specs import TensorSpec
from nami.fields._common import normalise_event_shape
from nami.generators.base import GeneratorOperator

if TYPE_CHECKING:
    from nami.losses.bregman import BregmanDivergence


class CTMCGeneratorOperator(GeneratorOperator):
    r"""Masking CTMC (pure-jump) generator over a finite vocabulary.

    Args:
        num_states (int): Number of data tokens ``K``. The absorbing ``MASK``
            token occupies index ``K``, so the full vocabulary size is
            ``K + 1``.
        event_shape (int or tuple[int, ...]): Shape of a single event; the
            trailing axis indexes the (independent) token coordinates.
        eps (float): Floor used when dividing by the masking fraction
            ``\alpha(t)`` near the data endpoint.
    """

    def __init__(
        self,
        num_states: int,
        event_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-6,
    ):
        if num_states < 2:
            msg = f"num_states must be at least 2, got {num_states}"
            raise ValueError(msg)
        self.num_states = int(num_states)
        self._spec = TensorSpec(normalise_event_shape(event_shape))
        self.eps = float(eps)
        super().__init__(runtime_kind="jump")

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._spec.event_shape

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        # one categorical distribution over the K data tokens per coordinate.
        return (*self.event_shape, self.num_states)

    @property
    def mask_index(self) -> int:
        """Vocabulary index of the absorbing ``MASK`` token (``= num_states``)."""
        return self.num_states

    @property
    def vocab_size(self) -> int:
        """Full vocabulary size including the mask token (``num_states + 1``)."""
        return self.num_states + 1

    def alpha(self, t: torch.Tensor | float) -> torch.Tensor | float:
        r"""Masking fraction ``\alpha(t)`` (linear schedule ``1 - t``)."""
        return 1.0 - t

    def project(self, params: torch.Tensor) -> torch.Tensor:
        """Map raw logits to the probability simplex over the data tokens."""
        self.validate_params(params)
        return torch.softmax(params, dim=-1)

    def decompose(self, params: torch.Tensor) -> dict[str, torch.Tensor]:
        # ``params`` is already a categorical distribution (see ``project``);
        # exposed as a single "rates" component matched with KL.
        return {"rates": params}

    def default_divergence(self) -> dict[str, BregmanDivergence]:
        # Deferred import: losses -> interpolants -> masking -> generators.ctmc
        # would cycle back here at module-import time.
        from nami.losses.bregman import KLDivergence  # noqa: PLC0415

        return {"rates": KLDivergence(dim=-1)}

    def jump_step(
        self,
        x: torch.Tensor,
        t: float,
        dt: float,
        params: torch.Tensor,
    ) -> torch.Tensor:
        r"""Advance the masked state one unmasking step.

        Args:
            x (torch.Tensor): Current integer token state, shape
                ``lead + event_shape``. Masked coordinates hold
                :attr:`mask_index`.
            t (float): Current time.
            dt (float): Step size.
            params (torch.Tensor): Denoiser posterior ``p_\theta(z \mid x_t)``
                over the data tokens (already on the simplex), shape
                ``lead + event_shape + (num_states,)``.

        Returns:
            torch.Tensor: Next integer token state. Coordinates revealed this
            step are drawn from ``params``; already-revealed coordinates are
            absorbing and unchanged.
        """
        alpha_t = float(self.alpha(t))
        alpha_next = float(self.alpha(t + dt))
        # fraction of still-masked coordinates revealed over [t, t+dt].
        denom = max(alpha_t, self.eps)
        p_unmask = (alpha_t - alpha_next) / denom
        p_unmask = min(max(p_unmask, 0.0), 1.0)

        masked = x == self.mask_index
        draw = torch.rand(x.shape, device=x.device)
        do_unmask = masked & (draw < p_unmask)

        sampled = torch.distributions.Categorical(probs=params).sample()
        return torch.where(do_unmask, sampled.to(x.dtype), x)
