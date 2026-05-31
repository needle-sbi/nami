r"""Conditional Generator Matching (CGM) loss.

The CGM loss is the tractable training target of Generator Matching
(Holderrieth et al., 2024, Eq. 17):

.. math::

   L_{\mathrm{cgm}}(\theta) = \mathbb{E}_{t,\,z\sim p_{\mathrm{data}},\,x\sim p_t(\cdot|z)}
   \Bigl[ D\bigl(F_t^z(x),\, F_t^\theta(x)\bigr) \Bigr],

where ``F_t^z`` is a tractable linear parameterisation of the *conditional*
generator (drift for flows, diffusion coefficient, jump rates, ...), ``F_t^\theta``
is the network's emission, and ``D`` is a Bregman divergence
(:mod:`nami.losses.bregman`). Proposition 2 establishes
``\nabla_\theta L_{\mathrm{gm}} = \nabla_\theta L_{\mathrm{cgm}}``, and requires ``D``
to be Bregman for that identity to hold.

This differs from :func:`~nami.losses.regression.regression_loss` in two ways
that matter for non-Euclidean generators:

#. The discrepancy is a caller-chosen Bregman divergence rather than a hardcoded
   MSE, so jump / CTMC rates can be matched with the KL / cross-entropy
   divergence on the simplex (the domain where MSE silently breaks the
   gradient identity).
#. The divergence is applied per generator component; drift, diffusion,
   jump rates each on their own convex domain, and summed. Sum-of-Bregman is
   itself Bregman, so composed / multimodal generators inherit the gradient
   identity (the Markov-superposition composition, prop. 7 of the paper).

When ``divergence`` is left at ``None``, each component uses the operator's
:meth:`~nami.generators.base.GeneratorOperator.default_divergence`; for an Itô
operator that is squared-``L_2`` throughout. For a drift-only (ODE) Itô operator
there is a single component and the default CGM loss coincides exactly with
``regression_loss``. With a diffusion component the CGM loss sums the
per-component MSEs, which differs from ``regression_loss``'s single MSE over
the whole packed tensor by the component-averaging convention.
"""

from __future__ import annotations

import torch

from nami.interpolants.protocol import Interpolant
from nami.losses._common import (
    leading_shape,
    reduce_loss,
    require_event_ndim,
    sample_t,
)
from nami.losses.bregman import BregmanDivergence
from nami.parameterizations import GeneratorParams, Parameterization


def cgm_loss(
    field,
    *,
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    interpolant: Interpolant,
    parameterization: Parameterization,
    divergence: BregmanDivergence | dict[str, BregmanDivergence] | None = None,
    eps_t: float = 1e-3,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Conditional Generator Matching loss with a per-component Bregman divergence.

    Parameters
    ----------
    field
        Network emitting raw operator parameters. Must expose ``event_ndim``.
    x_noise, x_data
        Endpoints of the conditional path (noise at ``t=0``, data at ``t=1``).
    t
        Optional pre-sampled times of shape matching the leading dims of
        ``x_data``. When ``None``, drawn from ``U[eps_t, 1 - eps_t]``.
    c
        Optional context.
    interpolant
        Path object implementing the
        :class:`~nami.interpolants.protocol.Interpolant` protocol. Its
        ``target(GeneratorParams(operator), state)`` arm supplies the packed
        conditional generator ``F_t^z``.
    parameterization
        Bundle of (target, weighting, output_transform). ``target`` **must** be
        :class:`~nami.parameterizations.GeneratorParams`; the CGM loss is
        defined against a generator parameterisation.
    divergence
        Bregman divergence ``D`` used to match ``F_t^z`` against ``F_t^\theta``.
        Either a single :class:`~nami.losses.bregman.BregmanDivergence` applied
        to every component, or a mapping keyed by the operator's component names
        (see :meth:`~nami.generators.base.GeneratorOperator.decompose`). When
        ``None``, the operator's
        :meth:`~nami.generators.base.GeneratorOperator.default_divergence` is
        used.
    eps_t
        Minimum distance from ``{0, 1}`` for auto-sampled ``t``. Pass ``0.0`` to
        disable clamping. Ignored when ``t`` is supplied.
    z
        Optional latent noise forwarded to ``interpolant.sample(..., noise=z)``.
    reduction
        ``"mean"`` | ``"sum"`` | ``"none"``.

    Returns
    -------
    torch.Tensor
        The reduced CGM loss.

    Raises
    ------
    TypeError
        If ``parameterization.target`` is not ``GeneratorParams``.
    KeyError
        If ``divergence`` is a mapping missing a component name.
    """
    target = parameterization.target
    if not isinstance(target, GeneratorParams):
        msg = (
            "cgm_loss requires a GeneratorParams target; got "
            f"{type(target).__name__}. Use generator_prediction(operator) to "
            "build the parameterization, or regression_loss for plain "
            "velocity/score/eps targets."
        )
        raise TypeError(msg)
    operator = target.operator

    event_ndim = require_event_ndim(field)
    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t)

    state = interpolant.sample(x_noise, x_data, t, noise=z)
    target_value = interpolant.target(target, state)

    raw = field(state.xt, t, c)
    prediction = parameterization.output_transform(raw)

    pred_parts = operator.decompose(prediction)
    target_parts = operator.decompose(target_value)

    divergence = operator.default_divergence() if divergence is None else divergence

    weight = parameterization.weighting(t)

    # TODO: per-position masking hook. Faithful masked-diffusion CGM restricts
    # the per-component divergence (and an alpha(t)-dependent weighting) to the
    # masked positions only, rather than summing over all coordinates. A hook
    # here (e.g. a position mask derived from `state`) would let the masking
    # CTMC supervise only masked tokens without coupling the loss to the operator.
    per_sample = torch.zeros(lead, device=x_data.device, dtype=prediction.dtype)
    for name, pred_part in pred_parts.items():
        d = divergence[name] if isinstance(divergence, dict) else divergence
        per_sample = per_sample + d(pred_part, target_parts[name], lead)

    if weight.shape != per_sample.shape:
        weight = weight.expand_as(per_sample)

    return reduce_loss(weight * per_sample, reduction)
