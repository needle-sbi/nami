r"""Action-matching loss.

The network emits a scalar potential :math:`s_\\theta(x, t)` and is
trained so that its spatial gradient matches the interpolant's
conditional velocity:

.. math::

    \\mathcal{L}_{\\mathrm{AM}}(\\theta)
    = \\mathbb{E}_{t, x_t}
        \\bigl\\lVert \\nabla_x s_\\theta(x_t, t) - u_t(x_t) \\bigr\\rVert^2

This is the gradient-regression form of Neklyudov et al., *Action
Matching: Learning Stochastic Dynamics from Samples*, 2023, where the field
is not trained as a vector predictor (it emits a scalar) and the
velocity used at inference time (see
:class:`~nami.processes.action.ActionMatching`) is recovered by autograd.
The conditional velocity targets here come from the stochastic-
interpolant family (Albergo & Vanden-Eijnden, *Stochastic Interpolants*,
2023, arXiv:2303.08797).

Precedent for the autograd plumbing is
:func:`~nami.losses.log_density.log_density_consistency_loss`, which
already uses ``torch.autograd.grad`` against a scalar-out head; this
loss reuses the same shape (sum-the-scalar + ``create_graph=True``).
"""

from __future__ import annotations

import torch

from nami.interpolants.protocol import Interpolant
from nami.losses._common import (
    leading_shape,
    per_sample_mse,
    reduce_loss,
    require_event_ndim,
    sample_t,
)
from nami.parameterizations import Action, Parameterization


def action_prediction() -> Parameterization:
    r"""Action-matching parameterisation with ``\omega(t) = 1``.

    Factory for the :class:`~nami.parameterizations.Action` target that
    pairs with :func:`action_matching_loss`.  No schedule argument: the
    action target is the conditional velocity of whatever
    :class:`~nami.interpolants.protocol.Interpolant` the loss is paired
    with, and that velocity carries its own scaling.  The published
    convention (Neklyudov et al., *Action Matching*, 2023) is uniform
    weighting in ``t``; this factory pins that default.

    Override ``weighting`` on the returned :class:`Parameterization` to
    reweight (e.g. min-SNR-style schedules); the override is the
    deliberate choice the factory was designed to make explicit.
    """
    return Parameterization(target=Action())


def _grad_x_s(
    field,
    xt: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor | None,
    *,
    create_graph: bool,
) -> torch.Tensor:
    r"""Compute ``\nabla_x s(x_t, t, c)`` with ``s = field(...)`` a scalar head.

    ``field(xt, t, c)`` returns a tensor of shape ``(*lead,)``.  Summing
    across the batch lets a single ``autograd.grad`` call recover the
    per-sample gradient — gradients for distinct batch entries do not
    cross because ``s_i`` does not depend on ``x_j`` for ``i != j``.
    """
    xt = xt.detach().requires_grad_(True)
    s = field(xt, t, c)
    # ActionHead returns ``(*lead,)``; a wrong head shape is caught by the
    # grad-vs-velocity shape check in ``action_matching_loss``.
    (grad_s,) = torch.autograd.grad(
        outputs=s.sum(),
        inputs=xt,
        create_graph=create_graph,
    )
    return grad_s


def action_matching_loss(
    field,
    *,
    x_noise: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    interpolant: Interpolant,
    parameterization: Parameterization | None = None,
    eps_t: float = 0.0,
    z: torch.Tensor | None = None,
    reduction: str = "mean",
    create_graph: bool = True,
) -> torch.Tensor:
    r"""Action-matching loss: regress ``\nabla_x s(x_t, t)`` against ``u_t``.

    Implements the gradient-regression objective of Neklyudov et al.,
    *Action Matching*, 2023, on the unified Interpolant +
    Parameterization vocabulary.  The mathematical object is a scalar
    potential whose spatial gradient is the conditional velocity target.

    The ``parameterization`` is expected to carry an
    :class:`~nami.parameterizations.Action` target; if omitted,
    :func:`action_prediction` is used (uniform ``\omega``, identity
    ``output_transform``).  Only the ``weighting`` slot is consumed at
    training time — ``output_transform`` is **ignored** because the
    field's emission is a scalar potential, not a velocity, and there
    is no canonical projection on the scalar (callers wanting to scale
    the predicted potential should compose it inside the head itself).

    Parameters
    ----------
    field
        Scalar-output head with the ``(x, t, c) -> (*lead,)`` shape.
        Must expose ``event_ndim``.  Use
        :class:`~nami.fields.action.ActionHead` for the canonical
        implementation.
    x_noise, x_data
        Endpoints of the conditional path (noise at ``t=0``, data at
        ``t=1`` per nami's FM convention).
    t
        Optional pre-sampled times.  When ``None``, drawn from
        ``U[eps_t, 1 - eps_t]``.
    c
        Optional context forwarded to ``field``.
    interpolant
        :class:`~nami.interpolants.protocol.Interpolant` whose ``target``
        arm for :class:`Action` returns the conditional velocity.
    parameterization
        Must have ``target = Action()``.  ``weighting`` is applied to
        the per-sample MSE.
    eps_t
        Default ``0.0`` because the conditional velocities supplied by
        the linear / cosine / stochastic / bridge interpolants are
        finite at the endpoints (the bridge interpolant guards its own
        ``eps``-floor in ``_velocity``).  Override for schedules that
        introduce endpoint singularities.
    z
        Latent noise forwarded as ``noise=z`` to ``interpolant.sample``.
        Required for ``StochasticLinearInterpolant`` so the bridge
        increment lines up with the velocity correction.
    reduction
        ``"mean"`` | ``"sum"`` | ``"none"``.
    create_graph
        Whether to build a graph for the gradient of ``\nabla_x s`` so that
        ``loss.backward()`` works.  Defaults to ``True`` — required for
        end-to-end training; set ``False`` for eval-only gradient
        evaluation to save memory.
    """
    if parameterization is None:
        parameterization = Parameterization(target=Action())

    if not isinstance(parameterization.target, Action):
        msg = (
            "action_matching_loss requires parameterization.target = Action(); "
            f"got {type(parameterization.target).__name__}.  Use "
            "regression_loss for tensor-valued targets."
        )
        raise TypeError(msg)

    event_ndim = require_event_ndim(field)
    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t)

    state = interpolant.sample(x_noise, x_data, t, noise=z)
    u_t = interpolant.target(parameterization.target, state)

    grad_s = _grad_x_s(field, state.xt, t, c, create_graph=create_graph)

    if grad_s.shape != u_t.shape:
        msg = (
            f"shape mismatch: \\nabla_x s has shape {tuple(grad_s.shape)} but "
            f"interpolant velocity has shape {tuple(u_t.shape)}.  The "
            "field's scalar output (shape (*lead,)) must broadcast to the "
            "event shape under \\nabla_x — check the head's event_ndim against "
            "x_data.ndim."
        )
        raise ValueError(msg)

    mse = per_sample_mse(grad_s, u_t, lead)

    weight = parameterization.weighting(t)
    if weight.shape != mse.shape:
        weight = weight.expand_as(mse)
    weighted = weight * mse

    return reduce_loss(weighted, reduction)
