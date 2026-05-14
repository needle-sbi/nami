from __future__ import annotations

"""Gaussian-noise interpolant for diffusion-style transport.

Implements ``x_t = \alpha(t) \cdot x_target + \sigma(t) \cdot x_source`` against any
:class:`~nami.schedules.base.NoiseSchedule`, plus the canonical
:class:`~nami.parameterizations.Parameterization` factories that bind a
target choice to its conventional weighting ``\omega(t)``.

The factories (:func:`epsilon_prediction`, :func:`score_prediction`,
:func:`x0_prediction`) replace the ``parameterization="eps"|"score"|"x0"``
string flag on the legacy ``Diffusion`` process.  Each factory's default
weighting is the one that makes its loss numerically equivalent to the
DDPM-uniform ``\epsilon``-prediction loss under change of variables — silent
re-weighting bugs (the Arruda Eq. 57–61 case) become structurally
impossible because changing target without changing ``\omega`` is changing the
``Parameterization``, not toggling a flag.
"""


from dataclasses import dataclass
from typing import assert_never

import torch

from nami.diffusion import expand_like
from nami.parameterizations import (
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Target,
    TensorLike,
    VPrediction,
    Velocity,
    X0,
)
from nami.schedules.base import NoiseSchedule
from nami.interpolants.protocol import InterpolantState


@dataclass(frozen=True)
class GaussianInterpolant:
    """Gaussian interpolant ``x_t = \alpha(t) x_target + \sigma(t) x_source``.

    The source endpoint plays the role of the latent noise ``\epsilon``; the
    ``noise`` slot on :class:`InterpolantState` is therefore left ``None``.
    Passing a separate ``noise`` argument to :meth:`sample` is rejected so
    the convention stays unambiguous.

    Supported targets: :class:`~nami.parameterizations.Epsilon`,
    :class:`~nami.parameterizations.X0`, :class:`~nami.parameterizations.Score`.
    :class:`~nami.parameterizations.Velocity` raises ``NotImplementedError``
    because a Gaussian conditional path's velocity requires schedule
    derivatives ``\alpha'(t), \sigma'(t)`` which the base ``NoiseSchedule`` does not
    expose.  :class:`~nami.parameterizations.GeneratorParams` is the wrong
    shape for a Gaussian path — those targets belong on operator-aware
    interpolants.

    .. warning::

       The :class:`~nami.parameterizations.Score` target divides by
       ``\sigma(t)``, which is zero at ``t=0`` for VP-style schedules. This
       interpolant deliberately does **not** clamp ``\sigma`` internally —
       silent clamps hide numerical bugs.  Callers must constrain
       ``t`` to a non-degenerate interval; the stage-1b
       ``regression_loss`` does so by default by sampling ``t`` from
       ``[eps_t, 1 - eps_t]``.  Direct evaluation at ``t=0`` will return
       ``inf`` / ``nan`` and that behaviour is pinned by a regression
       test.
    """

    schedule: NoiseSchedule

    def sample(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState:
        if noise is not None:
            msg = (
                "GaussianInterpolant uses x_source as the noise variable; "
                "pass the noise sample as x_source rather than via the "
                "noise= keyword."
            )
            raise ValueError(msg)
        a = expand_like(self.schedule.alpha(t), x_target)
        s = expand_like(self.schedule.sigma(t), x_target)
        xt = a * x_target + s * x_source
        return InterpolantState(
            xt=xt,
            x_target=x_target,
            x_source=x_source,
            t=t,
            noise=None,
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Epsilon():
                return state.x_source
            case X0():
                return state.x_target
            case Score():
                s = expand_like(self.schedule.sigma(state.t), state.x_source)
                return -state.x_source / s
            case Velocity():
                msg = (
                    "GaussianInterpolant does not implement Velocity — "
                    "schedule derivatives \\alpha'(t), \\sigma'(t) are required."
                )
                raise NotImplementedError(msg)
            case VPrediction():
                # Salimans-Ho v-target: ``v = \alpha(t) \epsilon - \sigma(t) x_0``.
                # Here ``\epsilon = state.x_source`` (the noise variable) and
                # ``x_0 = state.x_target`` by nami's convention.
                a = expand_like(self.schedule.alpha(state.t), state.x_source)
                s = expand_like(self.schedule.sigma(state.t), state.x_target)
                return a * state.x_source - s * state.x_target
            case Action():
                # Action's target is the conditional velocity (against
                # which ``\nabla_x s`` is regressed). GaussianInterpolant cannot
                # express that velocity without ``\alpha'(t), \sigma'(t)``, which the
                # base NoiseSchedule does not expose — same reason
                # Velocity raises here.  Use LinearInterpolant,
                # CosineInterpolant, StochasticLinearInterpolant, or
                # BrownianBridgeInterpolant for action matching.
                msg = (
                    "GaussianInterpolant does not support Action: the action "
                    "target is the conditional velocity \\nabla_x s should match, "
                    "and computing it requires schedule derivatives \\alpha'(t), "
                    "\\sigma'(t) which NoiseSchedule does not expose."
                )
                raise NotImplementedError(msg)
            case GeneratorParams():
                msg = (
                    "GaussianInterpolant does not implement GeneratorParams; "
                    "use an operator-aware interpolant instead."
                )
                raise NotImplementedError(msg)
        # See LinearInterpolant.target for the assert_never rationale.
        assert_never(target)


# ---------------------------------------------------------------------------
# Parameterization factories
# ---------------------------------------------------------------------------
# Each factory binds a target with the weighting ``\omega(t)`` that makes the
# resulting loss DDPM-equivalent (i.e. equal to ``\epsilon``-prediction with ``\omega=1``
# under change of variables).  Override `weighting` for min-SNR, EDM,
# or other non-equivalent schemes; the override is the *deliberate*
# reweighting the factory was designed to make explicit.


def epsilon_prediction(schedule: NoiseSchedule) -> Parameterization:
    """``\epsilon``-prediction with ``\omega(t)=1`` — the DDPM-standard convention.

    The network emits the standardised noise directly; the loss is plain
    MSE.  Equivalent (via change of variables) to ``score`` and ``x0``
    prediction with their conventional weightings.
    """
    del schedule  # unused — epsilon-prediction's omega is schedule-independent
    return Parameterization(
        target=Epsilon(),
        weighting=lambda t: torch.ones_like(t),
    )


def score_prediction(schedule: NoiseSchedule) -> Parameterization:
    """Score-matching with ``\omega(t)=\sigma^2(t)``.

    The ``\sigma^2`` weighting is the maximum-likelihood weighting (Song et al.,
    *Maximum Likelihood Training of Score-Based Diffusion Models*, 2021)
    and the value that makes this loss numerically equal to
    DDPM-uniform ``\epsilon``-prediction under reparameterisation
    ``score = -\epsilon / \sigma(t)``.

    Note: the score target itself is singular at ``t=0`` for VP-style
    schedules (division by ``\sigma(t)=0``). ``\omega(t)=\sigma^2(t)`` tames the *weighted*
    loss but the unweighted target value is still ``inf`` / ``nan``,
    which propagates through autograd.  Callers must restrict ``t`` to
    ``(0, 1]``; stage 1b's ``regression_loss`` does so by default.
    """

    def weighting(t: torch.Tensor) -> torch.Tensor:
        return schedule.sigma(t).pow(2)

    return Parameterization(target=Score(), weighting=weighting)


def v_prediction(schedule: NoiseSchedule) -> Parameterization:
    """Salimans-Ho v-prediction with ``\omega(t)=1``.

    The network emits ``v = \alpha(t) \epsilon - \sigma(t) x_0``. The default uniform
    weighting matches the ``\epsilon``-prediction equivalence path under the
    same change-of-variable argument that ``\epsilon`` / score / ``x_0``
    parameterisations share — but most published v-prediction recipes
    apply an SNR-shifted weighting at training time.  Pass a custom
    ``weighting`` to :class:`Parameterization` for those variants;
    this factory's job is to bind the *target* and a sensible default.

    Only :class:`~nami.interpolants.gaussian.GaussianInterpolant`
    implements the VPrediction target today.  Linear and Brownian-bridge
    interpolants raise ``NotImplementedError`` for VPrediction — see their
    respective ``target`` methods for the rationale.
    """
    del schedule  # omega=1 is schedule-independent at this default.
    return Parameterization(target=VPrediction())


def x0_prediction(schedule: NoiseSchedule) -> Parameterization:
    """``x_0``-prediction with ``\omega(t) = \mathrm{SNR}(t) = \alpha^2(t)/\sigma^2(t)``.

    The SNR weighting is the value that makes the ``x_0`` loss equivalent
    to DDPM-uniform ``\epsilon``-prediction. Override with ``\min(\mathrm{SNR}(t), \gamma)``
    (Hang et al., *Efficient Diffusion Training via Min-SNR Weighting*)
    for the practical training-stability variant.

    Note: ``\mathrm{SNR}(t)`` diverges as ``t \to 0`` for VP-style schedules
    (``\sigma(t) \to 0``). Callers must restrict ``t`` to ``(0, 1]``;
    stage 1b's ``regression_loss`` enforces this by default.
    """

    def weighting(t: torch.Tensor) -> torch.Tensor:
        return schedule.snr(t)

    return Parameterization(target=X0(), weighting=weighting)
