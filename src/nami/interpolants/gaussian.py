r"""Gaussian-noise interpolant for diffusion-style transport.

Implements ``x_t = \alpha(t) \cdot x_noise + \sigma(t) \cdot x_data`` against any
:class:`~nami.schedules.base.NoiseSchedule`, plus the canonical
:class:`~nami.parameterizations.Parameterization` factories that bind a
target choice to its conventional weighting ``\omega(t)``.

The factories (:func:`epsilon_prediction`, :func:`score_prediction`,
:func:`x0_prediction`) replace the ``parameterization="eps"|"score"|"x0"``
string flag on the legacy ``Diffusion`` process.  Each factory's default
weighting is the one that makes its loss numerically equivalent to the
DDPM-uniform ``\epsilon``-prediction loss under change of variables - silent
re-weighting bugs (the Arruda Eq. 57-61 case) become structurally
impossible because changing target without changing ``\omega`` is changing the
``Parameterization``, not toggling a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch

from nami.diffusion import expand_like
from nami.interpolants.protocol import InterpolantState
from nami.parameterizations import (
    X0,
    Action,
    Epsilon,
    GeneratorParams,
    Parameterization,
    Score,
    Target,
    TensorLike,
    Velocity,
    VPrediction,
)
from nami.schedules.base import NoiseSchedule


@dataclass(frozen=True)
class GaussianInterpolant:
    r"""Gaussian interpolant ``x_t = \alpha(t) x_noise + \sigma(t) x_data``.

    In the FM convention used here, ``\alpha(t)`` is the noise coefficient
    (``\alpha(0)=1, \alpha(1)=0``) and ``\sigma(t)`` is the data coefficient
    (``\sigma(0)=0, \sigma(1)=1``).  The :class:`NoiseSchedule` numerical
    contracts are unchanged — the schedule's ``alpha(t)`` plays the
    noise-coefficient role, and ``sigma(t)`` plays the data-coefficient role.

    The source endpoint ``x_noise`` plays the role of the latent ``\epsilon``;
    the ``noise`` slot on :class:`InterpolantState` is therefore left ``None``.
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
       ``\alpha(t)`` (the noise level), which is zero at ``t=1`` for
       VP-style schedules — the data endpoint in the FM convention. This
       interpolant deliberately does **not** clamp ``\alpha`` internally —
       silent clamps hide numerical bugs.  Callers must constrain
       ``t`` to a non-degenerate interval; the stage-1b
       ``regression_loss`` does so by default by sampling ``t`` from
       ``[eps_t, 1 - eps_t]``.  Direct evaluation at ``t=1`` will return
       ``inf`` / ``nan`` and that behaviour is pinned by a regression
       test.
    """

    # The single field is a NoiseSchedule (e.g. nami.VPSchedule(),
    # nami.VESchedule(), nami.EDMSchedule()) — not a tensor dim. A common
    # first-use trap is ``GaussianInterpolant(4)``; __post_init__ catches
    # that with an actionable message instead of failing later inside
    # ``sample`` with ``'int' object has no attribute 'alpha'``.
    schedule: NoiseSchedule

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, NoiseSchedule):
            msg = (
                f"GaussianInterpolant(schedule=...) expects a NoiseSchedule "
                f"instance (e.g. nami.VPSchedule()), got {type(self.schedule).__name__}. "
                f"If you want a velocity-style flow-matching interpolant with "
                f"no schedule, use nami.LinearInterpolant() instead."
            )
            raise TypeError(msg)

    def sample(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> InterpolantState:
        # ``noise=`` is rejected: for a Gaussian interpolant the "noise
        # variable" *is* x_noise (the source endpoint plays the role of
        # epsilon). Accepting a separate noise tensor would create two
        # parallel conventions for the same quantity.
        if noise is not None:
            msg = (
                "GaussianInterpolant uses x_noise as the noise variable; "
                "pass the noise sample as x_noise rather than via the "
                "noise= keyword."
            )
            raise ValueError(msg)
        a = expand_like(self.schedule.alpha(t), x_data)
        s = expand_like(self.schedule.sigma(t), x_data)
        xt = a * x_noise + s * x_data
        return InterpolantState(
            xt=xt,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            noise=None,
        )

    def target(self, target: Target, state: InterpolantState) -> TensorLike:
        match target:
            case Epsilon():
                return state.x_noise
            case X0():
                return state.x_data
            case Score():
                # In the FM convention, x_t = alpha(t) * x_noise + sigma(t) * x_data,
                # so the noise level (coefficient of x_noise) is alpha(t).
                # Score = -noise_var / noise_level = -x_noise / alpha(t).
                a = expand_like(self.schedule.alpha(state.t), state.x_noise)
                return -state.x_noise / a
            case Velocity():
                # Velocity needs schedule derivatives (alpha'(t), sigma'(t))
                # which the base NoiseSchedule contract does not expose. For
                # flow-matching with a Velocity target, use LinearInterpolant
                # (closed-form constant velocity = x_noise - x_data) or
                # CosineInterpolant (closed-form alpha'/sigma').
                msg = (
                    "GaussianInterpolant does not implement Velocity — "
                    "schedule derivatives \\alpha'(t), \\sigma'(t) are required. "
                    "Use nami.LinearInterpolant() (with velocity_prediction()) "
                    "or nami.CosineInterpolant() for a Velocity target."
                )
                raise NotImplementedError(msg)
            case VPrediction():
                # Salimans-Ho v-target: ``v = \alpha(t) \epsilon - \sigma(t) x_0``,
                # where in the FM convention here ``\alpha`` is the noise
                # coefficient (``schedule.alpha``) and ``\sigma`` is the data
                # coefficient (``schedule.sigma``).  ``\epsilon = state.x_noise``,
                # ``x_0 = state.x_data``.  The code form is identical to the
                # diffusion-convention version because the operand-swap from
                # the convention flip and the relabelling of \alpha/\sigma cancel.
                a = expand_like(self.schedule.alpha(state.t), state.x_noise)
                s = expand_like(self.schedule.sigma(state.t), state.x_data)
                return a * state.x_noise - s * state.x_data
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
    r"""``\epsilon``-prediction with ``\omega(t)=1`` — the DDPM-standard convention.

    The network emits the standardised noise directly; the loss is plain
    MSE.  Equivalent (via change of variables) to ``score`` and ``x0``
    prediction with their conventional weightings.
    """
    del schedule  # unused — epsilon-prediction's omega is schedule-independent

    def weighting(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    return Parameterization(target=Epsilon(), weighting=weighting)


def score_prediction(schedule: NoiseSchedule) -> Parameterization:
    r"""Score-matching with ``\omega(t)=\alpha^2(t)``.

    In the FM convention, ``score = -\epsilon / \alpha(t)`` (the noise level
    is the coefficient of ``x_noise``, which is ``\alpha(t)``). Therefore
    ``\omega(t)=\alpha^2(t)`` is the change-of-variable weighting that makes
    this loss numerically equal to DDPM-uniform ``\epsilon``-prediction —
    the role ``\sigma^2(t)`` played in the diffusion convention.

    Note: the score target itself is singular at ``t=1`` for VP-style
    schedules (``\alpha(1) \to 0``). ``\omega(t)=\alpha^2(t)`` tames the *weighted*
    loss but the unweighted target value is still very large (or
    ``inf`` / ``nan``), which propagates through autograd.  Callers must
    restrict ``t`` to ``[0, 1)``; stage 1b's ``regression_loss`` does so
    by default.
    """

    def weighting(t: torch.Tensor) -> torch.Tensor:
        return schedule.alpha(t).pow(2)

    return Parameterization(target=Score(), weighting=weighting)


def v_prediction(schedule: NoiseSchedule) -> Parameterization:
    r"""Salimans-Ho v-prediction with ``\omega(t)=1``.

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
    r"""``x_0``-prediction with ``\omega(t) = \sigma^2(t)/\alpha^2(t) = 1/\mathrm{SNR}(t)``.

    In the FM convention, ``x_0 = (x_t - \alpha(t)\epsilon)/\sigma(t)``,
    so ``x_0 - x_0_{\text{pred}} = (\sigma/\alpha)(\epsilon_{\text{pred}} - \epsilon)``
    and the change-of-variable weighting that makes the ``x_0`` loss
    equivalent to DDPM-uniform ``\epsilon``-prediction is
    ``(\sigma/\alpha)^2 = 1/\mathrm{SNR}(t)``.  This is the inverse of the diffusion-convention
    weighting and reflects the t-direction flip.

    Note: this weighting diverges as ``t \to 1`` for VP-style schedules
    (``\alpha(t) \to 0``). Callers must restrict ``t`` to ``[0, 1)``;
    stage 1b's ``regression_loss`` enforces this by default.
    """

    def weighting(t: torch.Tensor) -> torch.Tensor:
        return 1.0 / schedule.snr(t)

    return Parameterization(target=X0(), weighting=weighting)
