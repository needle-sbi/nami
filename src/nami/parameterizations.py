r"""Training-target language for transport processes.

* :data:`Target` is a sum type over the named mathematical objects a network
  can learn to emit (velocity, score, epsilon, x0, generator parameters).
  New target families are added as new variants, and pattern-match dispatch
  ensures every consumer is forced to handle them.
* :class:`Parameterization` bundles a :data:`Target` with the implicit
  weighting ``\omega(t)`` and the ``output_transform`` that maps a raw network
  emission into the target's space.  Carrying the weighting alongside the
  target choice closes the silent re-weighting bug class that string-flag
  parameterisations (``"eps" | "score" | "x0"``) historically permitted.
* The dispatch is a method on the :class:`~nami.interpolants.protocol.Interpolant`,
  not a free function: each interpolant knows which targets it can express and
  raises ``NotImplementedError`` for those it cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .generators.base import GeneratorOperator


# ---------------------------------------------------------------------------
# Note:
# ---------------------------------------------------------------------------
# Every variant is a frozen dataclass so it can carry per-target metadata
# (e.g. the GeneratorOperator that interprets GeneratorParams) without
# leaking into string-keyed registries.  Frozen dataclasses also receive
# ``__match_args__`` for free, which makes ``case Variant(field=x):``
# pattern matching work at call sites.


@dataclass(frozen=True)
class Velocity:
    """Conditional velocity ``u_t(x)`` — the flow-matching target."""


@dataclass(frozen=True)
class Score:
    r"""Stein score ``\nabla \log p_t(x)`` — diffusion / bridge-matching target."""


@dataclass(frozen=True)
class Epsilon:
    r"""Standardised noise ``\epsilon`` such that ``x_t = \alpha_t \epsilon + \sigma_t x_0`` (FM convention)."""


@dataclass(frozen=True)
class X0:
    r"""Clean data endpoint ``x_0`` — the denoising / x0-prediction target."""


@dataclass(frozen=True)
class GeneratorParams:
    r"""Packed generator parameters interpreted by an operator.

    The structural difference between scalar targets (a single tensor) and
    generator targets (drift + diffusion + jump rate, …) is encoded by the
    operator's ``pack_params`` / ``project`` rather than in the type system:
    the network still emits a single tensor in the operator's parameter
    space, and the operator is what makes that tensor mean ``(drift,
    diffusion)``.
    """

    operator: GeneratorOperator


@dataclass(frozen=True)
class VPrediction:
    r"""Salimans-Ho v-prediction target.

    The network emits ``v = \alpha(t) \epsilon - \sigma(t) x_0`` where ``\alpha``, ``\sigma`` come
    from a :class:`~nami.schedules.base.NoiseSchedule`.  v-prediction
    is well-defined whenever the interpolant has a Gaussian noise
    decomposition with a schedule available; for nami today that
    means :class:`~nami.interpolants.gaussian.GaussianInterpolant`.
    Other interpolants raise ``NotImplementedError`` (linear paths
    have no schedule; the Brownian bridge's noise variable is the
    bridge increment, not a standardised ``\epsilon``).

    The :func:`~nami.interpolants.gaussian.v_prediction` factory binds
    this target with the conventional uniform weighting ``\omega(t)=1``; for
    the SNR-shifted weighting (Salimans & Ho's "v-loss"), pass a
    custom ``weighting`` to :class:`Parameterization`.
    """


@dataclass(frozen=True)
class Action:
    r"""Action-matching target (Neklyudov et al., 2023).

    The network emits a **scalar potential** ``s(x, t)`` whose gradient
    is the conditional velocity: ``\nabla_x s(x, t) = u_t(x)``.  Structurally
    distinct from the tensor-valued targets above — the field's output
    shape is ``(*lead,)``, not ``(*lead, *event)`` — so the corresponding
    loss is *not* a tensor-vs-tensor regression but a regression of
    ``\nabla_x s`` against the interpolant's velocity, with autograd
    plumbing analogous to ``log_density_consistency_loss``.

    The variant exists so every :class:`Interpolant` is forced to declare
    whether it can express the action target; the matching
    ``action_matching_loss`` and ``ActionMatching`` Process land in a
    follow-up MR.  Until then every interpolant's ``target`` arm raises
    :class:`NotImplementedError` pointing at that work.
    """


Target = Velocity | Score | Epsilon | X0 | VPrediction | Action | GeneratorParams
"""Sum type over named training targets.

Adding a new target family means:

1. Adding a new frozen-dataclass variant in this module.
2. Extending the :data:`Target` union to include it.
3. Adding a ``case`` arm in every consumer's pattern match.

Step 3 is what prevents a parallel hierarchy from re-emerging: a missing
arm becomes a ``TypeError`` at call time, and static checkers flag the
non-exhaustive match.
"""


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

TensorLike = torch.Tensor
"""What a target value may be.

Today every concrete target reduces to a single tensor (generator targets
are packed by the operator).  If future targets need structured outputs,
that should be introduced as a separate generic path rather than widening
the common tensor-valued regression and process APIs.
"""


def _ones_weighting(t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(t)


def _identity_transform(y: TensorLike) -> TensorLike:
    return y


@dataclass(frozen=True)
class Parameterization:
    r"""Bundle of (target, weighting, output_transform).

    ``Parameterization`` is **dual-role**: the same instance is consumed by
    the training loss (``regression_loss``) and by the runtime ``Process``
    (sampling, log-density, drift extraction).  What it carries directly
    is the *target choice*, the *weighting* ``\omega(t)``, and a narrow
    projection ``output_transform`` — **not** cross-target conversion math.
    Runtime conversions (e.g. ``\epsilon`` to score for drift extraction from an
    ``\epsilon``-trained model) are the ``Process`` layer's job, dispatched
    exhaustively on ``parameterization.target`` using the algebraic
    helpers in :mod:`nami.diffusion`.

    Together, ``Parameterization`` plus that ``Process``-layer dispatch
    subsume the legacy runtime adapters (``ScoreFromRawNoise``,
    ``DriftFromVelocityScore``, ``MarkovizationDriftFromVelocityScore``,
    ``MirrorVelocityFromScore``, ``ScoreFromEta``) in
    ``nami.interpolants.transforms``.  Those wrappers are not subsumed by
    this class alone — they are subsumed by the architecture of
    ``Parameterization`` (target choice + weighting) plus the Process's
    sum-type dispatch.  All five wrappers are scheduled for removal in
    stage 4 of the refactor.

    Parameters
    ----------
    target
        Which mathematical object the network learns to emit.
    weighting
        ``\omega(t)`` applied to the per-sample loss.  Travels with the target
        choice so that switching parameterisations cannot silently change
        the effective objective.  At runtime the weighting is ignored —
        only the loss consumes it.
    output_transform
        Projects the raw network output into ``target``-space.  Pure
        function of the network output — does **not** receive ``t``,
        ``state``, or schedule; intentionally narrow.  Identity for
        scalar targets predicted directly; ``operator.project`` for
        :class:`GeneratorParams` (constraining e.g. positive diffusion).

        **Cross-target runtime conversions** (e.g. ``\epsilon`` to score for drift
        extraction from an ``\epsilon``-trained model) do *not* live here, because
        those conversions need ``t``, ``schedule``, and sometimes
        ``x_t`` — widening this signature would force every loss call
        to thread runtime context it does not need.  Such conversions
        are the :class:`~nami.processes` layer's job: a ``Process``
        consumes ``parameterization.target`` and pattern-matches on the
        variant to dispatch to the right algebraic identity (e.g.
        :func:`~nami.diffusion.eps_to_score`).  The non-recurrence
        of the legacy ``parameterization="eps"|"score"|"x0"`` string-flag
        bug comes from sum-type exhaustiveness over :data:`Target` at
        the Process layer, not from ``output_transform`` carrying the
        conversion.
    """

    target: Target
    weighting: Callable[[torch.Tensor], torch.Tensor] = field(default=_ones_weighting)
    output_transform: Callable[[TensorLike], TensorLike] = field(
        default=_identity_transform
    )

    @property
    def is_identity_transform(self) -> bool:
        r"""True when ``output_transform`` is the no-op projection.

        Lets ``Process`` layers gate paths that are only valid for the
        identity case (e.g. a field's bundled ``call_and_divergence``,
        whose divergence is taken of the *raw* output) without reaching
        across module boundaries for the private sentinel.
        """
        return self.output_transform is _identity_transform
