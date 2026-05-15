Parameterizations
=================

A field in nami predicts *something*, but there is more than one useful
choice of what that something is. The same transport map can be specified by
its velocity, by the score of its intermediate marginals, by the raw
Gaussian noise added to an interpolant, by the clean-data target, or by a
vector of operator parameters that an explicit operator object later
interprets. These are not different methods — they are different
parameterisations of one object — and nami lets you switch among them by
changing what the field predicts and which loss you regress it with.

Why several parameterisations exist
-----------------------------------

Each parameterisation makes a different quantity easy. A velocity field is
the natural thing to integrate when you want deterministic ODE sampling.
A score field plugs directly into reverse-time SDEs and into anything that
needs to evaluate :math:`\nabla \log p_t`. A noise (``Epsilon``)
parameterisation is what most pretrained diffusion checkpoints actually
predict, and is numerically well-behaved when the schedule is variance-
preserving. A clean-data (``X0``) head can be more stable when the noise
level is high. A "V"-prediction blends velocity and noise terms in a way
that is robust across schedules. And a generator-parameters head — the
``GeneratorParams`` target — predicts an opaque vector that an operator
later linearly pairs with a basis of derivatives, which is what makes the
generator-matching framework able to express drift, drift-plus-diffusion,
and jump processes in one objective.

The choice has practical consequences. The same neural network, trained
against a different parameterisation target, will converge at different
rates, tolerate different schedules, and require different transforms at
sampling time. None of these parameterisations is universally best; each
is a tool for a different job.

Switching at sampling time
--------------------------

A model trained under one parameterisation often needs to be *used* under
another at sampling time. nami exposes a small family of transforms for
exactly this purpose. :class:`~nami.ScoreFromEta` turns a model that
predicts :math:`\eta = \gamma(t) \nabla \log p_t` into a score; analogously
:class:`~nami.ScoreFromRawNoise` handles a raw-noise prediction.
:class:`~nami.DriftFromVelocityScore` combines a velocity head and a score
head into an SDE drift suitable for reverse-time integration. For diffusion
models the analogous conversion is :math:`\nabla \log p_t(x) = -
\varepsilon_\theta / \sigma(t)`, and is handled internally by
:class:`~nami.Diffusion` when you pass
:func:`~nami.epsilon_prediction`, :func:`~nami.score_prediction`, or
:func:`~nami.x0_prediction` as the ``parameterization`` argument.

The principle is that the *parameterisation is part of the contract
between the field and the sampling process*. If you train with
``epsilon_prediction`` and ask the process to interpret the output as
``score_prediction``, you will get garbage; this is the single most
common failure mode in practice.

A small concrete example
------------------------

For interpolant-style models, the most common conversion is from a noise
or eta head to a score. The transforms are thin wrappers that you compose
with the field at sampling time:

.. code-block:: python

   import nami

   gamma = nami.BrownianGamma()
   score = nami.ScoreFromEta(eta_model, gamma)       # eta -> score
   drift = nami.DriftFromVelocityScore(v_model, score, gamma)

The velocity field, the score-converting wrapper, and the drift wrapper
remain orthogonal: you can swap any one without touching the others, which
is the whole point of keeping parameterisations explicit rather than
hard-wired into a single class.

When to use which
-----------------

For deterministic flow matching, predict a velocity. For consistency flow
matching, predict a velocity (the consistency function is built from it).
For diffusion-style training, predict :math:`\varepsilon` against the
standard objective and let :class:`~nami.Diffusion` handle the conversion
at sampling time. For stochastic interpolants where SDE sampling is wanted,
train a velocity *and* a score (or use the raw-noise head together with
:class:`~nami.ScoreFromRawNoise`). For generator matching, predict raw
operator parameters and let the operator interpret them.

See also
--------

- :doc:`core-abstractions` — what a field is, and what ``event_ndim``
  controls.
- :doc:`model-families` — how the choice of parameterisation interacts with
  the choice of process.
- :doc:`numerical-considerations` — when the choice of parameterisation
  starts to interact with the choice of solver.
