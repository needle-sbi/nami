How-to guides
=============

Short, copy-pasteable recipes for common nami tasks. Each page assumes you
already have rough orientation; for the conceptual scaffolding, see
:doc:`../explanation/index`. For full signatures and defaults, see
:doc:`../api/index`.

.. toctree::
   :maxdepth: 1

   train-flow-matching
   train-stochastic-flow-matching
   train-consistency-flow-matching
   train-generator-matching
   sample-with-diffusion
   bridge-matching
   condition-on-context
   swap-interpolant
   convert-parameterizations
   compute-log-likelihood
   sanity-check-zero-gamma

Recipes at a glance
-------------------

- :doc:`train-flow-matching` — the default starting point: regress a
  velocity field, sample with :class:`~nami.RK4`.
- :doc:`train-stochastic-flow-matching` — add interpolant noise as a
  training-time regulariser.
- :doc:`train-consistency-flow-matching` — single-step sampling via a
  consistency objective; experimental.
- :doc:`train-generator-matching` — learn Itô generator parameters
  directly; recovers FM, stochastic FM, and diffusion as special cases.
- :doc:`sample-with-diffusion` — wrap a pretrained denoising model in
  :class:`~nami.Diffusion` to get a sampler.
- :doc:`bridge-matching` — learn velocity and score jointly along a
  Brownian-bridge path.
- :doc:`condition-on-context` — bind context :math:`c` at sampling time
  via the lazy process.
- :doc:`swap-interpolant` — change the path geometry without touching the
  rest of the pipeline.
- :doc:`convert-parameterizations` — turn a score, an eta, or a raw-noise
  prediction into the quantity a process actually wants.
- :doc:`compute-log-likelihood` — augmented ODE solve with Hutchinson or
  exact divergence.
- :doc:`sanity-check-zero-gamma` — verify stochastic FM reduces to
  deterministic FM at :math:`\gamma \equiv 0`.
