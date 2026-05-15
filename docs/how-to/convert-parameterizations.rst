Convert between parameterisations
=================================

When you have a model trained under one parameterisation but need another
for sampling, nami provides a small family of transforms that compose with
the field at sampling time. See :doc:`../explanation/parameterizations` for
why several parameterisations exist and when each is appropriate.

.. code-block:: python

   import nami

   gamma = nami.BrownianGamma()

   # Model predicts eta = gamma(t) * score(x, t)
   score = nami.ScoreFromEta(eta_model, gamma)

   # Model predicts the raw Gaussian noise z in X_t = I_t + gamma(t) z
   score_from_noise = nami.ScoreFromRawNoise(noise_model, gamma)

   # Combine a velocity and a score into an SDE drift
   drift = nami.DriftFromVelocityScore(v_model, score, gamma)

   # Mirror correction term used in backward-SDE formulas
   mirror_v = nami.MirrorVelocityFromScore(score, gamma)

These wrappers are for stochastic-interpolant style parameterisations and
are not the same as diffusion :math:`\varepsilon`-prediction. For diffusion
models, the analogous conversion
:math:`\nabla \log p_t(x) = -\varepsilon_\theta / \sigma(t)` is handled
internally by :class:`~nami.Diffusion` when you pass
``parameterization=nami.epsilon_prediction(schedule=schedule)``.
