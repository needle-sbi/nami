Convert between parameterisations
=================================

When you have a model trained under one parameterisation but need another
for sampling, nami provides functional conversions that map between the
diffusion quantities :math:`\varepsilon`, score, and :math:`x_0`, plus
composite fields that combine a velocity and a score into an SDE drift.

.. code-block:: python

   import nami

   # diffusion conversions are pure tensor functions keyed by the schedule
   # coefficients sigma(t) (and alpha(t) where needed).
   score = nami.diffusion.eps_to_score(eps, sigma)            # eps(x,t) -> score(x,t)
   eps   = nami.diffusion.score_to_eps(score, sigma)          # score(x,t) -> eps(x,t)
   x0    = nami.diffusion.score_to_x0(x, score, sigma, alpha) # score(x,t) -> x0(x,t)
   score = nami.diffusion.x0_to_score(x, x0, sigma, alpha)    # x0(x,t) -> score(x,t)

   # combine separately trained velocity and score heads into a
   # probability-flow / SDE drift: u(x,t) = v(x,t) gamma(t) gamma_dot(t) s(x,t)
   gamma = nami.BrownianGamma()
   drift = nami.DriftFromVelocityScore(v_model, score_model, gamma)

For diffusion models the same relation
:math:`\nabla \log p_t(x) = -\varepsilon_\theta / \sigma(t)` is applied
internally by :class:`~nami.Diffusion` when you pass
``parameterization=nami.epsilon_prediction(schedule=schedule)``, so you do
not normally need to convert by hand. To train or sample directly in the
score parameterisation, use
``parameterization=nami.score_prediction(schedule=schedule)``.
