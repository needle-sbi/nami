Compute log-likelihoods
=======================

For continuous normalising flows, evaluate

.. math::

   \log p(x) = \log p_{\mathrm{source}}(z)
             + \int \nabla \cdot v_t \,\mathrm{d}t

via the instantaneous change-of-variables formula. The augmented solve
costs more than plain sampling, since every solver step has to evaluate the
divergence of the velocity field, so use it only when log-densities are
actually wanted.

.. code-block:: python

   import nami

   estimator = nami.HutchinsonDivergence()
   log_p = fm(None).log_prob(x_test, estimator=estimator)

   # Or sample and accumulate log-density together in one augmented solve
   samples, log_p_samples = fm(None).sample(
       (128,), return_logp=True, estimator=estimator,
   )

:class:`~nami.HutchinsonDivergence` is unbiased and uses one extra backward
pass per step; this is feasible in high dimensions but introduces variance
into the log-density estimate. :class:`~nami.ExactDivergence` computes the
exact Jacobian trace at :math:`\mathcal{O}(d)` backward passes per step and
is appropriate for low-dimensional problems.

Note that ``return_logp=True`` does not turn flow-matching training into
CNF maximum-likelihood training, it reintroduces CNF-style inference cost
for that call only.
