Numerical considerations
========================

Once you have decided on a path, a parameterisation, and a family, a handful
of numerical choices remain: which solver to integrate with, how many steps
to use, whether to add noise to the interpolant, and how to estimate
divergence when likelihoods are wanted. These choices are largely
independent of the modelling choices and are worth thinking about on their
own.

Choosing a solver
-----------------

For ODE sampling, :class:`~nami.RK4` is the workhorse. Classical
fourth-order Runge--Kutta with a fixed step has local error
:math:`\mathcal{O}(h^4)` and produces clean samples for most problems with
50--100 steps. :class:`~nami.Heun` is cheaper per step (predictor-corrector,
:math:`\mathcal{O}(h^2)`) and a reasonable choice when the dynamics are
smooth and the step budget is tight. :class:`~nami.DPMSolverPP` is a
schedule-aware solver designed for diffusion probability paths; it adapts
its step schedule to the noise schedule and is the right choice for
:class:`~nami.Diffusion` when you want few-step generation without a
consistency model.

For SDE sampling, :class:`~nami.EulerMaruyama` is the only built-in
option. It has strong order :math:`1/2`, which means it converges slowly
in step count; expect to need more steps than for an equivalent ODE solve.
SDE sampling generally produces higher sample quality at higher compute
than the corresponding probability-flow ODE, particularly for diffusion
models with diagonal diffusion operators.

The common failure mode is using too few steps. ``RK4(steps=10)`` may
look fine on a low-dimensional toy distribution but produce visible
artefacts on a complex target. A reasonable workflow is to start with
50--100 steps, confirm the qualitative behaviour, and reduce from there.

Solver and operator have to agree. A drift-only operator
(``diffusion="none"``) is integrated as an ODE and pairs with
:class:`~nami.RK4` or :class:`~nami.Heun`; a drift-plus-diagonal-diffusion
operator (``diffusion="diagonal"``) is integrated as an SDE and requires
:class:`~nami.EulerMaruyama`. Mixing these up will raise an error or
produce incorrect samples.

The stochastic interpolant
--------------------------

The deterministic interpolant
:math:`X_t = (1-t)\,x_{\mathrm{target}} + t\,x_{\mathrm{source}}` has the
useful property of being maximally simple, but it can be replaced by a
stochastic interpolant :math:`X_t = I_t + \gamma(t)\,Z` with little cost.
The trade-off is regularisation against variance. With
:class:`~nami.BrownianGamma` (:math:`\gamma(t) = \sqrt{t(1-t)}`) the
interpolant carries Brownian-bridge noise, which can stabilise training
on rough targets but adds variance to the velocity target.
:class:`~nami.ScaledBrownianGamma` lets you tune the noise magnitude,
which is often the first knob to reach for when stochastic FM training is
unstable: try a smaller scale, e.g. 0.5. :class:`~nami.ZeroGamma` recovers
deterministic FM exactly and is useful as a sanity check.

The same trained field is sampled by the same
:class:`~nami.FlowMatching` process whether it was trained with or without
noise. If you want *SDE* sampling from a stochastic-FM-trained model, pair
it with a score estimate via :class:`~nami.DriftFromVelocityScore` and an
:class:`~nami.EulerMaruyama` solver.

Divergence: Hutchinson vs exact
-------------------------------

For continuous normalising flow log-likelihood,

.. math::

   \log p(x) = \log p_{\mathrm{source}}(z)
             + \int \nabla \cdot v_t \,\mathrm{d}t,

every solver step has to evaluate the divergence of the velocity field.
nami exposes two estimators with different cost profiles.

:class:`~nami.ExactDivergence` computes the full Jacobian trace
:math:`\operatorname{tr}(\partial v / \partial x)`. This costs
:math:`\mathcal{O}(d)` backward passes per step, which is fine in low
dimensions and prohibitive otherwise.

:class:`~nami.HutchinsonDivergence` is the stochastic trace estimator
:math:`\operatorname{tr}(J) = \mathbb{E}_\epsilon[\epsilon^\top J \epsilon]`
for :math:`\epsilon \sim \mathcal{N}(0, I)`. It is unbiased and requires a
single backward pass per step, which is what makes likelihood evaluation
feasible in high dimensions. The cost is variance: each call returns a
noisy estimate. For most use cases this is acceptable; for high-precision
work (e.g. profile-likelihood fits) you can run multiple probes per step
or fall back to the exact estimator.

A practical note on flow-matching cost
--------------------------------------

The standard flow-matching objective is a regression loss on path samples,
so training does not solve an ODE or estimate a divergence. That changes
only at inference: ``process.sample(...)`` solves the state ODE only, but
``process.log_prob(x, estimator=...)`` solves an augmented ODE that
evaluates the divergence at every solver step. ``return_logp=True`` does
the same augmented solve during sampling, so samples and log-densities
come back together from one pass — convenient but not free. The headline
is that ``return_logp=True`` does not turn flow-matching training into
CNF maximum-likelihood training; it reintroduces CNF-style inference cost
for that call only.

See also
--------

- :doc:`core-abstractions` — where solvers and divergence estimators sit in
  the stack.
- :doc:`parameterizations` — how the choice of parameterisation interacts
  with the SDE/ODE distinction.
- :doc:`model-families` — which family each numerical choice is most
  relevant to.
