:orphan:

Quickstart
==========

This page shows minimal end-to-end workflows: train a loss, configure a
sampler, draw samples. Each section is self-contained — copy any block
and it will run as-is.

Flow matching
-------------

Learn :math:`v_\theta \approx u_t = x_{\mathrm{data}} - x_{\mathrm{noise}}`
along the linear interpolant
:math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}`, then sample
by integrating the ODE :math:`\dot{x} = v_\theta(x, t)` from :math:`t=0`
to :math:`t=1`.

.. note::

   Nami uses :math:`t=0` for source / noise and :math:`t=1` for data across
   FM, consistency FM, stochastic FM, and generator matching. The
   :class:`~nami.Diffusion` process keeps the opposite (diffusion-native)
   orientation because the score-based reverse-time PF-ODE is intrinsic to
   that direction.

.. code-block:: python

   import torch
   import nami

   dim = 8
   field = nami.VelocityField(dim=dim)               # neural net predicting v(x, t)
   base = nami.StandardNormal(event_shape=(dim,))     # source distribution at t=1
   solver = nami.RK4(steps=50)                        # ODE integrator for sampling
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

   for step in range(500):
       x_data = torch.randn(256, dim)        # target samples (data)
       x_noise = torch.randn_like(x_data)    # source samples (noise)
       loss = nami.regression_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=nami.LinearInterpolant(),
           parameterization=nami.velocity_prediction(),
           eps_t=0.0,
       )
       optim.zero_grad()
       loss.backward()
       optim.step()

   # After training, wrap the field into a lazy process and sample
   fm = nami.FlowMatching(field, base, solver)
   samples = fm(None).sample((128,))    # (128, 8)

.. admonition:: What This Does
   :class: what-this-does

   1. **Training**: each iteration samples random time points internally and
      regresses the field against the conditional velocity :math:`u_t = x_1 - x_0`.
      You should see the loss decrease from roughly :math:`\sim d` (here 8) towards
      zero over the 500 steps.

   2. **Sampling**: ``FlowMatching`` is a *lazy process* — calling ``fm(None)``
      binds it (here with no conditioning context) into a runnable process. The
      solver then integrates from :math:`t=0` (noise) to :math:`t=1` (data) in
      50 RK4 steps, producing 128 samples of shape ``(dim,)``.

Consistency flow matching
-------------------------

Train the same velocity field as above, but with a consistency loss that
enables single-step sampling.

This is a **minimal API sketch**, not the recommended paper-faithful training
recipe. In local nami experiments, consistency training is best treated as an
experimental one-step approximation layer on top of a strong FM baseline.

.. code-block:: python

   import torch
   import nami

   dim = 8
   field = nami.VelocityField(dim=dim)
   base = nami.StandardNormal(event_shape=(dim,))
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

   interpolant = nami.LinearInterpolant()
   parameterization = nami.velocity_prediction()

   for step in range(500):
       x_data = torch.randn(256, dim)
       x_noise = torch.randn_like(x_data)
       # Simple experimental recipe: train forward + reverse consistency jointly.
       # For stable results, start from a pretrained FM field or use EMA targets.
       loss_f = nami.consistency_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant, parameterization=parameterization,
           target_time=1.0, delta=0.01,
       )
       loss_g = nami.consistency_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant, parameterization=parameterization,
           target_time=0.0, delta=0.01,
       )
       loss = loss_f + loss_g
       optim.zero_grad()
       loss.backward()
       optim.step()

   # Single-step sampling — no solver needed
   cfm = nami.ConsistencyFlowMatching(field, base)
   process = cfm()
   samples = process.sample((128,))    # (128, 8)

   # One-step inversion (data → noise)
   z = process.invert(samples)          # (128, 8)

.. admonition:: What This Does
   :class: what-this-does

   1. **Training**: two symmetric consistency losses are combined.
      :func:`~nami.consistency_loss` with ``target_time=1.0`` enforces that
      nearby trajectory points predict the same *data* endpoint via
      :math:`f(x_t, t) = x_t + (1 - t) \cdot v_\theta(x_t, t)`.
      The same loss with ``target_time=0.0`` does the analogous job toward
      the *noise* endpoint via
      :math:`g(x_t, t) = x_t - t \cdot v_\theta(x_t, t)`.
      Target outputs are stop-gradiented.

   2. **Sampling**: ``ConsistencyFlowMatching`` evaluates the consistency
      function once from pure noise — a single forward pass replaces the
      50-step ODE integration used in standard FM.

   3. **Inversion**: ``process.invert(x)`` maps data back to noise in one
      step via the reverse consistency function.  For log-likelihood, pass
      a solver: ``ConsistencyFlowMatching(field, base, nami.RK4(steps=50))``.

.. note::

   The original Consistency-FM paper (Yang et al., 2024) trains from scratch
   with EMA targets and additional schedule details. nami currently provides
   the core consistency losses and process wrapper, but not that full recipe.

To get **one-step log-densities**, add a log-prob head:

.. code-block:: python

   h_head = nami.ConsistencyHead(dim=dim)
   optim_h = torch.optim.Adam(
       list(field.parameters()) + list(h_head.parameters()), lr=1e-3,
   )

   interpolant = nami.LinearInterpolant()
   parameterization = nami.velocity_prediction()

   for step in range(500):
       x_data = torch.randn(256, dim)
       x_noise = torch.randn_like(x_data)
       loss_f = nami.consistency_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant, parameterization=parameterization,
           target_time=1.0, delta=0.01,
       )
       loss_g = nami.consistency_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant, parameterization=parameterization,
           target_time=0.0, delta=0.01,
       )
       loss_h = nami.log_density_consistency_loss(
           field, h_head,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant,
           delta=0.01,
       )
       loss = loss_f + loss_g + 0.1 * loss_h
       optim_h.zero_grad()
       loss.backward()
       optim_h.step()

   # One-step log-density — no ODE needed
   cfm = nami.ConsistencyFlowMatching(field, base, h_head=h_head)
   log_p = cfm().log_prob(samples)  # (128,)

Stochastic flow matching
-------------------------

Replace the deterministic interpolant with
:math:`X_t = I_t + \gamma(t)\,Z`, :math:`\;\gamma(t) = \sqrt{t(1-t)}`.
Everything else is identical — just swap the loss function:

.. code-block:: python

   loss = nami.stochastic_fm_loss(
       field, x_data, x_noise, gamma=nami.BrownianGamma(),
   )

.. admonition:: What This Does
   :class: what-this-does

   The stochastic loss adds Brownian bridge noise to the interpolant during
   training, which can act as a regulariser. At sampling time you still use
   the same ``FlowMatching`` process — the field has simply been trained with
   a noisier target. To recover exact deterministic FM, pass
   ``gamma=nami.ZeroGamma()``.

Diffusion
---------

VP-schedule diffusion with :math:`\varepsilon`-prediction. Sampling uses
the reverse SDE via Euler--Maruyama.

.. code-block:: python

   import nami

   model = nami.VelocityField(dim=8)
   schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)
   diffusion = nami.Diffusion(
       model=model,
       schedule=schedule,
       solver=nami.EulerMaruyama(steps=100),
       parameterization=nami.epsilon_prediction(schedule=schedule),
       event_shape=(8,),
   )
   samples = diffusion(None).sample((64,))

.. admonition:: What This Does
   :class: what-this-does

   Unlike flow matching, nami's ``Diffusion`` wrapper handles *sampling only*
   — training uses an external denoising loss (the standard
   :math:`\varepsilon`-prediction MSE). The ``parameterization`` argument
   tells the sampler how to interpret the model's output:
   :func:`~nami.epsilon_prediction` for noise-prediction,
   :func:`~nami.score_prediction` for :math:`\nabla \log p_t`, and
   :func:`~nami.x0_prediction` for clean-data prediction.

Generator matching
------------------

Learn Itô generator parameters :math:`(b_t, a_t)` via conditional
regression. Here we use a deterministic path (drift only, ODE sampling).

.. code-block:: python

   import torch
   import nami

   dim = 8
   # The operator defines the functional form of the generator
   operator = nami.ItoGeneratorOperator(event_shape=(dim,), diffusion="none")
   # The field predicts the operator's parameters from (x, t)
   field = nami.GeneratorField(dim=dim, param_shape=operator.parameter_shape)
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

   interpolant = nami.LinearInterpolant()
   parameterization = nami.generator_prediction(operator)

   for _ in range(500):
       x_data = torch.randn(256, dim)
       x_noise = torch.randn_like(x_data)
       loss = nami.regression_loss(
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant,
           parameterization=parameterization,
           eps_t=0.0,
       )
       optim.zero_grad()
       loss.backward()
       optim.step()

   gm = nami.GeneratorMatching(
       field, operator, nami.RK4(steps=50),
       event_shape=(dim,),
   )
   samples = gm(None).sample((128,))

.. admonition:: What This Does
   :class: what-this-does

   Generator matching splits the model into two parts: a **field** (neural net)
   that predicts raw parameters, and an **operator** that interprets them as
   drift :math:`b_t` (and optionally diffusion :math:`a_t`). With
   ``diffusion="none"``, the operator produces drift only, and sampling is a
   plain ODE. This setup recovers deterministic flow matching as a special
   case. See :doc:`models` for a full equivalence table.

For Brownian bridge interpolants with SDE sampling, swap the interpolant and
operator mode:

.. code-block:: python

   operator = nami.ItoGeneratorOperator(event_shape=(dim,), diffusion="diagonal")
   interpolant = nami.BrownianBridgeInterpolant(sigma=0.1)
   solver = nami.EulerMaruyama(steps=100)

Conditional generation
----------------------

Bind context :math:`c \in \mathbb{R}^{d_c}` at sampling time. The field
receives :math:`c` via its third argument.

.. code-block:: python

   context = torch.randn(16, 4)          # 16 conditioning vectors
   process = fm(context)                  # bind once
   samples = process.sample((1,))         # (1, 16, 8)

.. admonition:: What This Does
   :class: what-this-does

   The context tensor has batch shape ``(16,)``, so the bound process generates
   one sample *per context vector*. The output shape is ``(S, B, E)`` =
   ``(1, 16, 8)`` — one draw for each of the 16 conditioning inputs. To draw
   multiple samples per context, increase the sample shape: ``process.sample((50,))``
   gives ``(50, 16, 8)``.

Log-likelihood
--------------

For continuous normalising flows, evaluate
:math:`\log p(x) = \log p_1(z) + \int_0^1 \nabla \cdot v_t\,\mathrm{d}t`
via the instantaneous change-of-variables formula.

.. code-block:: python

   estimator = nami.HutchinsonDivergence()
   log_p = fm(None).log_prob(x_test, estimator=estimator)

   # Or sample and accumulate log-density in the same ODE solve
   samples, log_p_samples = fm(None).sample(
       (128,),
       return_logp=True,
       estimator=estimator,
   )

.. admonition:: What This Does
   :class: what-this-does

   ``log_prob`` integrates the ODE *reverse* (:math:`t: 1 \to 0`) while
   simultaneously accumulating the divergence of the velocity field. The
   :class:`~nami.HutchinsonDivergence` estimator uses a stochastic trace
   estimator (one extra backward pass per step), making it feasible in high
   dimensions. ``sample(..., return_logp=True)`` uses the same augmented
   dynamics during sampling, so generated samples can be returned together
   with their log-density estimate. For low-dimensional problems,
   :class:`~nami.ExactDivergence` computes the exact Jacobian trace. For the
   training-versus-inference tradeoff, see
   :ref:`fm-training-vs-likelihood-cost`.
