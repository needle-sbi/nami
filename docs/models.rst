:orphan:

Models
======

Nami supports five transport-map workflows. Each pairs a **loss** (training)
with a **lazy process** (sampling) and reuses the same solver, distribution,
and field abstractions. This page describes every model in detail.

.. contents:: On this page
   :local:
   :depth: 2

Deterministic flow matching
---------------------------

The deterministic formulation learns a velocity field :math:`v_\theta(x, t)` whose time-1 flow
pushes a simple source :math:`p_0` (e.g.\ :math:`\mathcal{N}(0, I)`) onto
a target :math:`p_1` (data).

Training
^^^^^^^^

A probability path :math:`p_t` interpolates between noise and data.
The default is the linear (so called optimal-transport) interpolant:

.. math::

   X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}},
   \qquad
   u_t = x_{\mathrm{data}} - x_{\mathrm{noise}},

and the loss regresses the field against the conditional velocity target:

.. math::

   \mathcal{L}_{\mathrm{FM}}
   = \mathbb{E}_{t,\, x_{\mathrm{noise}},\, x_{\mathrm{data}}}
     \bigl[\lVert v_\theta(X_t, t) - u_t \rVert^2\bigr].

.. code-block:: python

   import torch, nami

   field = nami.VelocityField(dim=8)
   x_data = torch.randn(32, 8)          # data
   x_noise = torch.randn_like(x_data)   # noise

   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )
   loss.backward()

Any :class:`~nami.LinearInterpolant` or :class:`~nami.CosineInterpolant`
can be passed via the ``interpolant`` keyword.

Sampling
^^^^^^^^

Wraps the trained field in a :class:`~nami.FlowMatching` lazy process and
integrates the ODE :math:`\dot x = v_\theta(x, t)` from :math:`t{=}0` to
:math:`t{=}1`:

.. code-block:: python

   fm = nami.FlowMatching(
       field,
       nami.StandardNormal((8,)),
       nami.RK4(steps=50),
   )
   samples = fm().sample((64,))
   samples, log_p = fm().sample(
       (64,),
       return_logp=True,
       estimator=nami.HutchinsonDivergence(),
   )

:class:`~nami.FlowMatching` also supports exact log-likelihood evaluation
via the continuous change-of-variables formula, using
:class:`~nami.ExactDivergence` or :class:`~nami.HutchinsonDivergence`.

.. _fm-training-vs-likelihood-cost:

Training vs likelihood cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flow matching keeps training simple relative to classical
continuous normalising flows.  The standard FM objective is a regression loss
on path samples (:func:`~nami.regression_loss` with
:class:`~nami.LinearInterpolant` and :func:`~nami.velocity_prediction`) so it
does not solve an ODE or estimate a divergence during training.

That changes only when you use the trained model for densities:

- ``process.sample(...)`` solves the state ODE only.
- ``process.log_prob(x, estimator=...)`` solves an augmented ODE and
  evaluates the divergence at every solver step.
- ``process.sample(..., return_logp=True, estimator=...)`` does the same
  augmented solve while sampling, so samples and log-densities come back
  together from one pass!

In other words, ``return_logp=True`` does not turn flow-matching
training into CNF maximum-likelihood training. It reintroduces
CNF-style inference cost for that call only.

This distinction matters in practice:

- If you care mainly about training speed and sample generation, standard
  FM keeps its usual advantage.
- If you need exact or estimated likelihoods, expect CNF-like runtime at
  evaluation time, because divergence tracking is the expensive part.
- If you use :class:`~nami.HutchinsonDivergence`, each solver step adds a
  stochastic trace estimate, typically one extra backward/VJP per step.

.. note::

   This is a convention choice, not a different method. Nami uses one global
   clock for FM, consistency FM, stochastic FM, and generator matching:
   :math:`t=0` for source / noise states and :math:`t=1` for data / clean
   states. The :class:`~nami.Diffusion` process is the exception — it keeps
   the diffusion-native clock (data at small time, noise at large time)
   because the score-based reverse-time PF-ODE is intrinsic to that
   direction. Reversing the clock is an affine reparameterisation of time,
   so the underlying probability path is unchanged as long as the equations
   are translated consistently. What does change is the sign of time
   derivatives when comparing against papers that use the opposite
   orientation.

.. warning::
   - **Too few solver steps**: ``RK4(steps=10)`` may look fine for simple
     distributions but will produce artefacts on complex targets. Start with
     50-100 steps and reduce once quality is confirmed is typically a good starting 
     point.

**Key components**: :func:`~nami.regression_loss`, :class:`~nami.FlowMatching`,
:class:`~nami.VelocityField`, :class:`~nami.LinearInterpolant`,
:class:`~nami.CosineInterpolant`.


Consistency flow matching
-------------------------

Consistency flow matching trains the same velocity field as deterministic FM, but replace the
regression loss with a *self-consistency* objective. Two points on the same
conditional trajectory must map to the same endpoint via a consistency
function, given by:

.. math::

   f(x_t, t) = x_t + (T - t)\,v_\theta(x_t, t),

where :math:`T` is the target time (the data endpoint at :math:`T=1` for
forward consistency, the noise endpoint at :math:`T=0` for reverse
consistency). After training, a single evaluation of :math:`f` generates a
sample, so no ODE integration needed given the one-step nature of the
consistency function. It is important to note that this is an
approximation and in general some variance can be introduced.

.. note::

   nami's consistency-flow-matching support is currently experimental.
   The losses and process APIs expose useful one-step consistency primitives,
   but the tutorial recipe shipped in this repository is a simplified research
   workflow rather than a faithful reproduction of Yang et al. (2024).
   In particular, the local examples use a linear path and often rely on FM
   pretraining, fixed ``delta``, and optional EMA targets instead of the
   paper's full from-scratch training schedule.

Training
^^^^^^^^

Two consistency losses are provided, one for each direction:

**Forward consistency** (:func:`~nami.consistency_loss` with
``target_time=1.0``): maps toward the data endpoint (:math:`t = 1`):

.. math::

   f(x_t, t) = x_t + (1 - t) \cdot v_\theta(x_t, t), \qquad
   \mathcal{L}_f
   = \mathbb{E}\bigl[\lVert f(X_t, t) - f(X_{t'}, t') \rVert^2\bigr].

**Reverse consistency** (:func:`~nami.consistency_loss` with
``target_time=0.0``): maps toward the noise endpoint (:math:`t = 0`):

.. math::

   g(x_t, t) = x_t - t \cdot v_\theta(x_t, t), \qquad
   \mathcal{L}_g
   = \mathbb{E}\bigl[\lVert g(X_{t'}, t') - g(X_t, t) \rVert^2\bigr].

Both calls sample two times :math:`t` and :math:`t' = t + \delta` on the
same trajectory (same :math:`x_{\mathrm{noise}}, x_{\mathrm{data}}` pair).
The target output is stop-gradiented by default; an optional
``target_field`` (e.g.\ an EMA copy) can be passed instead.

For stable toy experiments in nami, the most reliable starting point is to
train a good FM field first and then treat the consistency losses as a
one-step approximation objective layered on top.  Direct from-scratch
consistency training remains an open research workflow here.

Euler-stepped trajectory pairs
""""""""""""""""""""""""""""""

By default, both :math:`x_t` and :math:`x_{t'}` are sampled from the
conditional path (the linear interpolant).  This is unbiased but
introduces variance, because the conditional path and the learned ODE
trajectory do not coincide during training.

Passing ``euler_step=True`` to any of the three losses generates the
second point via a detached Euler step of the learned velocity instead:

.. math::

   x_{t'} = x_t + \delta \cdot v_\theta(x_t, t) \quad\text{(detached)}.

This places the pair on the learned ODE trajectory, eliminating the
trajectory-mismatch variance.  The Euler step is detached so that
gradients flow only through the consistency comparison, not through the
step construction.  For both :func:`~nami.consistency_loss` calls this
reuses the velocity already computed for the consistency function, so
there is no extra forward-pass cost.

.. code-block:: python

   common = dict(
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       euler_step=True,
   )
   loss_f = nami.consistency_loss(field, target_time=1.0, **common)
   loss_g = nami.consistency_loss(field, target_time=0.0, **common)
   loss_h = nami.log_density_consistency_loss(
       field, h_head,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       euler_step=True,
   )

**Log-prob consistency** (:func:`~nami.log_density_consistency_loss`)
trains a scalar head :math:`h_\theta(x_t, t)` to predict
:math:`\log p_t(x_t)` via

.. math::

   h(x_{t'}, t') \approx h(x_t, t) + \delta \cdot \operatorname{div} v_\theta(x_t, t)

with a boundary anchor :math:`h(x, 0) = \log p_{\mathrm{base}}(x)` at the
noise endpoint.  The divergence is estimated via Hutchinson during
training; at inference :math:`h_\theta(x, 1)` gives the log-density in one
forward pass.

.. code-block:: python

   import torch, nami

   field = nami.VelocityField(dim=8)
   h_head = nami.ConsistencyHead(dim=8, hidden=128, layers=2)
   x_data = torch.randn(32, 8)
   x_noise = torch.randn_like(x_data)

   # a simple experimental recipe: combine all three objectives jointly.
   # but for robust use, start from a strong FM field and treat this as
   # one-step fine-tuning
   interpolant = nami.LinearInterpolant()
   parameterization = nami.velocity_prediction()
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
   loss.backward()

Sampling, inversion, and log-prob
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrap the trained field in a :class:`~nami.ConsistencyFlowMatching` lazy
process. Sampling is a single forward pass.

.. code-block:: python

   cfm = nami.ConsistencyFlowMatching(
       field, nami.StandardNormal((8,)), h_head=h_head,
   )
   process = cfm()
   samples = process.sample((64,))        # one-step: noise to data

One-step inversion (data → noise) is also available via the reverse
consistency function:

.. code-block:: python

   z = process.invert(samples)             # one-step: data to noise

One-step log-density evaluation uses the trained ``h_head``:

.. code-block:: python

   log_p = process.log_prob(x)             # one-step via h_head

For exact log-likelihood, fall back to full ODE integration:

.. code-block:: python

   cfm = nami.ConsistencyFlowMatching(
       field, nami.StandardNormal((8,)), nami.RK4(steps=50), h_head=h_head,
   )
   log_p_ode = cfm().log_prob(x, ode=True, estimator=nami.HutchinsonDivergence())

.. warning::

   - **Delta schedule**: the ``delta`` parameter controls how close the two
     trajectory points are. Smaller values give tighter consistency but
     noisier gradients. ``0.01`` is a reasonable default.
   - **EMA target**: for stable training on complex distributions, pass an
     EMA copy of the field as ``target_field`` in the loss (and
     ``target_h_head`` for the log-prob loss).
   - **One-step vs ODE log_prob**: one-step log-density via ``h_head`` is
     fast but approximate.  For high-precision work (e.g.\ profile
     likelihood fits in HEP), use ``ode=True`` with a solver.
   - **Divergence variance**: the Hutchinson estimator used during training
     adds noise to the log-prob consistency signal.  Using more probes or a
     larger ``lambda_boundary`` can help stabilise convergence.
   - **Trajectory mismatch**: by default, both trajectory points come from the
     conditional path, which differs from the learned ODE trajectory during
     training.  This adds variance (not bias) to gradients.  Pass
     ``euler_step=True`` to place the pair on the learned trajectory instead.


Stochastic flow matching
------------------------

The stochastic formulation adds noise to the interpolant, following the stochastic interpolants
framework of Albergo et al. [1]_ [2]_. The interpolant becomes:

.. math::

   X_t = I_t + \gamma(t)\,Z,
   \qquad Z \sim \mathcal{N}(0, I),

where :math:`I_t` is a deterministic path and :math:`\gamma(t)` is a noise
schedule that vanishes at the endpoints.

Training
^^^^^^^^

The velocity target picks up an extra noise term:

.. math::

   \mathcal{L}_{\mathrm{SFM}}
   = \mathbb{E}_{t,\, x_0,\, x_1,\, Z}
     \bigl[\lVert v_\theta(X_t, t) - \bigl(u_t + \dot\gamma(t)\,Z\bigr) \rVert^2\bigr].

.. code-block:: python

   loss = nami.stochastic_fm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       gamma=nami.BrownianGamma(),
   )

Available gamma schedules:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Schedule
     - Definition
   * - :class:`~nami.ZeroGamma`
     - :math:`\gamma(t) = 0`. Recovers deterministic FM.
   * - :class:`~nami.BrownianGamma`
     - :math:`\gamma(t) = \sqrt{t(1-t)}`. Brownian bridge variance.
   * - :class:`~nami.ScaledBrownianGamma`
     - :math:`\gamma(t) = \sqrt{s \cdot t(1-t)}` for scale :math:`s > 0`.

Sampling
^^^^^^^^

Sampling uses the same :class:`~nami.FlowMatching` process.  For ODE
sampling, simply use the stochastic-FM--trained field with ``RK4`` or
``Heun``.  For SDE sampling, pair with ``EulerMaruyama`` and a
:class:`~nami.DriftFromVelocityScore` transform.

.. warning:: Common gotchas

   - **Choosing gamma**: ``BrownianGamma`` is a sensible default. If training
     is unstable, try ``ScaledBrownianGamma`` with a smaller scale (e.g. 0.5).
   - **Sampling**: at inference time, a stochastic-FM--trained field is used
     with the same ``FlowMatching`` process. SDE sampling requires an
     additional score estimate via :class:`~nami.DriftFromVelocityScore`.


Diffusion
---------

Diffusion instead defines a forward noising process
:math:`q(x_t \mid x_0) = \mathcal{N}(\alpha_t\,x_0,\;\sigma_t^2 I)` via
a noise schedule, and train a model to reverse it by predicting
:math:`\varepsilon`, the score :math:`\nabla\!\log p_t`, or the clean data
:math:`x_0`.

Noise schedules
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Schedule
     - Definition
   * - :class:`~nami.VPSchedule`
     - Variance-preserving: :math:`\alpha_t^2 + \sigma_t^2 = 1`, linear :math:`\beta`\-schedule [3]_.
   * - :class:`~nami.VESchedule`
     - Variance-exploding: :math:`\alpha_t = 1`, geometric :math:`\sigma_t` growth [4]_.
   * - :class:`~nami.EDMSchedule`
     - Karras et al.\ with analytic preconditioning :math:`c_{\mathrm{skip}}, c_{\mathrm{out}}, c_{\mathrm{in}}` [5]_.

Training
^^^^^^^^

Training uses an external denoising objective (e.g.\ the standard
:math:`\varepsilon`\-prediction MSE) paired with the chosen schedule.
The ``parameterization`` flag (``"eps"``, ``"score"``, or ``"x0"``) tells
the sampling process how to interpret the model output.

Sampling
^^^^^^^^

:class:`~nami.Diffusion` is a lazy process that converts any
parameterisation into the reverse-time SDE or probability-flow ODE:

.. code-block:: python

   schedule = nami.VPSchedule()
   diff = nami.Diffusion(
       model,
       schedule,
       nami.DPMSolverPP(steps=20),
       parameterization=nami.epsilon_prediction(schedule=schedule),
       event_shape=(8,),
   )
   samples = diff().sample((64,))

ODE solvers (:class:`~nami.RK4`, :class:`~nami.Heun`,
:class:`~nami.DPMSolverPP`) integrate the probability-flow ODE.
:class:`~nami.EulerMaruyama` integrates the reverse SDE.

.. warning::

   - **Parameterisation mismatch**: if you train with ``"eps"`` but set
     ``parameterization="score"`` at sampling time, the model will produce
     garbage. These must match!
   - **ODE vs SDE**: ODE solvers (``RK4``, ``Heun``, ``DPMSolverPP``)
     integrate the probability-flow ODE; ``EulerMaruyama`` integrates the
     reverse SDE. The SDE sampler generally produces higher quality at more
     compute.

.. warning:: Temporary note [LEVI] FIX

   - **Training is external**: unlike flow matching, nami does not provide
     the diffusion training loss in the API you train the model with your own
     denoising objective and hand it to ``Diffusion`` for sampling.


Generator matching
------------------

Rather than committing to a single parameterisation (velocity,
score, noise) up front, one can instead learn the *generator* :math:`L_t` of a
continuous-time Markov process directly. A field
:math:`F_\theta(x, t)` predicts operator parameters; the operator
interprets them through a linear pairing:

.. math::

   (L_t f)(x) = \bigl\langle K f(x),\; F_t(x) \bigr\rangle.

The generator
^^^^^^^^^^^^^

A :class:`~nami.GeneratorOperator` defines the functional form of
:math:`L_t`. The built-in :class:`~nami.ItoGeneratorOperator` covers
continuous diffusion generators:

.. math::

   (L_t f)(x)
   = b_t(x)^\top \nabla f(x)
   + \tfrac{1}{2}\operatorname{tr}\!\bigl(a_t(x)\,\nabla^2 f(x)\bigr),

with two modes:

- ``diffusion="none"``: learn drift :math:`b_t` only; sampling is ODE.
- ``diffusion="diagonal"``: learn drift + diagonal diffusion scale;
  sampling is SDE via :class:`~nami.EulerMaruyama`.

Conditional paths and targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training is driven by a conditional interpolant :math:`p_{t \mid Z}` between
:math:`x_{\mathrm{noise}}` (source) and :math:`x_{\mathrm{data}}` (target).
Available interpolants:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Interpolant
     - Definition
   * - :class:`~nami.LinearInterpolant`
     - :math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}`. Target drift :math:`b_t = x_{\mathrm{data}} - x_{\mathrm{noise}}`.
   * - :class:`~nami.BrownianBridgeInterpolant`
     - :math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}} + \sigma\sqrt{t(1-t)}\,Z`. Analytic conditional drift from the Brownian bridge.

Training
^^^^^^^^

.. math::

   \mathcal{L}_{\mathrm{GM}}
   = \mathbb{E}\!\bigl[\lVert F_\theta(X_t, t) - F_t(X_t \mid Z) \rVert^2\bigr].

The unified objective is :func:`~nami.regression_loss` with
:func:`~nami.generator_prediction(op)` — it regresses directly against
interpolant-derived operator parameters (the conditional-target form).
Standard marginalisation shows this shares gradients with the marginal
generator-matching expectation.

.. code-block:: python

   op = nami.ItoGeneratorOperator(event_shape=(8,), diffusion="none")
   field = nami.GeneratorField(dim=8, operator=op)

   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.generator_prediction(op),
       eps_t=0.0,
   )
   loss.backward()

Sampling
^^^^^^^^

:class:`~nami.GeneratorMatching` wraps field + operator + solver into a
lazy process. The operator's ``runtime_kind`` determines integration:

.. code-block:: python

   gm = nami.GeneratorMatching(
       field, op,
       nami.RK4(steps=50),
       event_shape=(8,),
   )
   samples = gm().sample((64,))

Recovering familiar models
^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+-----------------------------------+---------------------------------+
| Interpolant                     | Operator config                   | Recovers                        |
+=================================+===================================+=================================+
| ``LinearInterpolant``           | ``diffusion="none"``              | Deterministic flow matching     |
+---------------------------------+-----------------------------------+---------------------------------+
| ``BrownianBridgeInterpolant``   | ``diffusion="none"``              | Stochastic FM (drift only)      |
+---------------------------------+-----------------------------------+---------------------------------+
| ``BrownianBridgeInterpolant``   | ``diffusion="diagonal"``          | Full Ito diffusion model        |
+---------------------------------+-----------------------------------+---------------------------------+

.. note::

   - **Operator mode vs solver**: ``diffusion="none"`` means drift-only (ODE)
     so use ``RK4``. ``diffusion="diagonal"`` adds a learned diffusion
     coefficient so use ``EulerMaruyama``. Mixing these up will raise an error
     or produce incorrect samples.
   - **Conditional vs marginal targets**: ``regression_loss`` with
     :func:`~nami.generator_prediction` regresses directly against
     conditional path-derived operator parameters. The marginal generator-
     matching expectation shares gradients under marginalisation, so the
     conditional form is the practical choice.


Schrodinger bridge matching
----------------------------

Another formulation of the transport maps is to learn separate velocity and score fields, then reconstruct the
SDE drift via :class:`~nami.DriftFromVelocityScore`. This follows the
Schrodinger bridge matching approach of Tong et al.

Training
^^^^^^^^

.. code-block:: python

   loss = nami.bridge_matching_loss(
       flow_field, score_field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.BrownianBridgeInterpolant(),
   )

The loss jointly regresses velocity and score targets derived from the
Brownian bridge path.

Sampling
^^^^^^^^

After training, reconstruct the drift from the two fields and sample with a
standard :class:`~nami.FlowMatching` or :class:`~nami.Diffusion` process.

**Key components**: :func:`~nami.bridge_matching_loss`,
:class:`~nami.BrownianBridgeInterpolant`,
:class:`~nami.DriftFromVelocityScore`,
:class:`~nami.ScoreFromEta`,
:class:`~nami.ScoreFromRawNoise`.


Choosing a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Use case
     - Loss
     - Process
   * - Standard flow matching (ODE)
     - ``regression_loss``
     - ``FlowMatching``
   * - Fast few-step generation
     - ``consistency_loss``
     - ``ConsistencyFlowMatching``
   * - Stochastic interpolants
     - ``stochastic_fm_loss``
     - ``FlowMatching``
   * - Conventional diffusion (:math:`\varepsilon` / score / :math:`x_0`)
     - (external)
     - ``Diffusion``
   * - Unified drift + diffusion (operator-centric)
     - ``regression_loss`` + ``generator_prediction``
     - ``GeneratorMatching``
   * - Schrodinger bridge (flow + score)
     - ``bridge_matching_loss``
     - ``FlowMatching`` / ``Diffusion``


References
----------

.. [1] Albergo, M. S. and Vanden-Eijnden, E., *Building Normalizing Flows with
       Stochastic Interpolants*, ICLR 2023.

.. [2] Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E., *Stochastic
       Interpolants: A Unifying Framework for Flows and Diffusions*, 2023.

.. [3] Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.

.. [4] Song et al., *Score-Based Generative Modeling through Stochastic
       Differential Equations*, ICLR 2021.

.. [5] Karras et al., *Elucidating the Design Space of Diffusion-Based
       Generative Models*, 2022.
