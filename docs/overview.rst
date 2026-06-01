:orphan:

Core Concepts
=============

This page covers the foundational ideas behind nami. Understanding these
conventions will make the rest of the documentation (and your own code) much
easier to follow.

Project scope
-------------

Nami is best understood as a small **research toolkit**. The main goal is to
collect several related transport-map workflows behind a common API with as few
moving parts as possible.

That means:

- the core abstractions are the main product: fields, losses, paths, solvers,
  distributions, and lazy process wrappers
- some workflows are more mature than others
- notebook recipes should be read as experiments unless they explicitly claim
  stronger validation

It does **not** aim to be a production framework or to claim paper-faithful
reproductions by default.

Tensor layout
-------------

All tensors follow the shape convention

.. math::

   \underbrace{S}_{\text{sample}} \;\times\;
   \underbrace{B}_{\text{batch}} \;\times\;
   \underbrace{E}_{\text{event}}

where :math:`S` indexes independent draws, :math:`B` indexes parallel
computations sharing the same model parameters, and :math:`E` describes a
single data point. The integer property ``event_ndim`` declares how many
trailing dimensions constitute :math:`E`; all remaining leading dimensions
are treated as :math:`S \times B`.

Time convention
---------------

.. math::

   t = 0 \;\longleftrightarrow\; p_{\mathrm{source}}, \qquad
   t = 1 \;\longleftrightarrow\; p_{\mathrm{data}}.

Sampling integrates :math:`t : 0 \to 1`. Reverse (likelihood) evaluation
integrates :math:`t : 1 \to 0`, and the base-density anchor for the
change-of-variables identity is taken at :math:`t = 0` (the noise
endpoint). The :class:`~nami.Diffusion` process is the exception: its
intrinsic reverse-time PF-ODE keeps clean data at small time and noise
at large time.

.. note::

   This convention is a library-level choice. Nami includes both
   flow-matching and diffusion-style workflows, and those literatures are
   often written with different clocks. Diffusion expositions usually place
   clean / data states at small time and noisy / source states at large time,
   while the original Flow Matching presentation in Lipman et al. (2023)
   uses the opposite orientation.

   Because nami exposes these models through one shared API, it has to pick
   one global time axis. The library uses the flow-matching / stochastic-
   interpolant orientation: source / noise states at :math:`t=0` and data
   / clean states at :math:`t=1`. This keeps FM and generator matching
   aligned on the same endpoint convention. The :class:`~nami.Diffusion`
   process is the exception — it retains the diffusion-native orientation
   (data at small time, noise at large time) because the score-based
   reverse-time PF-ODE is intrinsic to that direction.

   This does not change the underlying method; it is a reparameterisation of
   time. To translate formulas from papers that use the opposite clock, set
   :math:`t_{\mathrm{nami}} = 1 - t_{\mathrm{paper}}`. Endpoint labels swap,
   and time derivatives pick up a minus sign; for example
   :math:`v_{\mathrm{nami}}(x, t) = -v_{\mathrm{paper}}(x, 1-t)`.

Lazy binding
------------

Model configuration is separated from context binding via a two-phase
protocol:

.. math::

   \underbrace{\texttt{LazyProcess}(\theta)}_{\text{configure}}
   \;\xrightarrow{\;c\;}
   \underbrace{\texttt{Process}(\theta, c)}_{\text{bind}}
   \;\xrightarrow{\;S\;}
   x \sim p_\theta(\cdot \mid c).

.. code-block:: python

   fm = FlowMatching(field, base, solver)   # configure
   process = fm(context)                     # bind c -> FlowMatchingProcess
   samples = process.sample((n,))            # draw

``LazyProcess`` wraps any matching family (flow matching, diffusion,
generator matching). ``LazyDistribution`` serves the same role for source
distributions whose parameters depend on context, carry trainable weights,
or require batch-shape inference at bind time. Concrete distributions (e.g.
a fixed ``StandardNormal``) need no lazy wrapper.

The field itself still follows the usual ``forward(x, t, c=None)``
contract. Lazy binding does not change how a conditional network is
written; it changes when context is attached at runtime.

Without a bound process, conditional and unconditional execution often
forces extra branching into training or sampling code:

.. code-block:: python

   if conditional:
       v = field(x, t, context)
       samples = sampler(field, context)
   else:
       v = field(x, t)
       samples = sampler(field)

With ``LazyProcess``, the model keeps the same field signature, but the
runtime object absorbs the conditioning step:

.. code-block:: python

   fm = FlowMatching(field, base, solver)  # configure once

   process = fm(context)                   # bind context once
   samples = process.sample((n,))          # then sample normally

This keeps conditional and unconditional workflows structurally close:
the field defines *how* context enters the network, while the process
defines *when* that context is bound for sampling or likelihood
evaluation.

Workflows
---------

Each workflow composes the same primitives differently.

.. list-table::
   :header-rows: 1
   :widths: 28 32 20 20

   * - Workflow
     - Loss
     - Sampler
     - Solver
   * - Deterministic FM
     - :func:`~nami.regression_loss`
     - :class:`~nami.FlowMatching`
     - :class:`~nami.RK4` / :class:`~nami.Heun`
   * - Consistency FM
     - :func:`~nami.consistency_loss`
     - :class:`~nami.ConsistencyFlowMatching`
     - (single-step; solver only for :meth:`log_prob`)
   * - Stochastic FM
     - :func:`~nami.stochastic_fm_loss`
     - :class:`~nami.FlowMatching`
     - :class:`~nami.RK4` / :class:`~nami.EulerMaruyama`
   * - Diffusion
     - (external)
     - :class:`~nami.Diffusion`
     - :class:`~nami.EulerMaruyama` / :class:`~nami.DPMSolverPP`
   * - Generator matching
     - :func:`~nami.regression_loss` / :func:`~nami.cgm_loss`
     - :class:`~nami.GeneratorMatching`
     - :class:`~nami.RK4` / :class:`~nami.EulerMaruyama`

Choosing a model
----------------

Not sure which workflow fits your problem? Use this decision guide:

.. admonition:: Quick decision guide

   - **"I want standard flow matching."**
     Use :func:`~nami.regression_loss` + :class:`~nami.FlowMatching` with
     :class:`~nami.LinearInterpolant`. This is the simplest setup and a
     good default starting point.

   - **"I want fast (few-step) sampling."**
     Use :func:`~nami.consistency_loss` + :class:`~nami.ConsistencyFlowMatching`.
     Same path and field as standard FM, but the consistency loss enables
     single-step generation. In nami this workflow is still experimental, and
     the most reliable local recipe starts from a strong FM baseline.

   - **"I want to add stochasticity to the interpolant."**
     Use :func:`~nami.stochastic_fm_loss` with :class:`~nami.BrownianGamma`.
     Same sampling process, but the noise can improve training stability.

   - **"I have a pretrained diffusion model (or want schedule-based noising)."**
     Use :class:`~nami.Diffusion` with a schedule (:class:`~nami.VPSchedule`,
     :class:`~nami.VESchedule`, or :class:`~nami.EDMSchedule`).
     Training uses an external denoising objective; nami handles sampling.

   - **"I want to learn drift and diffusion jointly (operator-centric)."**
     Use :func:`~nami.regression_loss` with :func:`~nami.generator_prediction`
     (or :func:`~nami.cgm_loss`) + :class:`~nami.GeneratorMatching`.
     Generator matching subsumes the other models as special cases, but this is
     still a research-oriented part of the library.

For a detailed comparison with code, see :doc:`models`.

How the pieces fit together
---------------------------

Every nami workflow composes the same four kinds of object:

.. code-block:: text

   ┌─────────┐     ┌──────┐     ┌────────┐     ┌─────────────┐
   │  Field  │────▶│ Loss │     │ Solver │     │ Distribution│
   │ (neural │     │      │     │        │     │  (base/     │
   │  net)   │     └──────┘     └───┬────┘     │   source)   │
   └────┬────┘                      │          └──────┬──────┘
        │         ┌─────────────┐   │                 │
        └────────▶│ LazyProcess │◀──┘─────────────────┘
                  │ (configure) │
                  └──────┬──────┘
                         │ bind context
                         ▼
                  ┌─────────────┐
                  │   Process   │──▶ .sample(), .log_prob()
                  └─────────────┘

- **Field** — the neural network being trained (velocity, score, or generator parameters).
- **Loss** — the training objective that regresses the field against path-derived targets.
- **Solver** — integrates the learned ODE/SDE at sampling time.
- **Distribution** — the source distribution at :math:`t=1` (usually standard normal).
- **LazyProcess** — bundles these pieces; calling it with context produces a runnable **Process**.

See also
--------

- :doc:`models` -- detailed description of every supported model.
- :doc:`components` -- component catalog with formal definitions.
- :doc:`api/index` -- full module-level API reference.
