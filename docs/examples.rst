:orphan:

Examples
========

This page collects self-contained code examples. Each one is copy-pasteable
and includes commentary explaining *why* each step matters, not just what
it does.

Deterministic flow matching
---------------------------

The simplest workflow: regress a velocity field against the conditional
target along a linear interpolation path.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_noise, x_data = torch.randn(32, 8), torch.randn(32, 8)

   # regression_loss internally: samples time t ~ U(0,1), builds the interpolant
   # X_t = (1-t)*x_noise + t*x_data, and regresses field(X_t, t) against
   # (x_data - x_noise).
   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )
   loss.backward()

Stochastic flow matching
-------------------------

Adds Brownian bridge noise :math:`\gamma(t) = \sqrt{t(1-t)}` to the
interpolant. The noisier interpolant can improve training stability and
sample diversity; at sampling time, the same ``FlowMatching`` process is used.
This form of flow matching is particularly useful for high-dimensional and 
multi-modal data.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_noise, x_data = torch.randn(32, 8), torch.randn(32, 8)

   # The only change from deterministic FM is the loss function and the
   # gamma schedule
   loss = nami.stochastic_fm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       gamma=nami.BrownianGamma(),
   )

``ZeroGamma`` deterministic parity
-----------------------------------

A useful sanity check: setting :math:`\gamma \equiv 0` must recover the
deterministic objective exactly. If this assertion ever fails, something
is wrong with your installation.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_noise, x_data = torch.randn(32, 8), torch.randn(32, 8)

   det = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0, reduction="none",
   )
   stoch = nami.stochastic_fm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       gamma=nami.ZeroGamma(), reduction="none",
   )
   assert torch.allclose(det, stoch, atol=1e-6)

Generator matching
------------------


.. code-block:: python

    import nami

    operator = nami.ItoGeneratorOperator((2,), diffusion="diagonal")
    field = nami.GeneratorField((2,), operator=operator, condition_dim=4)
    interpolant = nami.BrownianBridgeInterpolant(sigma=0.5)

    loss = nami.regression_loss(
        field,
        x_noise=x_base,
        x_data=x_data,
        c=context,
        interpolant=interpolant,
        parameterization=nami.generator_prediction(operator),
        eps_t=0.0,
    )

    process = nami.GeneratorMatching(
        field,
        operator,
        nami.EulerMaruyama(steps=64),
        event_shape=(2,),
    )
    samples = process(context).sample((128,))

Conditional generation
----------------------

To condition on external information, give the field a ``condition_dim``.
:class:`~nami.VelocityField` then expects a context vector ``c`` and the
process layer handles the rest: the lazy process binds context once, and
all samples drawn from that process share the same conditioning. Any field
following the ``forward(x, t, c=None)`` / ``event_ndim`` contract works the
same way — but for a plain MLP you rarely need to write your own.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8, condition_dim=4)
   base = nami.StandardNormal(event_shape=(8,))
   solver = nami.RK4(steps=32)
   fm = nami.FlowMatching(field, base, solver)

   # Binding 16 context vectors produces 16 parallel trajectories.
   # sample((1,)) draws one sample per context -> shape (1, 16, 8).
   # sample((50,)) would draw 50 samples per context -> shape (50, 16, 8).
   context = torch.randn(16, 4)
   samples = fm(context).sample((1,))   # (1, 16, 8)

Score-based diffusion
---------------------

Forward process :math:`q(x_t \mid x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)`
with a VP schedule. The model predicts :math:`\varepsilon`; sampling reverses
the SDE. Both halves live in nami: ``regression_loss`` trains the
:math:`\varepsilon`-model when paired with a :class:`~nami.GaussianInterpolant`
(which supplies the forward process above) and
:func:`~nami.epsilon_prediction`, and :class:`~nami.Diffusion` reverses it.

.. note::

   Despite its name, :class:`~nami.VelocityField` is just the generic MLP
   vector field — what it emits is set by the ``parameterization``, so here
   it predicts :math:`\varepsilon`, not velocity. The same class backs the
   velocity, score, and :math:`x_0` examples on this page. This abstraction
   will be updated in the future to prevent confusion.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)

   # Train: the GaussianInterpolant builds q(x_t | x_0) from the schedule,
   # and epsilon_prediction(schedule) regresses field(x_t, t) against the
   # noise epsilon with the matching weighting.
   x_noise, x_data = torch.randn(32, 8), torch.randn(32, 8)
   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.GaussianInterpolant(schedule=schedule),
       parameterization=nami.epsilon_prediction(schedule=schedule),
   )
   loss.backward()

   # Sample: reverse the SDE with the same schedule and parameterization.
   diffusion = nami.Diffusion(
       model=field,
       schedule=schedule,
       solver=nami.EulerMaruyama(steps=100),
       parameterization=nami.epsilon_prediction(schedule=schedule),
       event_shape=(8,),
   )
   samples = diffusion().sample((64,))

Parameterisation transforms
----------------------------

Convert between interpolant score-related parameterisations, velocity, and
drift representations. These are useful when you have a model trained with
one interpolant parameterisation but need another for sampling.

.. code-block:: python

   import nami

   # If the model predicts eta = gamma(t) * score(x, t)
   score = nami.ScoreFromEta(eta_model, nami.BrownianGamma())

   # If the model predicts the raw Gaussian noise z in x_t = I_t + gamma(t) z
   score_from_noise = nami.ScoreFromRawNoise(noise_model, nami.BrownianGamma())

   # Combine velocity + score into an SDE drift
   drift = nami.DriftFromVelocityScore(v_model, score, nami.BrownianGamma())

   # Construct the mirror correction term used in backward-SDE formulas
   mirror_v = nami.MirrorVelocityFromScore(score, nami.BrownianGamma())

.. note::

   For diffusion :math:`\varepsilon`-prediction, use
   :class:`~nami.Diffusion` with
   ``parameterization=nami.epsilon_prediction(schedule=schedule)``
   rather than these interpolant wrappers. Diffusion uses :math:`-\varepsilon/\sigma`,
   while ``ScoreFromEta`` and ``ScoreFromRawNoise`` use interpolant-specific
   :math:`\gamma(t)`.
