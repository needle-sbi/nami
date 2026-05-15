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

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_noise, x_data = torch.randn(32, 8), torch.randn(32, 8)

   # The only change from deterministic FM is the loss function and the
   # gamma schedule — the rest of the pipeline stays identical.
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

To condition on external information, build a field whose ``forward``
accepts a third argument ``c``. The lazy process binds context once;
all samples drawn from that process share the same conditioning.

.. code-block:: python

   import torch
   from torch import nn
   import nami

   class ConditionalField(nn.Module):
       """A minimal conditional velocity field.

       Concatenates x, t, and context c before passing through an MLP.
       Any architecture works — the only requirement is the signature
       ``forward(x, t, c=None)`` and an ``event_ndim`` property.
       """
       def __init__(self, dim: int, context_dim: int):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(dim + context_dim + 1, 128),
               nn.SiLU(),
               nn.Linear(128, dim),
           )

       @property
       def event_ndim(self) -> int:
           return 1

       def forward(self, x, t, c=None):
           t_exp = t.unsqueeze(-1).expand(*x.shape[:-1], 1)
           inputs = [x, t_exp] + ([c] if c is not None else [])
           return self.net(torch.cat(inputs, dim=-1))

   field = ConditionalField(dim=8, context_dim=4)
   base = nami.StandardNormal(event_shape=(8,))
   solver = nami.RK4(steps=32)
   fm = nami.FlowMatching(field, base, solver)

   # Binding 16 context vectors produces 16 parallel trajectories.
   # sample((1,)) draws one sample per context → shape (1, 16, 8).
   # sample((50,)) would draw 50 samples per context → shape (50, 16, 8).
   context = torch.randn(16, 4)
   samples = fm(context).sample((1,))   # (1, 16, 8)

Score-based diffusion
---------------------

Forward process :math:`q(x_t \mid x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)`
with a VP schedule. The model predicts :math:`\varepsilon`; sampling reverses
the SDE. Note that nami handles *sampling only* — you train the model with
your own denoising loss outside of nami.

.. code-block:: python

   import nami

   model = nami.VelocityField(dim=8)
   schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)
   solver = nami.EulerMaruyama(steps=100)

   diffusion = nami.Diffusion(
       model=model,
       schedule=schedule,
       solver=solver,
       parameterization=nami.epsilon_prediction(schedule=schedule),
       event_shape=(8,),
   )
   samples = diffusion(None).sample((64,))

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
