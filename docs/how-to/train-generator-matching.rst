Train a generator-matching model
================================

Learn Itô generator parameters :math:`(b_t, a_t)` via conditional
regression. The field predicts raw operator parameters; the operator
interprets them as drift (and optionally diffusion). With
``diffusion="none"``, the operator produces drift only and sampling is a
plain ODE; this configuration recovers deterministic flow matching as a
special case.

.. code-block:: python

   import torch
   import nami

   dim = 8
   operator = nami.ItoGeneratorOperator(event_shape=(dim,), diffusion="none")
   field = nami.GeneratorField(dim, operator=operator)
   interpolant = nami.LinearInterpolant()
   parameterization = nami.generator_prediction(operator)
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

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
       field, nami.RK4(steps=50),
       parameterization=parameterization, event_shape=(dim,),
   )
   samples = gm(None).sample((128,))

For Brownian-bridge interpolants with SDE sampling, swap the interpolant
and operator mode and use :class:`~nami.EulerMaruyama`:

.. code-block:: python

   operator = nami.ItoGeneratorOperator(event_shape=(dim,), diffusion="diagonal")
   interpolant = nami.BrownianBridgeInterpolant(sigma=0.1)
   solver = nami.EulerMaruyama(steps=100)

The operator mode and the solver must agree. ``diffusion="none"`` is an
ODE (use :class:`~nami.RK4` or :class:`~nami.Heun`); ``diffusion="diagonal"``
is an SDE (use :class:`~nami.EulerMaruyama`). Mixing these up will raise an
error or produce incorrect samples.

The unified :func:`~nami.regression_loss` with
:func:`~nami.generator_prediction(operator)` regresses directly against
conditional path-derived operator parameters. The marginal generator-
matching objective shares gradients under marginalisation, so the
conditional form is the practical choice.

The Conditional Generator Matching (CGM) loss
----------------------------------------------

:func:`~nami.regression_loss` matches the conditional generator with a single
mean-squared error over the whole packed parameter tensor. That is the
squared-:math:`L_2` instance of the *Conditional Generator Matching* loss
(Holderrieth et al., 2024, Eq. 17), whose gradient identity with the marginal
objective holds for any **Bregman divergence** :math:`D`, and not just MSE.
:func:`~nami.cgm_loss` exposes that choice and applies the divergence **per
generator component** (drift, diffusion, jump rates), summing the results:

.. code-block:: python

   loss = nami.cgm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=interpolant,
       parameterization=nami.generator_prediction(operator),
       # divergence=None -> operator.default_divergence()
   )

For an Itô operator the default is squared-:math:`L_2` on every component, so a
drift-only (ODE) generator gives exactly the same loss as
:func:`~nami.regression_loss`. Pass ``divergence=`` a
:class:`~nami.BregmanDivergence` (or a ``{component: divergence}`` mapping) to
match a non-Euclidean component on its proper domain, e.g.
:class:`~nami.KLDivergence` on the probability simplex,
:class:`~nami.ItakuraSaito` on the positive orthant. Matching a simplex-valued
target with MSE silently breaks the gradient identity; choosing :math:`D` to
fit the domain is structural and necessary.

Masking CTMC: a discrete pure-jump generator
---------------------------------------------

The KL divergence is the natural objective for the masking
:class:`~nami.CTMCGeneratorOperator`, a discrete-state (pure-jump) generator
that unmasks tokens. The network is a categorical denoiser; sampling runs the
:class:`~nami.TauLeapingSampler` from the all-mask state.

.. code-block:: python

   K, d = 8, 16  # 8 data tokens, 16 coordinates
   operator = nami.CTMCGeneratorOperator(num_states=K, event_shape=(d,))
   field = nami.CTMCField(operator, hidden=256, layers=3)
   interpolant = nami.MaskingInterpolant(operator)
   parameterization = nami.generator_prediction(operator)  # softmax projection
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

   for _ in range(1000):
       x_data = sample_tokens(256, d, K)              # long indices in [0, K)
       x_noise = torch.full_like(x_data, operator.mask_index)
       loss = nami.cgm_loss(                           # KL/cross-entropy by default
           field,
           x_noise=x_noise, x_data=x_data,
           interpolant=interpolant,
           parameterization=parameterization,
           eps_t=0.0,
       )
       optim.zero_grad(); loss.backward(); optim.step()

   gm = nami.GeneratorMatching(
       field, nami.TauLeapingSampler(steps=50),
       parameterization=parameterization,
       base=nami.AllMask((d,), mask_index=operator.mask_index),
       event_shape=(d,),
   )
   samples = gm().sample((128,))  # long token tensors, fully unmasked

The state space is ``num_states`` data tokens plus an absorbing ``MASK`` token at
index ``num_states``; :class:`~nami.AllMask` draws the fully-masked starting
state. With the default linear masking schedule this recovers masked-diffusion
training. The CGM loss here supervises every coordinate toward its clean token.
Production masked-diffusion typically restricts the sum (and applies an
:math:`\alpha`-dependent weighting) to the masked positions.
