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
   field = nami.GeneratorField(dim=dim, param_shape=operator.parameter_shape)
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
       field, operator, nami.RK4(steps=50), event_shape=(dim,),
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
