Train a flow-matching model
===========================

Standard deterministic flow matching: regress a velocity field against the
constant conditional velocity along a linear interpolant, then sample by
integrating the learned ODE.

This is the default starting point for almost any transport-map problem in
nami. If you are not sure which family to use, use this one. Other recipes
(stochastic, consistency, diffusion, generator matching) are useful when
their specific properties — noise regularisation, one-step sampling,
schedule-induced noising, operator-level expressivity — are wanted; see
:doc:`../explanation/model-families` for the comparison.

.. code-block:: python

   import torch
   import nami

   dim = 8
   field = nami.VelocityField(dim=dim)
   base = nami.StandardNormal(event_shape=(dim,))
   solver = nami.RK4(steps=50)
   optim = torch.optim.Adam(field.parameters(), lr=1e-3)

   interpolant = nami.LinearInterpolant()
   parameterization = nami.velocity_prediction()

   for step in range(500):
       x_data = torch.randn(256, dim)         # data
       x_noise = torch.randn_like(x_data)     # noise
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

   fm = nami.FlowMatching(field, base, solver)
   samples = fm(None).sample((128,))           # (128, 8)

Each training iteration samples random times internally and regresses the
field against the conditional velocity
:math:`u_t = x_{\mathrm{data}} - x_{\mathrm{noise}}`. After training,
``FlowMatching`` is a lazy process — calling ``fm(None)`` binds it with no
conditioning context and returns a runnable process; see
:doc:`../explanation/lazy-binding`.
