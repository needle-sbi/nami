Train a consistency flow-matching model
=======================================

Train the same velocity field as flow matching, but with a consistency loss
that enables single-step sampling. The trade-off is a one-step approximation
in exchange for skipping the ODE solve at inference.

This is a minimal API sketch, not the recommended paper-faithful training
recipe. In local nami experiments, consistency training is best treated as
an experimental one-step approximation layer on top of a strong flow-matching
baseline. The original paper (Yang et al., 2024) trains from scratch with
EMA targets and additional schedule details that nami does not currently
ship.

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

   cfm = nami.ConsistencyFlowMatching(field, base)
   process = cfm()
   samples = process.sample((128,))    # single-step sampling
   z = process.invert(samples)         # single-step inversion (data -> noise)

Two symmetric consistency losses are combined: :func:`~nami.consistency_loss`
with ``target_time=1.0`` anchors at the data endpoint (forward
consistency) via :math:`f(x_t, t) = x_t + (1 - t) \cdot v_\theta`, and
with ``target_time=0.0`` anchors at the noise endpoint (reverse
consistency) via :math:`g(x_t, t) = x_t - t \cdot v_\theta`. Target outputs are
stop-gradiented by default; an optional ``target_field`` (e.g. an EMA copy)
can be passed for stability.

The ``delta`` parameter controls how close the two trajectory points are;
smaller values give tighter consistency but noisier gradients. ``0.01`` is
a reasonable default. Passing ``euler_step=True`` places the second
trajectory point on the learned ODE trajectory rather than the conditional
path, which eliminates trajectory-mismatch variance.

To get one-step log-densities, add a log-prob head:

.. code-block:: python

   h_head = nami.ConsistencyHead(dim=dim)
   loss_h = nami.log_density_consistency_loss(
       field, h_head,
       x_noise=x_noise, x_data=x_data,
       interpolant=interpolant,
       delta=0.01,
   )
   # train field + h_head jointly with (loss_f + loss_g + 0.1 * loss_h)

   cfm = nami.ConsistencyFlowMatching(field, base, h_head=h_head)
   log_p = cfm().log_prob(samples)     # one-step log-density

Unlike :func:`~nami.consistency_loss` and :func:`~nami.regression_loss`,
:func:`~nami.log_density_consistency_loss` does not accept a
``parameterization=`` kwarg: it consumes the field's raw output as a
velocity for the divergence term, so it is fixed to the
:class:`~nami.parameterizations.Velocity` parameterization. For models
trained under other parameterizations, fall back to exact-likelihood ODE
integration by passing a solver to ``ConsistencyFlowMatching`` and
calling ``log_prob(x, ode=True, ...)``.

For exact log-likelihood, fall back to ODE integration by passing a solver
to ``ConsistencyFlowMatching`` and calling ``log_prob(x, ode=True, ...)``.
