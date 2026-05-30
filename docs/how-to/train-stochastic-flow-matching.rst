Train with the stochastic interpolant
=====================================

Replace the deterministic interpolant with a stochastic one,
:math:`X_t = I_t + \gamma(t)\,Z`, following the stochastic-interpolants
framework. The added noise can act as a training-time regulariser; at
sampling time the same :class:`~nami.FlowMatching` process is used.

The only change from deterministic FM is the loss function and the gamma
schedule. :class:`~nami.BrownianGamma` is a sensible default; if training
is unstable, try :class:`~nami.ScaledBrownianGamma` with a smaller scale
(e.g. 0.5). :class:`~nami.ZeroGamma` recovers deterministic FM exactly and
is useful as a sanity check.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_data = torch.randn(32, 8)
   x_noise = torch.randn_like(x_data)

   loss = nami.stochastic_fm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       gamma=nami.BrownianGamma(),
   )
   loss.backward()

For SDE sampling from a stochastic-FM-trained field, pair it with a score
estimate via :class:`~nami.DriftFromVelocityScore` and integrate with
:class:`~nami.EulerMaruyama`.
