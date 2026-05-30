Verify stochastic FM reduces to deterministic FM
================================================

A useful sanity check: setting :math:`\gamma \equiv 0` in the stochastic
interpolant must recover the deterministic flow-matching objective exactly.
If this assertion ever fails, something is wrong with the installation.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_data = torch.randn(32, 8)
   x_noise = torch.randn(32, 8)

   det = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
       reduction="none",
   )
   stoch = nami.stochastic_fm_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       gamma=nami.ZeroGamma(),
       reduction="none",
   )
   assert torch.allclose(det, stoch, atol=1e-6)

This is the cleanest way to confirm that a stochastic-FM run reduces to a
deterministic-FM run when the noise schedule is turned off, and a useful
smoke test after any change to the loss internals.
