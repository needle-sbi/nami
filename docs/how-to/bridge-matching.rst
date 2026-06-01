Train a Schrödinger bridge model
================================

Learn separate velocity and score fields jointly along a Brownian-bridge
path, then reconstruct the SDE drift from the two fields at sampling time.
This follows the bridge-matching approach of Tong et al.

.. code-block:: python

   import nami

   loss = nami.losses.bridge_matching_loss(
       flow_field, 
       score_field,
       x_noise=x_noise, 
       x_data=x_data,
       interpolant=nami.BrownianBridgeInterpolant(),
   )
   loss.backward()

After training, reconstruct the SDE drift from the two fields and sample
with a standard :class:`~nami.FlowMatching` or :class:`~nami.Diffusion`
process. The reconstruction uses
:class:`~nami.DriftFromVelocityScore`, which combines the velocity and
score heads into the probability-flow / SDE drift.
