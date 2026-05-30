Train a Schrödinger bridge model
================================

Learn separate velocity and score fields jointly along a Brownian-bridge
path, then reconstruct the SDE drift from the two fields at sampling time.
This follows the bridge-matching approach of Tong et al.

.. code-block:: python

   import nami

   loss = nami.bridge_matching_loss(
       flow_field, score_field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.BrownianBridgeInterpolant(),
   )
   loss.backward()

After training, reconstruct the SDE drift from the two fields and sample
with a standard :class:`~nami.FlowMatching` or :class:`~nami.Diffusion`
process. The reconstruction uses
:class:`~nami.DriftFromVelocityScore`; related helpers
(:class:`~nami.ScoreFromEta`, :class:`~nami.ScoreFromRawNoise`) cover the
case where the score field is parameterised through an interpolant
quantity rather than predicted directly.
