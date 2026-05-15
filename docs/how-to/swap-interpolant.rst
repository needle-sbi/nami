Swap the interpolant
====================

The default linear interpolant
:math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}` can be
replaced with any other interpolant that exposes the same protocol. The
two built-in choices are :class:`~nami.LinearInterpolant` and
:class:`~nami.CosineInterpolant`. Pass either via the ``interpolant``
keyword on the loss; nothing else changes.

.. code-block:: python

   import nami

   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.CosineInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )

The cosine interpolant is smooth at the endpoints, which can help when the
endpoint geometry is awkward; the linear interpolant gives a constant
conditional velocity and is the simpler default. See
:doc:`../explanation/core-abstractions` for the role of the interpolant in
the wider pipeline.
