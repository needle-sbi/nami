Core abstractions
=================

Nami is built from a small set of objects that compose along orthogonal axes. 
Once you have a feel for what each one does and where the seams between them live, 
the rest of the library reads as combinations of these pieces.

The pieces
----------

A **field** is a neural network with the following interface:
``forward(x, t, c=None)``. It is the only object that holds learnable
parameters. Fields predict whatever quantity the chosen training objective
regresses against; a velocity, a score, raw noise, or a vector of operator
parameters, and they declare an integer ``event_ndim`` so that the rest of
the sample/batch/event split of any tensor can be inferred.

An **interpolant** is a deterministic path for the
distribution :math:`p_t` that bridges source and target. It supplies a
conditional sample :math:`X_t` and the conditional target that the field
regresses against; the linear interpolant, for instance, gives
:math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}` and the
constant velocity :math:`x_{\mathrm{data}} - x_{\mathrm{noise}}`. Swapping
the interpolant thus changes the geometry the field learns.

A **loss** ties a field to an interpolant. It samples a time, evaluates the
interpolant's conditional target, evaluates the field, and returns a
regression scalar. Losses are pure functions, meaningthey do not own state and do
not bind context, which lets the same field be reused across very
different objectives. :func:`~nami.regression_loss` and
:func:`~nami.stochastic_fm_loss` both hand the same velocity field
different conditional targets; the field never sees the difference.

A **process** is the runtime object that knows how to actually move samples
around. This is done either by integrating an ODE or SDE from the source distribution,
or by evaluating a one-step consistency function. Processes are the only
place where the time direction is meaningful: the loss is direction-free,
the field is direction-free, but the process integrates.

A **solver** is the numerical scheme a process uses. ODE solvers
(:class:`~nami.RK4`, :class:`~nami.Heun`, :class:`~nami.DPMSolverPP`) and
SDE solvers (:class:`~nami.EulerMaruyama`) are interchangeable wherever the
process supports them; choosing a solver is independent of choosing a field
or a loss.

A **distribution** is the source the process draws from. For most workflows
it is a fixed :class:`~nami.StandardNormal`, but anything that exposes the
same sampling and log-density interface is acceptable.

How they compose
----------------

The composition is performed as follows: build a field, pick a loss,
train, then bundle the trained field with a distribution and a solver into a
lazy process and call it. The same field can be reused across processes. The
same solver can be reused across families. The same loss can be applied to
any field with the right output signature.

.. code-block:: python

   import torch, nami

   field = nami.VelocityField(dim=8)
   x_data = torch.randn(32, 8)
   x_noise = torch.randn_like(x_data)

   # train: a loss + a field
   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )
   loss.backward()

   # sample: a process bundles the field with a distribution and a solver
   fm = nami.FlowMatching(field, nami.StandardNormal((8,)), nami.RK4(steps=50))
   samples = fm().sample((64,))

This is why flow matching, diffusion, and generator matching do not
appear as deep class hierarchies inside the library. They are not
fundamentally different kinds of object; they are different (interpolant,
field-output, loss, process) tuples over the same primitives. The loss is
worth identifying as its own axis even though it is a pure function: two
families that predict the same quantity along the same path can still
differ in how that prediction is identified from data, and many of the
interesting research directions follow on from this fact. See
:doc:`model-families` for the longer version of that argument.

Tensor layout
-------------

All tensors follow a ``(sample, batch, event)`` convention. The leading
sample axis indexes independent draws, the batch axis indexes parallel
computations that share the same model parameters (typically one entry per
conditioning context), and the trailing event axes describe a single data
point. The integer property ``event_ndim`` on each field declares how many
trailing dimensions count as event; everything to the left is treated as
sample times batch.

``event_ndim`` is used to inform where a single data point begins, when counting from the right.
Everything to its left, however many axes, is leading (sample times batch). Nami never has to 
distinguish the two when peeling off the event dimensions. Concretely, a field's ``forward`` flattens
only the trailing ``event_ndim`` axes and treats the rest as one leading
block::

    x_flat = flatten_event(x, self.event_ndim)   # collapses the last event_ndim axes
    lead   = x_flat.shape[:-1]                    # all remaining axes, together

Worked example
~~~~~~~~~~~~~~

Suppose ``x`` has shape ``(32, 500, 2)`` and one data point is a 2-vector.
Then ``event_shape = (2,)`` and ``event_ndim = 1``. One axis forms a
sample, and the leading ``(32, 500)`` is sample times batch. Nami does not
care how that leading block is split into "32 samples" vs "500 batch";
only ``event_ndim`` matters.

The same tensor rank can imply different ``event_ndim``, which is why the
property cannot be inferred from the shape alone and must be declared:

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Your data point is
     - ``event_shape``
     - a tensor might look like
     - ``event_ndim``
   * - a scalar
     - ``()``
     - ``(32, 500)``
     - ``0``
   * - a 2-vector
     - ``(2,)``
     - ``(32, 500, 2)``
     - ``1``
   * - a 5-vector
     - ``(5,)``
     - ``(64, 5)``
     - ``1``
   * - a 3 x 8 x 8 image
     - ``(3, 8, 8)``
     - ``(32, 500, 3, 8, 8)``
     - ``3``

Note the last two rows of leading dims differ in count yet ``event_ndim``
ignores that, and note that ``(32, 500, 2)`` and ``(32, 500, 3, 8, 8)`` are
distinguished only by where the event begins. For some bare tensor, nami
genuinely cannot tell whether ``(32, 500, 2)`` is ``(32, 500)`` leading x
``(2,)`` events or ``(32,)`` leading x ``(500, 2)`` events. Declaring
``event_ndim`` is the field author resolving that ambiguity.

When a field does expose a concrete ``event_shape`` (not just an
``event_ndim``), nami validates the full shape against the base
distribution at process-construction time, so a mismatch such as
``VelocityField(8)`` paired with ``StandardNormal((4,))`` raises
immediately rather than surviving until ``sample()``; both have rank
``1``, so a rank-only check would miss it. Genuinely conditional fields,
whose shape is unknown until a context is bound, skip this eager check and
fall back to the rank check at bind time. Pass ``validate_args=False`` to a
process to opt out of validation entirely.

``event_ndim`` is used to inform where a single data point begins, when counting from the right.
Everything to its left, however many axes, is leading (sample times batch). Nami never has to 
distinguish the two when peeling off the event dimensions. Concretely, a field's ``forward`` flattens
only the trailing ``event_ndim`` axes and treats the rest as one leading
block::

    x_flat = flatten_event(x, self.event_ndim)   # collapses the last event_ndim axes
    lead   = x_flat.shape[:-1]                    # all remaining axes, together

Worked example
~~~~~~~~~~~~~~

Suppose ``x`` has shape ``(32, 500, 2)`` and one data point is a 2-vector.
Then ``event_shape = (2,)`` and ``event_ndim = 1``. One axis forms a
sample, and the leading ``(32, 500)`` is sample times batch. Nami does not
care how that leading block is split into "32 samples" vs "500 batch";
only ``event_ndim`` matters.

The same tensor rank can imply different ``event_ndim``, which is why the
property cannot be inferred from the shape alone and must be declared:

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Your data point is
     - ``event_shape``
     - a tensor might look like
     - ``event_ndim``
   * - a scalar
     - ``()``
     - ``(32, 500)``
     - ``0``
   * - a 2-vector
     - ``(2,)``
     - ``(32, 500, 2)``
     - ``1``
   * - a 5-vector
     - ``(5,)``
     - ``(64, 5)``
     - ``1``
   * - a 3 x 8 x 8 image
     - ``(3, 8, 8)``
     - ``(32, 500, 3, 8, 8)``
     - ``3``

Note the last two rows of leading dims differ in count yet ``event_ndim``
ignores that, and note that ``(32, 500, 2)`` and ``(32, 500, 3, 8, 8)`` are
distinguished only by where the event begins. For some bare tensor, nami
genuinely cannot tell whether ``(32, 500, 2)`` is ``(32, 500)`` leading x
``(2,)`` events or ``(32,)`` leading x ``(500, 2)`` events. Declaring
``event_ndim`` is the field author resolving that ambiguity.

When a field does expose a concrete ``event_shape`` (not just an
``event_ndim``), nami validates the full shape against the base
distribution at process-construction time, so a mismatch such as
``VelocityField(8)`` paired with ``StandardNormal((4,))`` raises
immediately rather than surviving until ``sample()``; both have rank
``1``, so a rank-only check would miss it. Genuinely conditional fields,
whose shape is unknown until a context is bound, skip this eager check and
fall back to the rank check at bind time. Pass ``validate_args=False`` to a
process to opt out of validation entirely.

Within a process, ``x_data`` is the endpoint at :math:`t=1` and ``x_noise``
is the endpoint at :math:`t=0`; sampling integrates from :math:`t=0` up to
:math:`t=1`. The :class:`~nami.Diffusion` process is the exception, 
since it retains the diffusion-native orientation (data at small time, 
noise at large time) because the score-based reverse-time PF-ODE is 
intrinsic to that direction. In running prose elsewhere we let the 
variable names (``x_data`` and ``x_noise``) carry the semantic, 
so that arguments about how the field behaves do not have to name endpoint 
values.

See also
--------

- :doc:`parameterizations`: what a field can predict, and why several
  options exist.
- :doc:`lazy-binding`: what ``LazyProcess`` and ``LazyDistribution`` buy
  you, and when to reach for them.
- :doc:`model-families`: how flow matching, diffusion, consistency, and
  generator matching are four views of one transport-map object.
