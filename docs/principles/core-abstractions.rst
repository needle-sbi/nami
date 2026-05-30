Core abstractions
=================

nami is built from a small set of objects that compose along orthogonal axes,
rather than a tall class hierarchy. Once you have a feel for what each one
does and where the seams between them live, the rest of the library reads as
combinations of these pieces.

The pieces
----------

A **field** is a neural network with the contract
``forward(x, t, c=None)``. It is the only object that holds learnable
parameters. Fields predict whatever quantity the chosen training objective
regresses against; a velocity, a score, raw noise, or a vector of operator
parameters, and they declare an integer ``event_ndim`` so that the rest of
the stack can recover the sample/batch/event split of any tensor without
guessing.

An **interpolant** is a deterministic recipe for the
distribution :math:`p_t` that bridges source and target. It supplies a
conditional sample :math:`X_t` and the conditional target the field is
supposed to regress against; the linear interpolant, for instance, gives
:math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}` and the
constant velocity :math:`x_{\mathrm{data}} - x_{\mathrm{noise}}`. Swapping
the interpolant changes the geometry the field is asked to learn while
leaving every other piece of the pipeline untouched.

A **loss** ties a field to an interpolant. It samples a time, evaluates the
interpolant's conditional target, evaluates the field, and returns a
regression scalar. Losses are pure functions — they do not own state and do
not bind context — which is what lets the same field be reused across very
different objectives. :func:`~nami.regression_loss` and
:func:`~nami.stochastic_fm_loss` both hand the same velocity field
different conditional targets; the field never sees the difference.

A **process** is the runtime object that knows how to actually move samples
around. This is done either by integrating an ODE or SDE from the source distribution,
or by evaluating a one-step consistency function. Processes are the only
place where the time direction is load-bearing: the loss is direction-free,
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

The composition is almost embarrassingly direct: build a field, pick a loss,
train; then bundle the trained field with a distribution and a solver into a
lazy process and call it. The same field can be reused across processes; the
same solver can be reused across families; the same loss can be applied to
any field with the right output signature. Nothing is welded to anything
else.

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

This is why "flow matching", "diffusion", and "generator matching" do not
appear as deep class hierarchies inside the library. They are not
fundamentally different kinds of object; they are different (interpolant,
field-output, loss, process) tuples over the same primitives. The loss is
worth naming as its own axis even though it is a pure function: two
families that predict the same quantity along the same path can still
differ in how that prediction is identified from data, and many of the
interesting research directions live exactly there. See
:doc:`model-families` for the longer version of that argument.

Tensor layout
-------------

All tensors follow a ``(sample, batch, event)`` convention. The leading
sample axis indexes independent draws, the batch axis indexes parallel
computations that share the same model parameters (typically one entry per
conditioning context), and the trailing event axes describe a single data
point. The integer property ``event_ndim`` on each field declares how many
trailing dimensions count as event; everything to the left is treated as
sample times batch. Most users never have to think about this — it is
plumbing that lets the same field handle conditional and unconditional
calls without reshape boilerplate.

Within a process, ``x_data`` is the endpoint at :math:`t=1` and ``x_noise``
is the endpoint at :math:`t=0`; sampling integrates from :math:`t=0` up to
:math:`t=1`. This is a library-level choice that keeps flow matching,
consistency FM, stochastic FM, and generator matching on the same clock.
The :class:`~nami.Diffusion` process is the exception, since it retains the
diffusion-native orientation (data at small time, noise at large time)
because the score-based reverse-time PF-ODE is intrinsic to that direction.
In running prose elsewhere we let the variable names — ``x_data`` and
``x_noise`` — carry the semantic, so that arguments about how the field
behaves do not have to name endpoint values.

See also
--------

- :doc:`parameterizations`: what a field can predict, and why several
  options exist.
- :doc:`lazy-binding`: what ``LazyProcess`` and ``LazyDistribution`` buy
  you, and when to reach for them.
- :doc:`model-families`: how flow matching, diffusion, consistency, and
  generator matching are four views of one transport-map object.
