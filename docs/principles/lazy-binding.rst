Lazy binding
============

Many transport-map workflows have to thread conditioning context through
training and sampling. The naive way to do this is to branch: pass context
when present, omit it when absent, and write each call site twice.
``LazyProcess`` and ``LazyDistribution`` exist so that you do not have to.

The idea is to separate the moment a process is *configured* — when its
field, distribution, and solver are decided — from the moment it is *bound*
to a concrete piece of context. Configuration happens once, usually at
training-script construction time. Binding happens at each call to the
runtime object. The field's own signature does not change; the lazy wrapper
absorbs the conditioning step so that the same sampling code runs whether
context is present, absent, or different on every call.

The configure-then-bind idiom
-----------------------------

Reading the pattern as a small diagram makes the two phases obvious:

.. code-block:: python

   fm = nami.FlowMatching(field, base, solver)   # configure
   process = fm(context)                          # bind
   samples = process.sample((n,))                 # draw

The configured ``fm`` is reusable: bind it to one context to produce one
batch of conditional samples, bind it to a different context for another
batch, or bind to ``None`` for the unconditional case. The same pattern
works for diffusion, consistency flow matching, and generator matching;
every family in the library exposes a ``LazyProcess`` as its public sampling
entry point.

Without lazy binding, conditional and unconditional execution often forces
extra branching into training or sampling code. With it, the field defines
*how* context enters the network and the process defines *when* that context
is bound for sampling or likelihood evaluation, and the two responsibilities
no longer leak into each other.

.. note::

   A planned extension of nami will include mechanisms by which
   the binding of the context can be inlcuded by different and 
   more robust mechanisms, e.g. via a context encoder, gated conditioning, etc.

When binding matters
--------------------

For an unconditional model, binding is simply a no-op, so ``fm()`` and ``fm(None)``
both produce a runnable process. For a conditional model, binding is where
the context tensor's batch shape is read; passing a context of shape
``(B, d_c)`` produces a process whose ``.sample((S,))`` returns a tensor of
shape ``(S, B, E)``. The library is consistent about this convention so that
sample-axis and batch-axis arithmetic does not require manual reshapes.

Binding also matters for sources whose parameters depend on context.
:class:`~nami.lazy.LazyDistribution` is the analog of ``LazyProcess`` for
distributions: it wraps any source whose parameters are not fixed at
configuration time. Examples include a Gaussian whose mean is a learned
function of the conditioning vector, or a source whose batch shape is
inferred from the bound context. Fixed sources like
:class:`~nami.StandardNormal` need no lazy wrapper and are bound trivially.

Conceptually
------------

The split is a design statement: the field is a pure function of
``(x, t, c)``; the loss is a pure function of ``(field, x_noise, x_data)``;
the lazy process is a small protocol that says "here is everything you need
to sample once you tell me what to condition on".

See also
--------

- :doc:`core-abstractions`: where the lazy process sits in the stack.
- :doc:`parameterizations`: how the bound process interprets the field's
  output.
