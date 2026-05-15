Condition on external context
=============================

Conditioning enters nami via the field's third argument ``c``, and the lazy
process binds it once per call. The built-in fields support conditioning
natively — there is no separate ``ConditionalField`` class to subclass —
so most workflows are a one-line change to the field constructor.

The default recipe
------------------

For low-dimensional context (a summary statistic, a parameter vector, a
class embedding), pass ``condition_dim=d_c`` to
:class:`~nami.VelocityField`. The field then expects a context tensor of
shape ``(B, d_c)`` at call time, which the lazy process binds for you.

.. code-block:: python

   import torch, nami

   field = nami.VelocityField(dim=8, condition_dim=4)
   base  = nami.StandardNormal(event_shape=(8,))
   fm    = nami.FlowMatching(field, base, nami.RK4(steps=32))

   context = torch.randn(16, 4)            # 16 conditioning vectors
   samples = fm(context).sample((50,))     # (50, 16, 8) — sample, batch, event

The output shape follows the ``(sample, batch, event)`` convention: with
``sample((S,))`` and a context of batch shape ``(B,)`` the result has
shape ``(S, B, E)`` — one trajectory per context, ``S`` independent samples
per trajectory. The same field also works unconditionally: omit
``condition_dim`` and call ``fm()`` (or ``fm(None)``) at sampling time.

Training uses the same loss functions as the unconditional case; pass the
context through the loss call and the field will receive it on every
forward pass:

.. code-block:: python

   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       c=context,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )
   loss.backward()

Picking between the built-in fields
-----------------------------------

nami ships three conditional fields, distinguished by how context enters
the network. :class:`~nami.VelocityField` is an MLP that concatenates the
context vector to its input; it is the cheapest option and the right
default for low-dimensional ``c`` with shallow-to-moderate networks.
:class:`~nami.AdaLNVelocityField` is also an MLP but re-injects ``t`` and
``c`` at every residual block via adaLN-zero modulation, which is the
conventional remedy when input concatenation washes out with depth.
:class:`~nami.TransformerVelocityField` projects the context to a single
cross-attention key/value token and is the natural choice when the data
event is itself token-shaped or when richer attention-based conditioning
is wanted. The choice between them is a constructor swap — the loss,
process, solver, and sampling code are unchanged.

A reasonable default progression: start with :class:`~nami.VelocityField`,
move to :class:`~nami.AdaLNVelocityField` when the network grows past a
few layers or when ``t``-ablation diagnostics suggest the time signal is
being lost, and reach for :class:`~nami.TransformerVelocityField` when
attention is genuinely required.

When the built-in fields are not enough
---------------------------------------

A field is anything that exposes ``forward(x, t, c=None)`` and declares an
integer ``event_ndim`` property. Writing your own is straightforward when
the built-in choices do not fit your conditioning modality (image-shaped
``c``, multimodal context, classifier-free guidance setups, and so on).

One thing to be aware of when rolling your own conditional architecture:
concatenating ``t`` and ``c`` only at the input layer is a known weak
pattern. The signals tend to be washed out by depth, with the network
either ignoring the context or collapsing onto the unconditional flow.
The conventional remedies are to re-inject ``t`` and ``c`` at every block
— for example via FiLM or adaLN modulation, GLU gating, or cross-attention.
:class:`~nami.AdaLNVelocityField` and :class:`~nami.TransformerVelocityField`
are working references for the modulation and cross-attention halves of
that pattern respectively.

See also
--------

- :doc:`../explanation/lazy-binding` — what the lazy process is doing
  when it binds ``c`` and why the field stays a pure function of
  ``(x, t, c)``.
- :doc:`../explanation/core-abstractions` — where the field sits among
  the other primitives.
