Model families as views of one object
=====================================

nami covers flow matching, diffusion, consistency flow matching, and
generator matching. From a distance these look like four separate methods;
from close up, they are four views of the same transport-map object. This
essay walks through what each view emphasises and what genuinely differs.

The shared object
-----------------

Every workflow in the library is built from the same primitive: a path
:math:`p_t` linking source and target, and a field that the training
objective regresses against a path-derived target. The differences between
the families are differences in *what* the field predicts, *what target*
the loss compares it to, and *what process* uses the trained field at
inference time. The primitives — interpolants, schedules, solvers,
distributions — are shared verbatim.

Flow matching
-------------

Deterministic flow matching trains a velocity field
:math:`v_\theta(x, t)` whose flow pushes the source distribution onto the
target. The default path is the linear (sometimes called optimal-transport)
interpolant:

.. math::

   X_t = (1-t)\,x_{\mathrm{target}} + t\,x_{\mathrm{source}},
   \qquad u_t = x_{\mathrm{source}} - x_{\mathrm{target}},

and the loss regresses :math:`v_\theta(X_t, t)` against this constant
conditional velocity. Sampling integrates the learned ODE from the source
distribution to produce samples. The standard objective is a regression
loss on path samples, so training does not solve an ODE or estimate a
divergence; that cost reappears only when you ask for log-densities.

Consistency flow matching
-------------------------

Consistency flow matching trains the same velocity field, but replaces the
regression loss with a *self-consistency* objective. Two points on the same
conditional trajectory must map to the same endpoint via a consistency
function :math:`f(x_t, t) = x_t + (1-t)\,v_\theta(x_t, t)` (toward the data
endpoint) or :math:`g(x_t, t) = x_t - t\,v_\theta(x_t, t)` (toward the
noise endpoint). After training, a single forward pass of :math:`f` produces
a sample, so no ODE integration is required at inference; the price is a
one-step approximation that introduces some variance.


Stochastic flow matching
------------------------

Stochastic flow matching follows the stochastic-interpolants framework of
Albergo et al. and replaces the deterministic interpolant with
:math:`X_t = I_t + \gamma(t)\,Z` for :math:`Z \sim \mathcal{N}(0, I)` and a
noise schedule :math:`\gamma(t)` that vanishes at the endpoints. The
velocity target picks up an extra :math:`\dot\gamma(t)\,Z` term; the field
and the sampling process are otherwise the standard flow-matching pair.
The added noise is best understood as a training-time regulariser rather
than as a different family of model.

.. note:: 
   A planned extension will include antithetic sampling capabilities.

Diffusion
---------

Diffusion-style models define a forward noising process
:math:`q(x_t \mid x_0) = \mathcal{N}(\alpha_t\,x_0,\;\sigma_t^2 I)` via a
schedule and train a network to reverse it. nami's
:class:`~nami.Diffusion` is a sampling-only wrapper: it accepts a model that
predicts :math:`\varepsilon`, the score, or :math:`x_0`, plus a schedule
(:class:`~nami.VPSchedule`, :class:`~nami.VESchedule`,
:class:`~nami.EDMSchedule`) and a solver, and turns these into the
reverse-time SDE or the probability-flow ODE. Training uses an external
denoising objective; the parameterisation flag tells the sampler how to
interpret the model output. See :doc:`parameterizations` for why several
parameterisations exist.

Generator matching
------------------

Generator matching takes the most general view: rather than committing to a
velocity, a score, or a noise prediction up front, learn the *generator*
:math:`L_t` of a continuous-time Markov process directly. A field
:math:`F_\theta(x, t)` predicts operator parameters, and an operator object
interprets them through a linear pairing
:math:`(L_t f)(x) = \langle K f(x), F_t(x) \rangle`. The built-in
:class:`~nami.ItoGeneratorOperator` covers continuous diffusion generators
in two modes — drift-only (``diffusion="none"``, ODE sampling) and
drift-plus-diagonal-diffusion (``diffusion="diagonal"``, SDE sampling).

This framing recovers the other families as special cases. A linear path
with a drift-only operator is deterministic flow matching. A Brownian-bridge
path with a drift-only operator is stochastic flow matching. A
Brownian-bridge path with a diagonal-diffusion operator is a full Itô
diffusion model. The point of the framework is twofold. It provides the
language in which "what is the difference between these methods?" has a
precise answer: a path, an operator, a parameterisation, a loss. And it
provides an extension point: new operators — non-diagonal diffusion, jump
processes, anything expressible as a linear pairing with a basis of
derivatives — drop in without rewriting the loss API or the process
protocol. Most everyday problems do not need this generality, and flow
matching has nicer training dynamics for the cases that fit it; but the
framework is there for the cases that do.

Schrödinger bridge matching
---------------------------

Bridge matching is the fifth family. Where flow matching and diffusion both
transport a *fixed* source distribution onto the target, bridge matching
learns the dynamics of a stochastic bridge between two distributions whose
endpoints are coupled — most naturally, a forward diffusion bridge that
respects a given coupling, in the spirit of the static Schrödinger bridge
problem. It is the right family when the relationship between source and
target is not "noise to data" but a genuine joint distribution: paired
samples, optimal-transport couplings, or domain-translation tasks where
both endpoints carry structure.

In nami the workflow trains separate velocity and score heads on
Brownian-bridge path samples. The velocity head regresses the bridge's
conditional drift; the score head regresses :math:`\nabla \log p_t` on the
same path. At sampling time, :class:`~nami.DriftFromVelocityScore` combines
the two into an SDE drift, which then feeds a standard
:class:`~nami.FlowMatching` or :class:`~nami.Diffusion` process; the bridge
itself does not need a separate runtime class. This follows the
bridge-matching approach of Tong et al. and is closely related to the
diffusion Schrödinger bridge matching of Shi et al. and to Peluchetti's
bridge-mixture transports.

What bridge matching makes visible — and what the other families hide — is
that the *coupling* between source and target is a modelling choice in its
own right. Flow matching and diffusion implicitly assume an independent
coupling :math:`(x_{\mathrm{source}}, x_{\mathrm{target}}) \sim
p_{\mathrm{source}} \otimes p_{\mathrm{target}}`; bridge matching lets the
joint be whatever the problem actually has. The same primitives —
interpolants, schedules, solvers — apply unchanged.

What actually differs
---------------------

Across the families, four things actually vary. The path
(linear, Brownian bridge, schedule-induced) sets the geometry of the
intermediate marginals. What the field predicts (velocity, score, eps, x0,
operator parameters) sets the parameterisation. The loss
(regression-of-conditional-target, self-consistency, joint regression for
bridges, operator-pairing for generator matching) sets *how* the prediction
is identified from data. And what the process does at inference time
(integrate an ODE, integrate an SDE, evaluate a one-step consistency
function) sets the runtime cost and the kind of output. Everything else —
the distribution interface, the solver protocol, the lazy-binding
mechanism, the divergence estimators — is shared. If you find yourself
writing custom plumbing to switch between two families, that is usually a
sign that something belongs in one of these four axes instead.

The loss axis deserves emphasis because it is the easiest to overlook: a
consistency model and a flow-matching model can predict *the same*
velocity, trained against *different* objectives, and produce very
different sampling behaviour. Many of the interesting research directions
in this space live on the loss axis rather than the field-output axis.

See also
--------

- :doc:`core-abstractions` — the primitives all the families share.
- :doc:`parameterizations` — the second axis: what the field predicts.
- :doc:`numerical-considerations` — the third axis: how the process
  integrates.
