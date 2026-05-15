.. image:: assets/nami_logo.svg
   :class: only-light landing-banner

.. image:: assets/nami_logo.svg
   :class: only-dark landing-banner


.. rst-class:: tagline

Composable transport maps

`nami <https://github.com/LeviSamuelEvans/nami>`_ is a Python package for flow matching, consistency flow matching,
stochastic interpolants, diffusion-style models, and generator matching in
PyTorch. It is built around composable abstractions such as fields,
paths, losses, schedules, solvers, and processes with which to build
transport maps. 

`ONGOING: lightning module examples and integrations with other libraries.`

.. container:: guide-grid

   .. container:: guide-section

      **Learn**

      .. container:: guide-links

         :doc:`Core Principles <principles/index>`

         A brief overview of the core abstractions, parameterisations, lazy binding,
         model families, and numerical choices.

         :doc:`How-to guides <how-to/index>`

         Quick snippets and recipes for training and sampling.

         :doc:`Tutorial notebooks <books/tutorials/index>`

         Guided walkthroughs in Jupyter. (UNDERGOING RE-WRITE)

   .. container:: guide-section

      **Reference**

      .. container:: guide-links

         :doc:`API Reference <api/index>`

         Full module-level API docs.

         :doc:`Experiment notebooks <books/experiment/index>`

         Applied experiments and toy problems.

         :doc:`External notebooks <books/external/index>`

         Notebooks ported from external sources.

To support conditional and reusable workflows, :mod:`nami` separates lazy binding
from runnable processes. :class:`nami.lazy.LazyDistribution` keeps source
distributions bindable when they depend on context or trainable parameters, while
:class:`nami.lazy.LazyProcess` lets flow matching, diffusion, and generator
matching wrappers be configured once and then bound to a concrete context at run
time. Please find some installation instructions below and first steps to getting started.

Installation
------------

For local development, install the project from the repository root with ``pixi``.

.. code-block:: console

   pixi run setup

If you prefer a plain editable install instead:

.. code-block:: console

   pip install -e .

The package will soon be available on PyPI, but for now it is only available from the repository.

Getting started
---------------

Every workflow is built from the same small pieces: a field, a loss, a
solver, and a source distribution, composed in two phases:

1. **Train**: sample a mini-batch, compute a regression loss, and backprop.
2. **Sample**: configure a lazy process from its components, bind
   optional context, and draw samples by integrating the learned field.

.. code-block:: python

   import torch
   import nami

   field = nami.VelocityField(dim=8)
   x_data = torch.randn(32, 8)          # data
   x_noise = torch.randn_like(x_data)   # noise

   # train: regress the conditional velocity target
   loss = nami.regression_loss(
       field,
       x_noise=x_noise, x_data=x_data,
       interpolant=nami.LinearInterpolant(),
       parameterization=nami.velocity_prediction(),
       eps_t=0.0,
   )
   loss.backward()

   # sample: configure a process, bind context, integrate
   fm = nami.FlowMatching(field,
                          nami.StandardNormal((8,)),
                          nami.RK4(steps=50)
                          )
   samples = fm().sample((64,))

The same configure-then-bind pattern extends to consistency flow matching,
diffusion, and generator matching: swap the loss, field, and solver while
keeping the rest of the code unchanged.

For recipes covering the most common workflows, see the
:doc:`how-to guides <how-to/index>`. For a guided walkthrough in Jupyter,
see the :doc:`tutorial notebooks <books/tutorials/index>`.

.. .. card:: Fun fact
..    :class-card: border-info

..    

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: nami

   principles/index
   how-to/index
   api/index
   books/tutorials/index
   books/experiment/index
   books/toys/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   Source <https://github.com/LeviSamuelEvans/nami>
   Issues <https://github.com/LeviSamuelEvans/nami/issues>
