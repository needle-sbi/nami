.. image:: assets/nami_logo.svg
   :class: only-light landing-banner

.. image:: assets/nami_logo.svg
   :class: only-dark landing-banner


.. rst-class:: tagline

Primitives for transport-map models

`nami <https://github.com/LeviSamuelEvans/nami>`_ is a Python package for flow matching, consistency flow matching,
stochastic interpolants, diffusion-style models, and generator matching in
PyTorch. It is built around small, composable abstractions such as fields,
paths, losses, schedules, solvers, and processes with which to build
transport maps.

.. container:: guide-grid

   .. container:: guide-section

      **Learn**

      .. container:: guide-links

         :doc:`Overview <overview>`

         Core concepts, tensor layout, and lazy binding.

         :doc:`Quickstart <quickstart>`

         Get up and running in minutes.

         :doc:`Models <models>`

         All supported transport-map models.

         :doc:`Experiments <experiments>`

         Interactive notebooks on toy and applied problems.

   .. container:: guide-section

      **Reference**

      .. container:: guide-links

         :doc:`Components <components>`

         Component catalog with formal definitions.

         :doc:`Examples <examples>`

         Code recipes for every workflow.

         :doc:`API Reference <api/index>`

         Full module-level API docs.

The project is intentionally modest in scope. It is meant to collect related
models behind a shared API, keep the dependency surface small, and make it easy
to explore new objectives and applications when combined with toher libraries and orchestration layers. It is not trying to be a
production framework.

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
   x_target = torch.randn(32, 8)          # data
   x_source = torch.randn_like(x_target)  # noise

   # train: regress the conditional velocity target
   loss = nami.fm_loss(field, x_target, x_source)
   loss.backward()

   # sample: configure a process, bind context, integrate
   fm = nami.FlowMatching(field, 
                          nami.StandardNormal((8,)), 
                          nami.RK4(steps=50)
                          )
   samples = fm().sample((64,))

.. warning::

   Nami uses one time convention across all workflows:
   :math:`t=0 \leftrightarrow` data and :math:`t=1 \leftrightarrow`
   source / noise. Sampling therefore runs from :math:`t=1` to
   :math:`t=0`. This differs from conventions used in the literature, 
   but is a library-level choice to keep the API consistent across all workflows.

The same configure-then-bind pattern extends to consistency flow matching,
diffusion, and generator matching: swap the loss, field, and solver while
keeping the rest of the code unchanged.

For more examples, see the :doc:`quickstart`.

.. card:: Fun fact
   :class-card: border-info

   **Nami** is named after the Japanese word for wave (なみ) and also the character "Nami" in the anime "One Piece", where she is the navigator of the crew.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: nami

   overview
   quickstart
   models
   components
   examples
   experiments
   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   Source <https://github.com/LeviSamuelEvans/nami>
   Issues <https://github.com/LeviSamuelEvans/nami/issues>