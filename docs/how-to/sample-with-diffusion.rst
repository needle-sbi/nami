Sample with a diffusion model
=============================

Wrap a pretrained denoising model in :class:`~nami.Diffusion` to get a
sampler. nami handles the reverse-time SDE or probability-flow ODE; you
supply the model and a schedule. Training uses your own external denoising
objective.

Choose the schedule (:class:`~nami.VPSchedule`, :class:`~nami.VESchedule`,
:class:`~nami.EDMSchedule`) to match how the model was trained, and set
``parameterization`` to whichever of :func:`~nami.epsilon_prediction`,
:func:`~nami.score_prediction`, or :func:`~nami.x0_prediction` the model
predicts. A parameterisation mismatch produces garbage samples.

.. code-block:: python

   import nami

   model = nami.VelocityField(dim=8)
   schedule = nami.VPSchedule(beta_min=0.1, beta_max=20.0)
   diffusion = nami.Diffusion(
       model=model,
       schedule=schedule,
       solver=nami.EulerMaruyama(steps=100),
       parameterization=nami.epsilon_prediction(schedule=schedule),
       event_shape=(8,),
   )
   samples = diffusion(None).sample((64,))

ODE solvers (:class:`~nami.RK4`, :class:`~nami.Heun`,
:class:`~nami.DPMSolverPP`) integrate the probability-flow ODE.
:class:`~nami.EulerMaruyama` integrates the reverse SDE. The SDE sampler
generally produces higher quality at more compute; for few-step generation
without a consistency model, :class:`~nami.DPMSolverPP` is the right
choice.
