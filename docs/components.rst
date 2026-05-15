:orphan:

Components
==========

Distributions
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Definition
   * - :class:`~nami.StandardNormal`
     - :math:`\mathcal{N}(0, I_d)` with explicit ``event_shape``.
   * - :class:`~nami.DiagonalNormal`
     - :math:`\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))`.

Solvers
-------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Component
     - Type
     - Definition
   * - :class:`~nami.RK4`
     - ODE
     - Classical fourth-order Runge--Kutta. Fixed step, :math:`\mathcal{O}(h^4)` local error.
   * - :class:`~nami.Heun`
     - ODE
     - Explicit trapezoidal (predictor--corrector). Fixed step, :math:`\mathcal{O}(h^2)` local error.
   * - :class:`~nami.DPMSolverPP`
     - ODE
     - DPM-Solver++ for diffusion probability paths. Adapts step schedule to the noise schedule.
   * - :class:`~nami.EulerMaruyama`
     - SDE
     - :math:`X_{k+1} = X_k + f_k \,\Delta t + g_k \sqrt{|\Delta t|}\, Z_k`, :math:`\;Z_k \sim \mathcal{N}(0, I)`. Strong order :math:`\tfrac{1}{2}`.

Diffusion schedules
-------------------

Each schedule defines a forward process :math:`q(x_t \mid x_0) = \mathcal{N}(\alpha_t\, x_0,\; \sigma_t^2 I)` via signal and noise coefficients.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Definition
   * - :class:`~nami.VPSchedule`
     - Variance-preserving: :math:`\alpha_t^2 + \sigma_t^2 = 1`, linear :math:`\beta`-schedule with :math:`\beta_{\min}, \beta_{\max}` [1]_.
   * - :class:`~nami.VESchedule`
     - Variance-exploding: :math:`\alpha_t = 1`, :math:`\sigma_t` grows geometrically [2]_.
   * - :class:`~nami.EDMSchedule`
     - Karras et al. schedule with analytic :math:`c_{\mathrm{skip}}`, :math:`c_{\mathrm{out}}`, :math:`c_{\mathrm{in}}` scaling [3]_.

Flow-matching paths
-------------------

A probability path :math:`p_t` interpolates between :math:`p_0 = p_{\mathrm{source}}` and :math:`p_1 = p_{\mathrm{data}}`. Each interpolant defines a conditional sample :math:`X_t` and a conditional velocity target :math:`u_t`.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Component
     - Definition
   * - :class:`~nami.LinearInterpolant`
     - :math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}`, :math:`\;u_t = x_{\mathrm{data}} - x_{\mathrm{noise}}`. Constant velocity.
   * - :class:`~nami.CosineInterpolant`
     - :math:`X_t = \cos(\tfrac{\pi}{2}t)\,x_{\mathrm{noise}} + \sin(\tfrac{\pi}{2}t)\,x_{\mathrm{data}}`. Smooth at endpoints; :math:`u_t = \dot{\alpha}_t\,x_{\mathrm{noise}} + \dot{\beta}_t\,x_{\mathrm{data}}`.

Stochastic FM gamma schedules
------------------------------

The stochastic interpolant :math:`X_t = I_t + \gamma(t)\,Z` adds noise :math:`Z \sim \mathcal{N}(0, I)` to a deterministic interpolant :math:`I_t`, following Albergo et al. [4]_ [5]_.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Definition
   * - :class:`~nami.ZeroGamma`
     - :math:`\gamma(t) = 0`. Recovers deterministic FM (idempotent with :func:`~nami.regression_loss` under :class:`~nami.LinearInterpolant`).
   * - :class:`~nami.BrownianGamma`
     - :math:`\gamma(t) = \sqrt{t(1-t)}`. Brownian bridge variance.
   * - :class:`~nami.ScaledBrownianGamma`
     - :math:`\gamma(t) = \sqrt{s \cdot t(1-t)}` for scale :math:`s > 0`.

Losses
------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Definition
   * - :func:`~nami.regression_loss`
     - :math:`\mathcal{L} = \mathbb{E}_{t, x_{\mathrm{noise}}, x_{\mathrm{data}}}\!\bigl[\|\text{out\_transform}(F_\theta(X_t, t)) - T_t\|^2\bigr]`. Unified weighted MSE; specialises to deterministic FM with :class:`~nami.LinearInterpolant` + :func:`~nami.velocity_prediction`.
   * - :func:`~nami.stochastic_fm_loss`
     - Same objective with stochastic interpolant: :math:`X_t = I_t + \gamma(t) Z`, target :math:`u_t + \dot\gamma(t) Z`.
   * - :func:`~nami.consistency_loss`
     - Self-consistency MSE between two trajectory points; ``target_time=1.0`` for forward (data anchor), ``target_time=0.0`` for reverse (noise anchor).
   * - :func:`~nami.regression_loss` + :func:`~nami.generator_prediction`
     - :math:`\mathcal{L} = \mathbb{E}\!\bigl[\|F_\theta(X_t, t) - F_t(X_t \mid Z)\|^2\bigr]`. Regresses conditional generator parameters.

Generator matching
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Definition
   * - :class:`~nami.GeneratorMatching`
     - Lazy process: field + operator + solver :math:`\to` ODE/SDE sampler.
   * - :class:`~nami.ItoGeneratorOperator`
     - Itô generator :math:`(L_t f)(x) = b_t(x)^\top \nabla f + \tfrac{1}{2}\operatorname{tr}(a_t \nabla^2 f)`. Modes: ``"none"`` (drift only, ODE) or ``"diagonal"`` (drift + diagonal diffusion, SDE).
   * - :class:`~nami.LinearInterpolant`
     - Deterministic interpolant: :math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}`. Target drift :math:`b_t = x_{\mathrm{data}} - x_{\mathrm{noise}}`.
   * - :class:`~nami.BrownianBridgeInterpolant`
     - Brownian bridge: :math:`X_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}} + \sigma\sqrt{t(1-t)}\,Z`. Analytic conditional drift.
   * - :class:`~nami.GeneratorField`
     - MLP predicting operator parameters. Signature: ``forward(x, t, c=None)``.

Parameterisation transforms
---------------------------

Transforms between velocity :math:`v`, score :math:`\nabla \log p`,
interpolant parameterisations, and SDE drift :math:`f`.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Component
     - Definition
   * - :class:`~nami.ScoreFromEta`
     - :math:`\nabla \log p_t(x) = \eta_\theta(x, t) / \gamma(t)` for an interpolant parameterisation with :math:`\eta_\theta = \gamma(t)\,\nabla \log p_t`.
   * - :class:`~nami.ScoreFromRawNoise`
     - :math:`\nabla \log p_t(x) = -z_\theta(x, t) / \gamma(t)` when the model predicts the raw additive Gaussian noise in :math:`X_t = I_t + \gamma(t)\,z`.
   * - :class:`~nami.DriftFromVelocityScore`
     - :math:`f_t(x) = v_t(x) - \gamma(t)\gamma'(t)\nabla \log p_t(x)`.
   * - :class:`~nami.MirrorVelocityFromScore`
     - :math:`m_t(x) = \gamma(t)\gamma'(t)\nabla\log p_t(x)`. Mirror correction term used in backward-SDE formulas.

.. note::

   These wrappers are for stochastic-interpolant style parameterisations.
   They are not the same as diffusion :math:`\varepsilon`-prediction. For
   diffusion models, the corresponding relation is
   :math:`\nabla \log p_t(x) = -\varepsilon_\theta(x, t) / \sigma(t)`, which
   is handled by the diffusion-specific conversions in :mod:`nami.diffusion`
   and by :class:`~nami.Diffusion`.

Divergence estimators
---------------------

Required for continuous normalising flow log-likelihood :math:`\log p(x) = \log p_1(z) + \int_0^1 \nabla \cdot v_t \,\mathrm{d}t`.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Definition
   * - :class:`~nami.ExactDivergence`
     - Full Jacobian trace :math:`\operatorname{tr}(\partial v / \partial x)`. Cost :math:`\mathcal{O}(d)` backward passes.
   * - :class:`~nami.HutchinsonDivergence`
     - Stochastic trace estimator: :math:`\operatorname{tr}(J) = \mathbb{E}_\epsilon[\epsilon^\top J \epsilon]` for :math:`\epsilon \sim \mathcal{N}(0, I)` [6]_. Unbiased, single backward pass.

References
----------

.. [1] Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.

.. [2] Song et al., *Score-Based Generative Modeling through Stochastic
       Differential Equations*, ICLR 2021.

.. [3] Karras et al., *Elucidating the Design Space of Diffusion-Based
       Generative Models*, 2022.

.. [4] Albergo, M. S. and Vanden-Eijnden, E., *Building Normalizing Flows with
       Stochastic Interpolants*, ICLR 2023.

.. [5] Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E., *Stochastic
       Interpolants: A Unifying Framework for Flows and Diffusions*, 2023.

.. [6] Hutchinson, M. F., *A Stochastic Estimator of the Trace of the
       Influence Matrix for Laplacian Smoothing Splines*, 1990.
