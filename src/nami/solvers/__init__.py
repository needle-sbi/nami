"""ODE / SDE solvers consumed by Process classes.

Fixed-step integrators with an ``integrate(f, x0, *, t0, t1, ...)``
signature; ODE solvers additionally provide ``integrate_augmented`` for
joint state + log-density propagation; SDE solvers expose
``is_sde = True`` for dispatch.

References
----------
- Karras et al., *EDM*, 2022 (arXiv:2206.00364) — Heun stochastic sampler.
- Lu et al., *DPM-Solver*, 2022 (arXiv:2206.00927).
- Lu et al., *DPM-Solver++*, 2022 (arXiv:2211.01095).
- Chen et al., *Neural ODE*, 2018 (arXiv:1806.07366) — augmented-state log-density.
"""

from __future__ import annotations

from nami.solvers.dpm import DPMSolverPP
from nami.solvers.heun import Heun
from nami.solvers.ode import RK4
from nami.solvers.sde import EulerMaruyama

__all__ = ["RK4", "DPMSolverPP", "EulerMaruyama", "Heun"]
