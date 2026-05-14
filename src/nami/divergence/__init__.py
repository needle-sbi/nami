"""Divergence estimators for change-of-variables log-density.

Exact trace (loop over event dimensions) for small problems;
Hutchinson stochastic-trace estimator for high-dimensional cases.

References
----------
- Hutchinson, *A stochastic estimator of the trace of the influence
  matrix for Laplacian smoothing splines*, 1989 — Hutchinson trace
  estimator.
- Grathwohl et al., *FFJORD: Free-form Continuous Dynamics for
  Scalable Reversible Generative Models*, 2018 (arXiv:1810.01367) —
  Hutchinson trace integrated with continuous normalising flows.
- Chen et al., *Neural ODE*, 2018 (arXiv:1806.07366).
"""

from __future__ import annotations



